"""Scikit-learn pipeline candidates and task-aware evaluation."""

from __future__ import annotations

import importlib.util
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from torch_like.automl.backends import get_backend, has_backend

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from torch_like.automl.candidate_registry import list_candidate_specs
from torch_like.automl.specs import CandidatePolicy, SearchSpec, SearchStrategy, TaskType

from torch_like.evaluation import (
    classification_metrics,
    forecasting_metrics,
    metric_direction as _shared_metric_direction,
    multi_target_regression_metrics,
    regression_metrics,
)

CLASSIFICATION_TASKS = {
    TaskType.TABULAR_CLASSIFICATION,
    TaskType.TIME_SERIES_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
}
REGRESSION_TASKS = {
    TaskType.TABULAR_REGRESSION,
    TaskType.RUL_REGRESSION,
    TaskType.MULTI_TARGET_REGRESSION,
    TaskType.TIME_SERIES_FORECASTING,
}
LIGHTGBM_CANDIDATES = {"LightGBMClassifier", "LightGBMRegressor"}


def _lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


class SequenceFeatureTransformer(BaseEstimator, TransformerMixin):
    """Convert variable-length sequences into fixed-size statistical features."""

    def __init__(self, strategy: str = "statistics"):
        self.strategy = strategy

    def fit(self, x, y=None):
        del y
        if self.strategy not in {"statistics", "vibration_statistics", "flatten"}:
            raise ValueError(f"Unsupported sequence feature strategy: {self.strategy}")
        lengths = [len(sequence) for sequence in x]
        if not lengths or min(lengths) <= 0:
            raise ValueError("Sequences must be non-empty")
        if self.strategy == "flatten" and len(set(lengths)) != 1:
            raise ValueError("flatten sequence features require equal sequence lengths")
        self.sequence_length_ = lengths[0] if len(set(lengths)) == 1 else None
        return self

    def transform(self, x):
        if self.strategy == "flatten":
            return np.stack([np.asarray(sequence, dtype=np.float64).reshape(-1) for sequence in x])
        rows = []
        for sequence in x:
            values = np.asarray(sequence, dtype=np.float64)
            positions = np.arange(len(values), dtype=np.float64)
            centered = positions - positions.mean()
            denominator = float(np.square(centered).sum())
            slope = np.zeros(values.shape[1]) if denominator == 0 else centered @ values / denominator
            basic = [
                np.nanmean(values, axis=0),
                np.nanstd(values, axis=0),
                np.nanmin(values, axis=0),
                np.nanmax(values, axis=0),
                np.nanmedian(values, axis=0),
                values[-1],
                slope,
            ]
            if self.strategy == "statistics":
                rows.append(np.concatenate(basic))
                continue

            eps = np.finfo(np.float64).eps
            mean = basic[0]
            std = basic[1]
            centered_values = values - mean
            absolute_values = np.abs(values)
            absolute_mean = np.nanmean(absolute_values, axis=0)
            rms = np.sqrt(np.nanmean(np.square(values), axis=0))
            peak = np.nanmax(absolute_values, axis=0)
            root_amplitude = np.square(np.nanmean(np.sqrt(absolute_values), axis=0))
            skewness = np.nanmean(np.power(centered_values, 3), axis=0) / np.maximum(std**3, eps)
            kurtosis = np.nanmean(np.power(centered_values, 4), axis=0) / np.maximum(std**4, eps)

            spectrum = np.abs(np.fft.rfft(np.nan_to_num(centered_values), axis=0))
            power = np.square(spectrum)
            power_sum = np.maximum(power.sum(axis=0), eps)
            frequencies = np.fft.rfftfreq(len(values))[:, None]
            spectral_centroid = (frequencies * power).sum(axis=0) / power_sum
            spectral_spread = np.sqrt((np.square(frequencies - spectral_centroid) * power).sum(axis=0) / power_sum)
            probabilities = power / power_sum
            spectral_entropy = -(probabilities * np.log(np.maximum(probabilities, eps))).sum(axis=0)
            spectral_entropy /= np.log(max(len(frequencies), 2))
            if len(frequencies) > 1:
                dominant_frequency = frequencies[1:][np.argmax(power[1:], axis=0), 0]
            else:
                dominant_frequency = np.zeros(values.shape[1])
            rows.append(
                np.concatenate(
                    [
                        *basic,
                        rms,
                        absolute_mean,
                        np.nanmax(values, axis=0) - np.nanmin(values, axis=0),
                        np.nanquantile(values, 0.25, axis=0),
                        np.nanquantile(values, 0.75, axis=0),
                        skewness,
                        kurtosis,
                        peak / np.maximum(rms, eps),
                        peak / np.maximum(absolute_mean, eps),
                        rms / np.maximum(absolute_mean, eps),
                        peak / np.maximum(root_amplitude, eps),
                        spectral_centroid,
                        spectral_spread,
                        dominant_frequency,
                        spectral_entropy,
                    ]
                )
            )
        return np.stack(rows)


class NamedFeatureFrameTransformer(BaseEstimator, TransformerMixin):
    """Give array features stable names for estimators that preserve schema."""

    def fit(self, x, y=None):
        del y
        values = np.asarray(x)
        if values.ndim != 2:
            raise ValueError("Named feature conversion expects a 2D array")
        self.feature_names_out_ = tuple(f"feature_{index}" for index in range(values.shape[1]))
        return self

    def transform(self, x):
        return pd.DataFrame(np.asarray(x), columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features=None):
        del input_features
        return np.asarray(self.feature_names_out_, dtype=object)


@dataclass(frozen=True)
class AlgorithmCandidate:
    name: str
    preprocessing: str
    parameters: dict[str, Any]
    sequence_features: str | None = None
    backend: str = "sklearn"

    @property
    def trial_name(self) -> str:
        feature = f"+{self.sequence_features}" if self.sequence_features else ""
        params = ",".join(f"{key}={value}" for key, value in sorted(self.parameters.items()))
        return f"{self.backend}:{feature}+{self.preprocessing}+{self.name}[{params}]".replace(":+", ":")

    @property
    def model_family(self) -> str:
        return get_backend(self.backend).model_family

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_candidate_policy(candidates: list[AlgorithmCandidate], search: SearchSpec) -> list[AlgorithmCandidate]:
    policy = search.candidate_policy
    if policy == CandidatePolicy.PRODUCTION:
        return candidates

    allowed_family = {
        CandidatePolicy.TORCH_LIKE_ONLY: "torch_like",
        CandidatePolicy.BASELINE_ONLY: "baseline",
    }.get(policy)
    if allowed_family is not None:
        rejected = sorted({candidate.name for candidate in candidates if candidate.model_family != allowed_family})
        if rejected and search.algorithms is not None:
            raise ValueError(f"candidate_policy={policy.value!r} rejects explicitly requested algorithms: {rejected}")
        return [candidate for candidate in candidates if candidate.model_family == allowed_family]

    # Stable family ordering lets budget truncation spend remaining trials on
    # torch_like variants while retaining baseline anchors for comparison.
    return sorted(candidates, key=lambda candidate: candidate.model_family != "torch_like")


def _limit_candidates(candidates: list[AlgorithmCandidate], search: SearchSpec, seed: int) -> list[AlgorithmCandidate]:
    if not candidates:
        return []
    if search.strategy == SearchStrategy.NONE:
        return candidates[:1]
    anchors: list[AlgorithmCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.name not in seen:
            anchors.append(candidate)
            seen.add(candidate.name)
    if search.max_trials < len(anchors):
        if search.algorithms is not None:
            raise ValueError(
                f"max_trials={search.max_trials} cannot cover all {len(anchors)} explicitly requested algorithms"
            )
        if search.candidate_policy == CandidatePolicy.PREFER_TORCH_LIKE and search.max_trials >= 2:
            torch_anchors = [candidate for candidate in anchors if candidate.model_family == "torch_like"]
            baseline_anchors = [candidate for candidate in anchors if candidate.model_family == "baseline"]
            if torch_anchors and baseline_anchors:
                return [*torch_anchors[: search.max_trials - 1], baseline_anchors[0]]
        return anchors[: search.max_trials]
    anchor_ids = {id(candidate) for candidate in anchors}
    variants = [candidate for candidate in candidates if id(candidate) not in anchor_ids]
    remaining = search.max_trials - len(anchors)
    if (
        search.strategy == SearchStrategy.RANDOM
        and search.candidate_policy == CandidatePolicy.PREFER_TORCH_LIKE
        and len(variants) > remaining
    ):
        rng = np.random.default_rng(seed)
        torch_variants = [candidate for candidate in variants if candidate.model_family == "torch_like"]
        baseline_variants = [candidate for candidate in variants if candidate.model_family == "baseline"]
        rng.shuffle(torch_variants)
        rng.shuffle(baseline_variants)
        variants = [*torch_variants, *baseline_variants]
    elif search.strategy == SearchStrategy.RANDOM and len(variants) > remaining:
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(len(variants), size=remaining, replace=False).tolist())
        variants = [variants[index] for index in selected]
    elif search.strategy == SearchStrategy.GRID and len(variants) > remaining:
        selected_variants: list[AlgorithmCandidate] = []
        preprocessing = {candidate.preprocessing for candidate in anchors}
        name_counts = Counter(candidate.name for candidate in anchors)
        pair_counts = Counter((candidate.name, candidate.preprocessing) for candidate in anchors)
        indexed_variants = list(enumerate(variants))
        while indexed_variants and len(selected_variants) < remaining:
            position, (_, candidate) = min(
                enumerate(indexed_variants),
                key=lambda item: (
                    item[1][1].preprocessing in preprocessing,
                    pair_counts[(item[1][1].name, item[1][1].preprocessing)],
                    name_counts[item[1][1].name],
                    item[1][0],
                ),
            )
            indexed_variants.pop(position)
            selected_variants.append(candidate)
            preprocessing.add(candidate.preprocessing)
            name_counts[candidate.name] += 1
            pair_counts[(candidate.name, candidate.preprocessing)] += 1
        variants = selected_variants
    return [*anchors, *variants[:remaining]]


def build_candidates(
    task_type: TaskType,
    search: SearchSpec,
    *,
    is_sequence: bool,
    equal_sequence_lengths: bool,
    seed: int,
) -> list[AlgorithmCandidate]:
    if task_type == TaskType.ANOMALY_DETECTION:
        from torch_like.automl.anomaly_catalog import ANOMALY_BASELINE_GRIDS, discover_detection_model_grids

        # Honour capability assembly: a suppressed anomaly backend contributes no candidates.
        baseline_grids = (
            {name: list(grid) for name, grid in ANOMALY_BASELINE_GRIDS.items()}
            if has_backend("anomaly_baseline")
            else {}
        )
        torch_grids = (
            discover_detection_model_grids(include_experimental=search.algorithms is not None)
            if has_backend("anomaly_pytorch")
            else {}
        )
        available = set(baseline_grids) | set(torch_grids)
        if search.algorithms is not None:
            unknown = set(search.algorithms) - available
            if unknown:
                raise ValueError(f"Unsupported algorithms for {task_type.value}: {sorted(unknown)}")
            baseline_grids = {name: baseline_grids[name] for name in search.algorithms if name in baseline_grids}
            torch_grids = {name: torch_grids[name] for name in search.algorithms if name in torch_grids}
        candidates: list[AlgorithmCandidate] = []

        # First allocate one trial to every requested algorithm. Additional
        # preprocessing/configuration variants are appended only afterwards.
        for name, parameter_grid in baseline_grids.items():
            preferred = "none" if name == "GlobalSTD" else search.preprocessing[0]
            candidates.append(AlgorithmCandidate(name, preferred, parameter_grid[0], backend="anomaly_baseline"))
        for name, parameter_grid in torch_grids.items():
            candidates.append(
                AlgorithmCandidate(name, search.preprocessing[0], parameter_grid[0], backend="anomaly_pytorch")
            )
        for name, parameter_grid in baseline_grids.items():
            preprocessing_choices = ("none",) if name == "GlobalSTD" else search.preprocessing
            candidates.extend(
                AlgorithmCandidate(name, preprocessing, parameters, backend="anomaly_baseline")
                for preprocessing in preprocessing_choices
                for parameters in parameter_grid
                if preprocessing != ("none" if name == "GlobalSTD" else search.preprocessing[0])
                or parameters != parameter_grid[0]
            )
        candidates.extend(
            AlgorithmCandidate(name, preprocessing, parameters, backend="anomaly_pytorch")
            for preprocessing in search.preprocessing
            for name, parameter_grid in torch_grids.items()
            for parameters in parameter_grid
            if preprocessing != search.preprocessing[0] or parameters != parameter_grid[0]
        )
        return _limit_candidates(_apply_candidate_policy(candidates, search), search, seed)

    specs = list(
        list_candidate_specs(
            task_type,
            is_sequence=is_sequence,
            equal_sequence_lengths=equal_sequence_lengths,
            include_experimental=search.algorithms is not None,
        )
    )
    if search.algorithms is not None:
        available = {spec.name for spec in specs}
        unknown = set(search.algorithms) - available
        if unknown:
            raise ValueError(f"Unsupported algorithms for {task_type.value}: {sorted(unknown)}")
        by_name = {spec.name: spec for spec in specs}
        specs = [by_name[name] for name in search.algorithms]
    unavailable = [spec for spec in specs if spec.optional_dependency == "lightgbm" and not _lightgbm_available()]
    if unavailable and search.algorithms is not None:
        raise ImportError("LightGBM candidates require: pip install 'torch_like[lightgbm]'")
    specs = [spec for spec in specs if spec not in unavailable]
    grids = {spec.name: list(spec.parameter_grid) for spec in specs if get_backend(spec.backend).kind == "sklearn"}
    baseline_grids = {spec.name: list(spec.parameter_grid) for spec in specs if get_backend(spec.backend).kind == "baseline"}
    torch_grids = {spec.name: list(spec.parameter_grid) for spec in specs if get_backend(spec.backend).kind == "torch"}
    feature_strategies: tuple[str | None, ...] = (None,)
    if task_type == TaskType.TIME_SERIES_FORECASTING:
        feature_strategies = ("flatten",)
    elif is_sequence:
        strategies = ["statistics"]
        if task_type == TaskType.RUL_REGRESSION:
            strategies.append("vibration_statistics")
        if equal_sequence_lengths:
            strategies.append("flatten")
        feature_strategies = tuple(strategies)
    candidates = [
        AlgorithmCandidate(name, preprocessing, parameters, feature_strategy)
        for feature_strategy in feature_strategies
        for preprocessing in search.preprocessing
        for name, parameter_grid in grids.items()
        for parameters in parameter_grid
    ]
    candidates.extend(
        AlgorithmCandidate(name, "none", parameters, backend="baseline")
        for name, parameter_grid in baseline_grids.items()
        for parameters in parameter_grid
    )
    candidates.extend(
        AlgorithmCandidate(name, preprocessing, parameters, backend="pytorch")
        for preprocessing in search.preprocessing
        for name, parameter_grid in torch_grids.items()
        for parameters in parameter_grid
    )
    return _limit_candidates(_apply_candidate_policy(candidates, search), search, seed)


def _scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "none":
        return "passthrough"
    raise ValueError(f"Unsupported preprocessing: {name}")


def _estimator(candidate: AlgorithmCandidate, task_type: TaskType, seed: int, workers: int):
    params = dict(candidate.parameters)
    multilabel = task_type == TaskType.MULTILABEL_CLASSIFICATION
    multi_target = task_type in {TaskType.MULTI_TARGET_REGRESSION, TaskType.TIME_SERIES_FORECASTING}
    if candidate.name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=seed, **params)
        return MultiOutputClassifier(model, n_jobs=1) if multilabel else model
    if candidate.name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=seed, n_jobs=workers, **params)
    if candidate.name == "ExtraTreesClassifier":
        return ExtraTreesClassifier(random_state=seed, n_jobs=workers, **params)
    if candidate.name == "HistGradientBoostingClassifier":
        model = HistGradientBoostingClassifier(random_state=seed, **params)
        return MultiOutputClassifier(model, n_jobs=1) if multilabel else model
    if candidate.name == "LightGBMClassifier":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(random_state=seed, n_jobs=workers, verbosity=-1, **params)
        return MultiOutputClassifier(model, n_jobs=1) if multilabel else model
    if candidate.name == "Ridge":
        return Ridge(**params)
    if candidate.name == "RandomForestRegressor":
        return RandomForestRegressor(random_state=seed, n_jobs=workers, **params)
    if candidate.name == "ExtraTreesRegressor":
        return ExtraTreesRegressor(random_state=seed, n_jobs=workers, **params)
    if candidate.name == "HistGradientBoostingRegressor":
        model = HistGradientBoostingRegressor(random_state=seed, **params)
        return MultiOutputRegressor(model, n_jobs=1) if multi_target else model
    if candidate.name == "LightGBMRegressor":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(random_state=seed, n_jobs=workers, verbosity=-1, **params)
        return MultiOutputRegressor(model, n_jobs=1) if multi_target else model
    raise ValueError(f"Unsupported algorithm: {candidate.name}")


def build_pipeline(
    candidate: AlgorithmCandidate,
    task_type: TaskType,
    *,
    seed: int,
    workers: int,
) -> Pipeline:
    if candidate.backend != "sklearn":
        raise ValueError(f"build_pipeline only accepts sklearn candidates, got {candidate.backend!r}")
    steps: list[tuple[str, Any]] = []
    if candidate.sequence_features:
        steps.append(("sequence_features", SequenceFeatureTransformer(candidate.sequence_features)))
    steps.extend(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", _scaler(candidate.preprocessing)),
        ]
    )
    if candidate.name in LIGHTGBM_CANDIDATES:
        steps.append(("feature_names", NamedFeatureFrameTransformer()))
    steps.append(("estimator", _estimator(candidate, task_type, seed, workers)))
    return Pipeline(steps)


def default_primary_metric(task_type: TaskType) -> str:
    if task_type == TaskType.ANOMALY_DETECTION:
        return "event_f0_5_corrected"
    if task_type in CLASSIFICATION_TASKS:
        return "f1_macro"
    if task_type == TaskType.MULTI_TARGET_REGRESSION:
        return "macro_nrmse"
    if task_type in REGRESSION_TASKS:
        return "rmse"
    raise ValueError(f"No default metric for task {task_type.value!r}")


def supported_metrics(task_type: TaskType) -> set[str]:
    if task_type == TaskType.ANOMALY_DETECTION:
        return {
            "point_precision",
            "point_recall",
            "point_f1",
            "false_alarm_rate",
            "event_precision_corrected",
            "event_recall",
            "event_f0_5_corrected",
            "alarming_precision",
            "channel_event_precision",
            "channel_event_recall",
            "channel_event_f0_5",
            "average_precision",
            "roc_auc",
        }
    if task_type in CLASSIFICATION_TASKS:
        metrics = {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
        if task_type == TaskType.MULTILABEL_CLASSIFICATION:
            metrics.add("f1_micro")
        return metrics
    metrics = {"mae", "rmse", "r2"}
    if task_type == TaskType.TIME_SERIES_FORECASTING:
        metrics.add("smape")
    if task_type == TaskType.MULTI_TARGET_REGRESSION:
        metrics.update({"macro_nrmse", "weighted_nrmse"})
    return metrics


def metric_direction(metric: str) -> str:
    """Return the shared optimization direction for one metric name."""

    return _shared_metric_direction(metric)


def evaluate_predictions(
    task_type: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate AutoML predictions through the shared evaluation primitives."""

    if task_type in CLASSIFICATION_TASKS:
        return classification_metrics(
            y_true,
            y_pred,
            multilabel=task_type == TaskType.MULTILABEL_CLASSIFICATION,
        )
    if task_type == TaskType.TIME_SERIES_FORECASTING:
        return forecasting_metrics(y_true, y_pred)
    if task_type == TaskType.MULTI_TARGET_REGRESSION:
        return multi_target_regression_metrics(y_true, y_pred, target_weights=target_weights)
    return regression_metrics(y_true, y_pred)


def estimate_parameter_count(pipeline: Pipeline) -> int | None:
    estimator = pipeline.named_steps["estimator"]
    estimators = getattr(estimator, "estimators_", None)
    if hasattr(estimator, "coef_"):
        return int(np.asarray(estimator.coef_).size + np.asarray(estimator.intercept_).size)
    if estimators is not None:
        flat = np.asarray(estimators, dtype=object).reshape(-1)
        tree_nodes = [getattr(getattr(item, "tree_", None), "node_count", 0) for item in flat]
        if any(tree_nodes):
            return int(sum(tree_nodes))
    return None
