"""Scikit-learn pipeline candidates and task-aware evaluation."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any

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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from torch_like.automl.specs import SearchSpec, SearchStrategy, TaskType

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _algorithm_grids(task_type: TaskType) -> dict[str, list[dict[str, Any]]]:
    if task_type in CLASSIFICATION_TASKS:
        grids = {
            "LogisticRegression": [{"C": value} for value in (0.1, 1.0, 10.0)],
            "RandomForestClassifier": [{"n_estimators": 100, "max_depth": depth} for depth in (None, 12)],
            "ExtraTreesClassifier": [{"n_estimators": 100, "max_depth": None}],
            "HistGradientBoostingClassifier": [{"learning_rate": value} for value in (0.05, 0.1)],
        }
        if _lightgbm_available():
            grids["LightGBMClassifier"] = [
                {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
                {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15},
            ]
        return grids
    if task_type in REGRESSION_TASKS:
        grids = {
            "Ridge": [{"alpha": value} for value in (0.1, 1.0, 10.0)],
            "RandomForestRegressor": [{"n_estimators": 100, "max_depth": depth} for depth in (None, 12)],
            "ExtraTreesRegressor": [{"n_estimators": 100, "max_depth": None}],
            "HistGradientBoostingRegressor": [{"learning_rate": value} for value in (0.05, 0.1)],
        }
        if _lightgbm_available():
            grids["LightGBMRegressor"] = [
                {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
                {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15},
            ]
        return grids
    raise ValueError(f"No first-stage sklearn algorithms for task {task_type.value!r}")


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

        baseline_grids = {name: list(grid) for name, grid in ANOMALY_BASELINE_GRIDS.items()}
        torch_grids = discover_detection_model_grids(include_experimental=search.algorithms is not None)
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
        if search.strategy == SearchStrategy.NONE:
            candidates = candidates[:1]
        elif search.strategy == SearchStrategy.RANDOM and len(candidates) > search.max_trials:
            rng = np.random.default_rng(seed)
            selected = sorted(rng.choice(len(candidates), size=search.max_trials, replace=False).tolist())
            candidates = [candidates[index] for index in selected]
        return candidates[: search.max_trials]
    grids = _algorithm_grids(task_type)
    if task_type == TaskType.TIME_SERIES_FORECASTING:
        grids = {"Ridge": grids["Ridge"]}
    if search.algorithms is not None and set(search.algorithms) & LIGHTGBM_CANDIDATES and not _lightgbm_available():
        raise ImportError("LightGBM candidates require: pip install 'torch_like[lightgbm]'")
    torch_grids: dict[str, list[dict[str, Any]]] = {}
    baseline_grids: dict[str, list[dict[str, Any]]] = {}
    if is_sequence and equal_sequence_lengths and task_type == TaskType.TIME_SERIES_CLASSIFICATION:
        torch_grids = {
            "InceptionTime": [{"d_model": 32, "num_blocks": 3}],
            "ROCKET": [{"num_kernels": 128}],
        }
    elif is_sequence and equal_sequence_lengths and task_type == TaskType.RUL_REGRESSION:
        torch_grids = {
            "THGNN": [{"d_model": 32, "graph_depth": 1}],
            "InformerRUL": [{"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64}],
            "AutoformerRUL": [{"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64}],
            "LogTransRUL": [{"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64}],
            "TCNRUL": [{"d_model": 32, "e_layers": 2, "d_ff": 64, "kernel_size": 5}],
            "PatchTSTRUL": [
                {"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64, "patch_len": 16, "patch_stride": 8}
            ],
            "TransformerRUL": [{"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64}],
        }
    elif is_sequence and equal_sequence_lengths and task_type == TaskType.TIME_SERIES_FORECASTING:
        baseline_grids = {"LastValue": [{}]}
        torch_grids = {
            "DLinear": [{"moving_avg": 25, "individual": True, "use_norm": True}],
            "PatchTST": [{"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64, "patch_len": 16}],
            "TSMixer": [{"e_layers": 2, "d_ff": 64}],
        }
    if search.algorithms is not None:
        available = set(grids) | set(torch_grids) | set(baseline_grids)
        unknown = set(search.algorithms) - available
        if unknown:
            raise ValueError(f"Unsupported algorithms for {task_type.value}: {sorted(unknown)}")
        grids = {name: grids[name] for name in search.algorithms if name in grids}
        baseline_grids = {name: baseline_grids[name] for name in search.algorithms if name in baseline_grids}
        torch_grids = {name: torch_grids[name] for name in search.algorithms if name in torch_grids}
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
    if search.strategy == SearchStrategy.NONE:
        candidates = candidates[:1]
    elif search.strategy == SearchStrategy.RANDOM and len(candidates) > search.max_trials:
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(len(candidates), size=search.max_trials, replace=False).tolist())
        candidates = [candidates[index] for index in selected]
    return candidates[: search.max_trials]


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
        return MultiOutputClassifier(model, n_jobs=workers) if multilabel else model
    if candidate.name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=seed, n_jobs=workers, **params)
    if candidate.name == "ExtraTreesClassifier":
        return ExtraTreesClassifier(random_state=seed, n_jobs=workers, **params)
    if candidate.name == "HistGradientBoostingClassifier":
        model = HistGradientBoostingClassifier(random_state=seed, **params)
        return MultiOutputClassifier(model, n_jobs=workers) if multilabel else model
    if candidate.name == "LightGBMClassifier":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(random_state=seed, n_jobs=workers, verbosity=-1, **params)
        return MultiOutputClassifier(model, n_jobs=workers) if multilabel else model
    if candidate.name == "Ridge":
        return Ridge(**params)
    if candidate.name == "RandomForestRegressor":
        return RandomForestRegressor(random_state=seed, n_jobs=workers, **params)
    if candidate.name == "ExtraTreesRegressor":
        return ExtraTreesRegressor(random_state=seed, n_jobs=workers, **params)
    if candidate.name == "HistGradientBoostingRegressor":
        model = HistGradientBoostingRegressor(random_state=seed, **params)
        return MultiOutputRegressor(model, n_jobs=workers) if multi_target else model
    if candidate.name == "LightGBMRegressor":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(random_state=seed, n_jobs=workers, verbosity=-1, **params)
        return MultiOutputRegressor(model, n_jobs=workers) if multi_target else model
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
    return (
        "minimize"
        if metric in {"mae", "rmse", "smape", "macro_nrmse", "weighted_nrmse", "false_alarm_rate"}
        or metric.endswith(("_mae", "_rmse"))
        else "maximize"
    )


def evaluate_predictions(
    task_type: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_weights: np.ndarray | None = None,
) -> dict[str, float]:
    if task_type in CLASSIFICATION_TASKS:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        if task_type == TaskType.MULTILABEL_CLASSIFICATION:
            metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        return metrics
    if task_type == TaskType.TIME_SERIES_FORECASTING:
        if y_true.shape != y_pred.shape or y_true.ndim != 3:
            raise ValueError("Forecasting targets and predictions must have shape (samples, horizon, targets)")
        y_true_2d = y_true.reshape(len(y_true), -1)
        y_pred_2d = y_pred.reshape(len(y_pred), -1)
    else:
        y_true_2d = y_true[:, None] if y_true.ndim == 1 else y_true
        y_pred_2d = y_pred[:, None] if y_pred.ndim == 1 else y_pred
    metrics = {
        "mae": float(mean_absolute_error(y_true_2d, y_pred_2d)),
        "rmse": float(mean_squared_error(y_true_2d, y_pred_2d) ** 0.5),
        "r2": float(r2_score(y_true_2d, y_pred_2d, multioutput="uniform_average")),
    }
    if task_type == TaskType.TIME_SERIES_FORECASTING:
        denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), np.finfo(np.float64).eps)
        metrics["smape"] = float(np.mean(2.0 * np.abs(y_pred - y_true) / denominator))
        for horizon in range(y_true.shape[1]):
            metrics[f"horizon_{horizon + 1}_mae"] = float(mean_absolute_error(y_true[:, horizon], y_pred[:, horizon]))
            metrics[f"horizon_{horizon + 1}_rmse"] = float(
                mean_squared_error(y_true[:, horizon], y_pred[:, horizon]) ** 0.5
            )
    if task_type == TaskType.MULTI_TARGET_REGRESSION:
        target_ranges = np.ptp(y_true_2d, axis=0)
        target_scales = np.where(target_ranges > 0, target_ranges, np.maximum(np.std(y_true_2d, axis=0), 1.0))
        target_rmse = np.sqrt(np.mean(np.square(y_true_2d - y_pred_2d), axis=0))
        normalized_rmse = target_rmse / target_scales
        metrics["macro_nrmse"] = float(np.mean(normalized_rmse))
        if target_weights is not None:
            normalized_weights = target_weights / target_weights.sum()
            metrics["weighted_nrmse"] = float(normalized_rmse @ normalized_weights)
        for index in range(y_true_2d.shape[1]):
            metrics[f"target_{index}_mae"] = float(mean_absolute_error(y_true_2d[:, index], y_pred_2d[:, index]))
            metrics[f"target_{index}_rmse"] = float(mean_squared_error(y_true_2d[:, index], y_pred_2d[:, index]) ** 0.5)
    return metrics


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
