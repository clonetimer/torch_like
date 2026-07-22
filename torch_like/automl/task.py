"""Auditable leaderboard-task planning and execution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from torch_like.automl.anomaly_backend import run_anomaly_candidate
from torch_like.automl.data import (
    DataSplit,
    GroupCVSplit,
    PreparedData,
    RollingOriginSplit,
    ValidationFold,
    prepare_dataset,
    split_dataset,
    split_group_cv,
    split_rolling_origin,
)
from torch_like.automl.protocol import EvaluationTrustAudit, ProtocolAudit, audit_evaluation_trust, audit_split_protocol
from torch_like.automl.sklearn_backend import (
    AlgorithmCandidate,
    build_candidates,
    build_pipeline,
    default_primary_metric,
    estimate_parameter_count,
    evaluate_predictions,
    metric_direction,
    supported_metrics,
)
from torch_like.automl.specs import LabelGranularity, LeaderboardConfig, SplitType, TaskType
from torch_like.automl.torch_backend import run_torch_candidate

SUPPORTED_TASKS = {
    TaskType.TABULAR_CLASSIFICATION,
    TaskType.TABULAR_REGRESSION,
    TaskType.TIME_SERIES_CLASSIFICATION,
    TaskType.RUL_REGRESSION,
    TaskType.TIME_SERIES_FORECASTING,
    TaskType.MULTILABEL_CLASSIFICATION,
    TaskType.MULTI_TARGET_REGRESSION,
    TaskType.ANOMALY_DETECTION,
}


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    name: str
    status: str
    duration_seconds: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    parameter_count: int | None
    artifact_dir: str
    error: str | None = None
    candidate_index: int | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeaderboardResult:
    run_dir: Path
    leaderboard: pd.DataFrame
    trials: tuple[TrialResult, ...]
    primary_metric: str

    @property
    def best_trial(self) -> TrialResult | None:
        completed = {trial.trial_id: trial for trial in self.trials if trial.status == "completed"}
        if self.leaderboard.empty:
            return None
        return completed.get(str(self.leaderboard.iloc[0]["trial_id"]))


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_frame(sample_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    payload: dict[str, Any] = {"sample_id": sample_ids}
    true_2d = y_true.reshape(len(y_true), -1) if y_true.ndim > 1 else y_true[:, None]
    pred_2d = y_pred.reshape(len(y_pred), -1) if y_pred.ndim > 1 else y_pred[:, None]
    for index in range(true_2d.shape[1]):
        payload[f"target_{index}"] = true_2d[:, index]
        payload[f"prediction_{index}"] = pred_2d[:, index]
    return pd.DataFrame(payload)


def _fit_targets(task_type: TaskType, targets: np.ndarray) -> np.ndarray:
    return targets.reshape(len(targets), -1) if task_type == TaskType.TIME_SERIES_FORECASTING else targets


def _reshape_predictions(task_type: TaskType, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return predictions.reshape(targets.shape) if task_type == TaskType.TIME_SERIES_FORECASTING else predictions


def _last_value_forecast(x: np.ndarray, target_indices: list[int], target_shape: tuple[int, ...]) -> np.ndarray:
    last_values = np.stack([np.asarray(sequence)[-1, target_indices] for sequence in x])
    return np.repeat(last_values[:, None, :], target_shape[1], axis=1)


def _write_sklearn_artifact(path: Path, pipeline, candidate: AlgorithmCandidate) -> None:
    joblib.dump(pipeline, path)
    optional_versions = {}
    if candidate.name in {"LightGBMClassifier", "LightGBMRegressor"}:
        optional_versions["lightgbm"] = importlib.metadata.version("lightgbm")
    _json_dump(
        path.parent / "artifact_manifest.json",
        {
            "backend": "sklearn",
            "format": "joblib",
            "trusted_local_only": True,
            "sha256": _file_sha256(path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            **optional_versions,
        },
    )


def _inner_group_split(fold: ValidationFold, groups: np.ndarray, seed: int) -> tuple[DataSplit, list[str]]:
    outer_train_groups = np.unique(groups[fold.train].astype(str))
    if len(outer_train_groups) < 2:
        raise ValueError("PyTorch group-CV training requires at least two outer-train groups for inner validation")
    shuffled = outer_train_groups.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    inner_val_groups = shuffled[:1]
    inner_val_mask = np.isin(groups[fold.train].astype(str), inner_val_groups)
    inner_train = fold.train[~inner_val_mask]
    inner_val = fold.train[inner_val_mask]
    if min(len(inner_train), len(inner_val), len(fold.val)) == 0:
        raise ValueError("Inner group validation produced an empty train, validation, or outer-validation set")
    return DataSplit(inner_train, inner_val, fold.val), inner_val_groups.tolist()


def _inner_time_split(fold: ValidationFold) -> DataSplit:
    inner_val_count = max(1, int(round(len(fold.train) * 0.2)))
    if inner_val_count >= len(fold.train):
        raise ValueError("Rolling-origin outer-train window is too small for inner validation")
    return DataSplit(fold.train[:-inner_val_count], fold.train[-inner_val_count:], fold.val)


def _trial_worker(payload: dict[str, Any], queue) -> None:
    started = time.monotonic()
    trial_dir = Path(payload["trial_dir"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    candidate: AlgorithmCandidate = payload["candidate"]
    try:
        x, y, split = payload["x"], payload["y"], payload["split"]
        if payload["task_type"] == TaskType.ANOMALY_DETECTION:
            anomaly_result = run_anomaly_candidate(
                candidate,
                x,
                y,
                split,
                sample_ids=payload["sample_ids"],
                channels=payload["feature_columns"],
                seed=payload["seed"],
                workers=payload["workers"],
                device=payload["device"],
                epochs=payload["torch_epochs"],
                batch_size=payload["torch_batch_size"],
                learning_rate=payload["torch_learning_rate"],
                optimizer_name=payload["torch_optimizer"],
                weight_decay=payload["torch_weight_decay"],
                grad_clip_norm=payload["torch_grad_clip_norm"],
                threshold_candidates=payload["anomaly_threshold_candidates"],
                trial_dir=trial_dir,
                save_checkpoint=payload["save_checkpoint"],
                save_predictions=payload["save_predictions"],
                validation_only=payload.get("anomaly_validation_only", False),
            )
            validation_metrics = anomaly_result.validation_metrics
            test_metrics = anomaly_result.test_metrics
            parameter_count = anomaly_result.parameter_count
            val_prediction = anomaly_result.validation_scores
            test_prediction = (
                anomaly_result.validation_scores if anomaly_result.test_scores is None else anomaly_result.test_scores
            )
        elif candidate.backend == "pytorch":
            validation_metrics, test_metrics, parameter_count, val_prediction, test_prediction = run_torch_candidate(
                candidate,
                payload["task_type"],
                x,
                y,
                split,
                seed=payload["seed"],
                epochs=payload["torch_epochs"],
                batch_size=payload["torch_batch_size"],
                learning_rate=payload["torch_learning_rate"],
                workers=payload["workers"],
                device=payload["device"],
                trial_dir=trial_dir,
                save_checkpoint=payload["save_checkpoint"],
                final_only=payload.get("final_only", False),
                optimizer_name=payload["torch_optimizer"],
                weight_decay=payload["torch_weight_decay"],
                loss_name=payload["torch_loss"],
                grad_clip_norm=payload["torch_grad_clip_norm"],
                lr_scheduler_name=payload["torch_lr_scheduler"],
                early_stopping_patience=payload["torch_early_stopping_patience"],
                min_delta=payload["torch_min_delta"],
                restore_best=payload["torch_restore_best"],
                target_constraint=payload["torch_target_constraint"],
            )
        elif candidate.backend == "baseline":
            final_only = payload.get("final_only", False)
            val_indices = split.test if final_only else split.val
            val_prediction = _last_value_forecast(
                x[val_indices], payload["forecast_target_indices"], y[val_indices].shape
            )
            test_prediction = (
                val_prediction
                if final_only
                else _last_value_forecast(x[split.test], payload["forecast_target_indices"], y[split.test].shape)
            )
            validation_metrics = (
                {}
                if final_only
                else evaluate_predictions(payload["task_type"], y[split.val], val_prediction, payload["target_weights"])
            )
            test_metrics = evaluate_predictions(
                payload["task_type"], y[split.test], test_prediction, payload["target_weights"]
            )
            parameter_count = 0
            _json_dump(
                trial_dir / "artifact_manifest.json",
                {"backend": "baseline", "name": candidate.name, "parameter_count": 0},
            )
        else:
            pipeline = build_pipeline(
                candidate,
                payload["task_type"],
                seed=payload["seed"],
                workers=payload["workers"],
            )
            pipeline.fit(x[split.train], _fit_targets(payload["task_type"], y[split.train]))
            final_only = payload.get("final_only", False)
            val_prediction = (
                np.asarray(pipeline.predict(x[split.test]))
                if final_only
                else np.asarray(pipeline.predict(x[split.val]))
            )
            test_prediction = val_prediction if final_only else np.asarray(pipeline.predict(x[split.test]))
            val_prediction = _reshape_predictions(
                payload["task_type"], val_prediction, y[split.test] if final_only else y[split.val]
            )
            test_prediction = _reshape_predictions(payload["task_type"], test_prediction, y[split.test])
            validation_metrics = (
                {}
                if final_only
                else evaluate_predictions(payload["task_type"], y[split.val], val_prediction, payload["target_weights"])
            )
            test_metrics = evaluate_predictions(
                payload["task_type"], y[split.test], test_prediction, payload["target_weights"]
            )
            parameter_count = estimate_parameter_count(pipeline)
            if payload["save_checkpoint"]:
                _write_sklearn_artifact(trial_dir / "pipeline.joblib", pipeline, candidate)
        if np.issubdtype(val_prediction.dtype, np.number) and not np.isfinite(val_prediction).all():
            raise FloatingPointError("Validation predictions contain NaN or infinite values")
        if np.issubdtype(test_prediction.dtype, np.number) and not np.isfinite(test_prediction).all():
            raise FloatingPointError("Test predictions contain NaN or infinite values")
        selection_only = payload.get("selection_only", False)
        if selection_only:
            test_metrics = {}
        if payload["save_predictions"] and payload["task_type"] != TaskType.ANOMALY_DETECTION:
            prediction_indices = split.val if selection_only else split.test
            prediction_values = val_prediction if selection_only else test_prediction
            prediction_name = "validation_predictions.csv" if selection_only else "predictions.csv"
            _prediction_frame(
                payload["sample_ids"][prediction_indices], y[prediction_indices], prediction_values
            ).to_csv(trial_dir / prediction_name, index=False)
        result = TrialResult(
            trial_id=payload["trial_id"],
            name=candidate.trial_name,
            status="completed",
            duration_seconds=time.monotonic() - started,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            parameter_count=parameter_count,
            artifact_dir=str(trial_dir),
            candidate_index=payload.get("candidate_index"),
            seed=payload["seed"],
        )
        _json_dump(trial_dir / "metrics.json", result.to_dict())
        _json_dump(trial_dir / "config.json", {"candidate": candidate.to_dict(), "seed": payload["seed"]})
        queue.put(result)
    except BaseException as exc:  # noqa: BLE001 - a failed trial must not terminate the search
        message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        lowered = str(exc).lower()
        status = (
            "oom"
            if isinstance(exc, MemoryError) or "out of memory" in lowered
            else "nan"
            if "nan" in lowered
            else "failed"
        )
        result = TrialResult(
            trial_id=payload["trial_id"],
            name=candidate.trial_name,
            status=status,
            duration_seconds=time.monotonic() - started,
            validation_metrics={},
            test_metrics={},
            parameter_count=None,
            artifact_dir=str(trial_dir),
            error=message,
            candidate_index=payload.get("candidate_index"),
            seed=payload["seed"],
        )
        _json_dump(trial_dir / "error.json", result.to_dict())
        queue.put(result)


def _cv_trial_worker(payload: dict[str, Any], queue) -> None:
    started = time.monotonic()
    trial_dir = Path(payload["trial_dir"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    candidate: AlgorithmCandidate = payload["candidate"]
    try:
        x, y = payload["x"], payload["y"]
        fold_metrics = []
        parameter_count = None
        for fold_index, fold in enumerate(payload["folds"]):
            fold_dir = trial_dir / f"fold-{fold_index:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_seed = payload["seed"] + fold_index
            if candidate.backend == "pytorch":
                if payload["cv_protocol"] == "group_cv":
                    fold_split, inner_val_groups = _inner_group_split(fold, payload["groups"], fold_seed)
                else:
                    fold_split = _inner_time_split(fold)
                    inner_val_groups = []
                inner_metrics, validation_metrics, parameter_count, _, val_prediction = run_torch_candidate(
                    candidate,
                    payload["task_type"],
                    x,
                    y,
                    fold_split,
                    seed=fold_seed,
                    epochs=payload["torch_epochs"],
                    batch_size=payload["torch_batch_size"],
                    learning_rate=payload["torch_learning_rate"],
                    workers=payload["workers"],
                    device=payload["device"],
                    trial_dir=fold_dir,
                    save_checkpoint=payload["save_checkpoint"],
                    optimizer_name=payload["torch_optimizer"],
                    weight_decay=payload["torch_weight_decay"],
                    loss_name=payload["torch_loss"],
                    grad_clip_norm=payload["torch_grad_clip_norm"],
                    lr_scheduler_name=payload["torch_lr_scheduler"],
                    early_stopping_patience=payload["torch_early_stopping_patience"],
                    min_delta=payload["torch_min_delta"],
                    restore_best=payload["torch_restore_best"],
                    target_constraint=payload["torch_target_constraint"],
                )
            elif candidate.backend == "baseline":
                inner_metrics = None
                inner_val_groups = []
                val_prediction = _last_value_forecast(
                    x[fold.val], payload["forecast_target_indices"], y[fold.val].shape
                )
                validation_metrics = evaluate_predictions(
                    payload["task_type"], y[fold.val], val_prediction, payload["target_weights"]
                )
                parameter_count = 0
                _json_dump(
                    fold_dir / "artifact_manifest.json",
                    {"backend": "baseline", "name": candidate.name, "parameter_count": 0},
                )
            else:
                inner_metrics = None
                inner_val_groups = []
                pipeline = build_pipeline(
                    candidate,
                    payload["task_type"],
                    seed=fold_seed,
                    workers=payload["workers"],
                )
                pipeline.fit(x[fold.train], _fit_targets(payload["task_type"], y[fold.train]))
                val_prediction = _reshape_predictions(
                    payload["task_type"], np.asarray(pipeline.predict(x[fold.val])), y[fold.val]
                )
                validation_metrics = evaluate_predictions(
                    payload["task_type"], y[fold.val], val_prediction, payload["target_weights"]
                )
                parameter_count = estimate_parameter_count(pipeline)
                if payload["save_checkpoint"]:
                    _write_sklearn_artifact(fold_dir / "pipeline.joblib", pipeline, candidate)
            if np.issubdtype(val_prediction.dtype, np.number) and not np.isfinite(val_prediction).all():
                raise FloatingPointError(f"Validation predictions contain NaN or infinite values in fold {fold_index}")
            if payload["save_predictions"]:
                _prediction_frame(payload["sample_ids"][fold.val], y[fold.val], val_prediction).to_csv(
                    fold_dir / "validation_predictions.csv", index=False
                )
            training_summary_path = fold_dir / "training_summary.json"
            training_summary = (
                json.loads(training_summary_path.read_text(encoding="utf-8"))
                if training_summary_path.exists()
                else None
            )
            fold_record = {
                "fold": fold_index,
                "seed": fold_seed,
                "metrics": validation_metrics,
                "inner_validation_groups": inner_val_groups,
                "inner_validation_metrics": inner_metrics,
                "training": training_summary,
            }
            fold_metrics.append(fold_record)
            _json_dump(fold_dir / "metrics.json", fold_record)
        metric_names = tuple(fold_metrics[0]["metrics"])
        validation_metrics = {
            metric: float(np.mean([record["metrics"][metric] for record in fold_metrics])) for metric in metric_names
        }
        validation_metrics.update(
            {
                f"{metric}_std": float(np.std([record["metrics"][metric] for record in fold_metrics]))
                for metric in metric_names
            }
        )
        result = TrialResult(
            trial_id=payload["trial_id"],
            name=candidate.trial_name,
            status="completed",
            duration_seconds=time.monotonic() - started,
            validation_metrics=validation_metrics,
            test_metrics={},
            parameter_count=parameter_count,
            artifact_dir=str(trial_dir),
            candidate_index=payload.get("candidate_index"),
            seed=payload["seed"],
        )
        _json_dump(trial_dir / "fold_metrics.json", fold_metrics)
        _json_dump(trial_dir / "metrics.json", result.to_dict())
        _json_dump(
            trial_dir / "config.json",
            {"candidate": candidate.to_dict(), "seed": payload["seed"], "cv_folds": len(payload["folds"])},
        )
        queue.put(result)
    except BaseException as exc:  # noqa: BLE001 - a failed trial must not terminate the search
        message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        lowered = str(exc).lower()
        status = (
            "oom"
            if isinstance(exc, MemoryError) or "out of memory" in lowered
            else "nan"
            if "nan" in lowered
            else "failed"
        )
        result = TrialResult(
            trial_id=payload["trial_id"],
            name=candidate.trial_name,
            status=status,
            duration_seconds=time.monotonic() - started,
            validation_metrics={},
            test_metrics={},
            parameter_count=None,
            artifact_dir=str(trial_dir),
            error=message,
            candidate_index=payload.get("candidate_index"),
            seed=payload["seed"],
        )
        _json_dump(trial_dir / "error.json", result.to_dict())
        queue.put(result)


def _run_isolated_trial(payload: dict[str, Any], timeout_seconds: float) -> TrialResult:
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    worker = _cv_trial_worker if "folds" in payload else _trial_worker
    process = context.Process(target=worker, args=(payload, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        trial_dir = Path(payload["trial_dir"])
        result = TrialResult(
            trial_id=payload["trial_id"],
            name=payload["candidate"].trial_name,
            status="timeout",
            duration_seconds=timeout_seconds,
            validation_metrics={},
            test_metrics={},
            parameter_count=None,
            artifact_dir=str(trial_dir),
            error=f"Trial exceeded {timeout_seconds:.1f} seconds",
            candidate_index=payload.get("candidate_index"),
            seed=payload["seed"],
        )
        _json_dump(trial_dir / "error.json", result.to_dict())
        return result
    try:
        return queue.get(timeout=2)
    except Empty:
        trial_dir = Path(payload["trial_dir"])
        result = TrialResult(
            trial_id=payload["trial_id"],
            name=payload["candidate"].trial_name,
            status="failed",
            duration_seconds=0.0,
            validation_metrics={},
            test_metrics={},
            parameter_count=None,
            artifact_dir=str(trial_dir),
            error=f"Trial process exited with code {process.exitcode} without a result",
            candidate_index=payload.get("candidate_index"),
            seed=payload["seed"],
        )
        _json_dump(trial_dir / "error.json", result.to_dict())
        return result


def _load_trial_result(
    trial_dir: Path,
    trial_id: str,
    candidate: AlgorithmCandidate,
    candidate_index: int,
    seed: int,
) -> TrialResult | None:
    for name in ("metrics.json", "error.json"):
        path = trial_dir / name
        if not path.is_file():
            continue
        result = TrialResult(**json.loads(path.read_text(encoding="utf-8")))
        if (
            result.trial_id != trial_id
            or result.name != candidate.trial_name
            or result.candidate_index != candidate_index
            or result.seed != seed
        ):
            raise ValueError(f"Stored trial identity does not match the immutable plan: {path}")
        return result
    return None


class LeaderboardTask:
    """Validate, plan, and execute one immutable leaderboard task."""

    def __init__(self, config: LeaderboardConfig):
        self.config = config
        self._data: PreparedData | None = None
        self._split: DataSplit | GroupCVSplit | RollingOriginSplit | None = None
        self._protocol_audit: ProtocolAudit | None = None
        self._evaluation_trust_audit: EvaluationTrustAudit | None = None

    def validate(self) -> None:
        if self.config.task.type not in SUPPORTED_TASKS:
            raise ValueError(f"Task {self.config.task.type.value!r} is enumerated but not executable in v1")
        if self.config.dataset.label_granularity != LabelGranularity.SAMPLE:
            raise ValueError("The first executable backend currently supports sample-level labels only")
        device = self.config.resources.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but torch.cuda.is_available() is false: {device}")
        if self.config.pretraining.enabled:
            raise ValueError("Pretraining is disabled until the PyTorch backend is connected")
        if self.config.search.pruning_enabled:
            raise ValueError("Parameter-count pruning is planned for P1 and is not executable yet")
        if (
            self.config.search.torch_target_constraint == "unit_interval"
            and self.config.task.type != TaskType.RUL_REGRESSION
        ):
            raise ValueError("torch_target_constraint='unit_interval' currently supports only rul_regression")
        self._data = prepare_dataset(self.config.dataset)

    def profile_dataset(self) -> dict[str, Any]:
        if self._data is None:
            self._data = prepare_dataset(self.config.dataset)
        return self._data.profile.to_dict()

    def _evaluation_seeds(self) -> tuple[int, ...]:
        return self.config.search.evaluation_seeds or (self.config.seed,)

    def _planned_trials(self, candidates: list[AlgorithmCandidate]) -> list[tuple[int, AlgorithmCandidate, int, str]]:
        seeds = self._evaluation_seeds()
        return [
            (
                candidate_index,
                candidate,
                seed,
                f"trial-{candidate_index:04d}" if len(seeds) == 1 else f"trial-{candidate_index:04d}-seed-{seed:010d}",
            )
            for candidate_index, candidate in enumerate(candidates)
            for seed in seeds
        ]

    def build_plan(self) -> dict[str, Any]:
        data = self._data or prepare_dataset(self.config.dataset)
        if self.config.split.type == SplitType.GROUP_CV:
            split = split_group_cv(data, self.config.split)
        elif self.config.split.type == SplitType.ROLLING_ORIGIN:
            split = split_rolling_origin(data, self.config.split)
        else:
            split = split_dataset(data, self.config.split, self.config.task.type)
        protocol_audit = audit_split_protocol(data, split, self.config.split, self.config.task.type)
        equal_lengths = len(set(data.sequence_lengths)) <= 1
        candidates = build_candidates(
            self.config.task.type,
            self.config.search,
            is_sequence=data.is_sequence,
            equal_sequence_lengths=equal_lengths,
            seed=self.config.seed,
        )
        if not candidates:
            raise ValueError(
                f"candidate_policy={self.config.search.candidate_policy.value!r} produced no compatible "
                f"candidates for {self.config.task.type.value}"
            )
        primary = self.config.ranking.primary_metric or default_primary_metric(self.config.task.type)
        available_metrics = supported_metrics(self.config.task.type)
        if self.config.task.type == TaskType.MULTI_TARGET_REGRESSION:
            for index in range(len(self.config.dataset.label_columns)):
                available_metrics.update({f"target_{index}_mae", f"target_{index}_rmse"})
        requested_metrics = {primary, *self.config.ranking.additional_metrics}
        unavailable = requested_metrics - available_metrics
        if unavailable:
            raise ValueError(
                f"Unsupported ranking metrics for {self.config.task.type.value}: {sorted(unavailable)}; "
                f"available={sorted(available_metrics)}"
            )
        if primary == "weighted_nrmse" and self.config.ranking.target_weights is None:
            raise ValueError("weighted_nrmse requires ranking.target_weights")
        planned_trials = self._planned_trials(candidates)
        evaluation_trust_audit = audit_evaluation_trust(self.config, candidates, planned_trials, protocol_audit)
        self._data, self._split, self._protocol_audit = data, split, protocol_audit
        self._evaluation_trust_audit = evaluation_trust_audit
        plan = {
            "task_id": self.config.task_id,
            "task_type": self.config.task.type.value,
            "trust_level": self.config.trust.level.value,
            "primary_metric": primary,
            "metric_direction": metric_direction(primary),
            "candidate_policy": self.config.search.candidate_policy.value,
            "samples": len(data.x),
            "candidate_count": len(candidates),
            "trial_count": len(candidates) * len(self._evaluation_seeds()),
            "evaluation_seeds": list(self._evaluation_seeds()),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "warnings": list(data.profile.warnings),
            "protocol_audit": protocol_audit.to_dict(),
            "evaluation_trust_audit": evaluation_trust_audit.to_dict(),
        }
        if isinstance(split, (GroupCVSplit, RollingOriginSplit)):
            plan.update(
                {
                    "protocol": (
                        "group_cv_sealed_test" if isinstance(split, GroupCVSplit) else "rolling_origin_sealed_test"
                    ),
                    "split_sizes": {
                        "development": len(split.development),
                        "test": len(split.test),
                    },
                    "validation_folds": [{"train": len(fold.train), "val": len(fold.val)} for fold in split.folds],
                }
            )
        else:
            plan.update(
                {
                    "protocol": (
                        "predefined_validation_sealed_test"
                        if self.config.task.type == TaskType.ANOMALY_DETECTION
                        else "single_validation_sealed_test"
                    ),
                    "split_sizes": {
                        "train": len(split.train),
                        "val": len(split.val),
                        "test": len(split.test),
                    },
                }
            )
        return plan

    def run(self, *, resume: bool = False) -> LeaderboardResult:
        self.validate()
        plan = self.build_plan()
        assert self._data is not None and self._split is not None and self._protocol_audit is not None
        assert self._evaluation_trust_audit is not None
        run_dir = self.config.run_dir
        self._prepare_run_directory(run_dir, plan, resume=resume)
        trials_dir = run_dir / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)

        candidates = [AlgorithmCandidate(**candidate) for candidate in plan["candidates"]]
        if self.config.task.type == TaskType.ANOMALY_DETECTION:
            return self._run_anomaly_detection(run_dir, trials_dir, plan, candidates)
        if isinstance(self._split, (GroupCVSplit, RollingOriginSplit)):
            return self._run_cross_validation(run_dir, trials_dir, plan, candidates)
        return self._run_single_split(run_dir, trials_dir, plan, candidates)

    def _prepare_run_directory(self, run_dir: Path, plan: dict[str, Any], *, resume: bool) -> None:
        assert self._data is not None and self._split is not None
        expected = {
            "config.json": self.config.to_dict(),
            "dataset_profile.json": self._data.profile.to_dict(),
            "data_split.json": self._split.to_dict(),
            "protocol_audit.json": self._protocol_audit.to_dict(),
            "evaluation_trust_audit.json": self._evaluation_trust_audit.to_dict(),
            "plan.json": plan,
        }
        non_empty = run_dir.exists() and any(run_dir.iterdir())
        if non_empty and not resume:
            raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, value in expected.items():
            path = run_dir / name
            normalized = json.loads(json.dumps(value, ensure_ascii=False, default=str))
            if path.is_file():
                stored = json.loads(path.read_text(encoding="utf-8"))
                if stored != normalized:
                    raise ValueError(f"Cannot resume because immutable task state changed: {path}")
            else:
                _json_dump(path, value)
        environment_path = run_dir / "environment.json"
        if not environment_path.is_file():
            _json_dump(environment_path, self._environment())

    @staticmethod
    def _completed_sealed_test(run_dir: Path) -> dict[str, Any] | None:
        path = run_dir / "sealed_test_metrics.json"
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return record if record.get("status") == "completed" else None

    def _run_single_split(
        self,
        run_dir: Path,
        trials_dir: Path,
        plan: dict[str, Any],
        candidates: list[AlgorithmCandidate],
    ) -> LeaderboardResult:
        assert self._data is not None and isinstance(self._split, DataSplit)
        split = self._split
        development = np.concatenate((split.train, split.val))
        local_train = np.arange(len(split.train), dtype=np.int64)
        local_val = np.arange(len(split.train), len(development), dtype=np.int64)
        local_split = DataSplit(local_train, local_val, local_val)
        results: list[TrialResult] = []
        task_started = time.monotonic()
        total_limit = self.config.resources.max_total_minutes * 60
        timeout = self.config.resources.timeout_per_trial_minutes * 60
        for candidate_index, candidate, seed, trial_id in self._planned_trials(candidates):
            trial_dir = trials_dir / trial_id
            stored = _load_trial_result(trial_dir, trial_id, candidate, candidate_index, seed)
            if stored is not None:
                results.append(stored)
                continue
            elapsed = time.monotonic() - task_started
            if elapsed >= total_limit:
                break
            payload = self._trial_payload(
                trial_id,
                trial_dir,
                candidate,
                x=self._data.x[development],
                y=self._data.y[development],
                sample_ids=self._data.sample_ids[development],
                target_weights=self._target_weights(),
                candidate_index=candidate_index,
                seed=seed,
            )
            payload["split"] = local_split
            payload["selection_only"] = True
            results.append(_run_isolated_trial(payload, min(timeout, total_limit - elapsed)))

        leaderboard = self._build_leaderboard(results, plan["primary_metric"], candidates)
        self._write_leaderboards(run_dir, leaderboard)
        _json_dump(run_dir / "trials.json", [result.to_dict() for result in results])
        sealed_test = self._evaluate_selected_on_single_test(
            run_dir,
            leaderboard,
            results,
            candidates,
            development,
            timeout,
            max(0.0, total_limit - (time.monotonic() - task_started)),
        )
        _json_dump(run_dir / "sealed_test_metrics.json", sealed_test)
        (run_dir / "final_report.md").write_text(
            self._report(leaderboard, results, plan["primary_metric"], sealed_test=sealed_test), encoding="utf-8"
        )
        return LeaderboardResult(run_dir, leaderboard, tuple(results), plan["primary_metric"])

    def _evaluate_selected_on_single_test(
        self,
        run_dir: Path,
        leaderboard: pd.DataFrame,
        results: list[TrialResult],
        candidates: list[AlgorithmCandidate],
        development: np.ndarray,
        timeout: float,
        remaining_time: float,
    ) -> dict[str, Any]:
        assert self._data is not None and isinstance(self._split, DataSplit)
        completed = self._completed_sealed_test(run_dir)
        if completed is not None:
            return completed
        if leaderboard.empty or leaderboard.iloc[0]["status"] != "completed":
            return {"status": "not_run", "reason": "No completed validation trial"}
        if remaining_time <= 0:
            return {"status": "not_run", "reason": "Task time budget exhausted before final retraining"}
        best_id = str(leaderboard.iloc[0]["trial_id"])
        candidate_index = int(leaderboard.iloc[0]["candidate_index"])
        candidate = candidates[candidate_index]
        candidate_results = [result for result in results if result.candidate_index == candidate_index]
        selected_epochs = self._selected_training_epochs(candidate_results, candidate)
        final_seed = self._evaluation_seeds()[0]
        final_split = DataSplit(train=development, val=self._split.test, test=self._split.test)
        payload = self._trial_payload(
            best_id,
            run_dir / "best_model",
            candidate,
            x=self._data.x,
            y=self._data.y,
            sample_ids=self._data.sample_ids,
            target_weights=self._target_weights(),
            candidate_index=candidate_index,
            seed=final_seed,
        )
        payload["split"] = final_split
        payload["final_only"] = True
        if selected_epochs is not None:
            payload["torch_epochs"] = selected_epochs
            payload["torch_early_stopping_patience"] = None
            payload["torch_restore_best"] = False
        final_result = _run_isolated_trial(payload, min(timeout, remaining_time))
        record = {
            "status": final_result.status,
            "selected_trial_id": best_id,
            "selected_pipeline": candidate.trial_name,
            "development_samples": len(development),
            "test_samples": len(self._split.test),
            "test_metrics": final_result.test_metrics,
            "selected_training_epochs": selected_epochs,
            "evaluation_seeds": list(self._evaluation_seeds()),
            "final_training_seed": final_seed,
            "duration_seconds": final_result.duration_seconds,
            "artifact_dir": final_result.artifact_dir,
            "error": final_result.error,
        }
        _json_dump(run_dir / "best_model" / "selection.json", record)
        return record

    def _run_anomaly_detection(
        self,
        run_dir: Path,
        trials_dir: Path,
        plan: dict[str, Any],
        candidates: list[AlgorithmCandidate],
    ) -> LeaderboardResult:
        assert self._data is not None and isinstance(self._split, DataSplit)
        split = self._split
        development = np.concatenate((split.train, split.val))
        local_train = np.arange(len(split.train), dtype=np.int64)
        local_val = np.arange(len(split.train), len(development), dtype=np.int64)
        local_split = DataSplit(local_train, local_val, local_val)
        results: list[TrialResult] = []
        task_started = time.monotonic()
        total_limit = self.config.resources.max_total_minutes * 60
        timeout = self.config.resources.timeout_per_trial_minutes * 60
        for candidate_index, candidate, seed, trial_id in self._planned_trials(candidates):
            trial_dir = trials_dir / trial_id
            stored = _load_trial_result(trial_dir, trial_id, candidate, candidate_index, seed)
            if stored is not None:
                results.append(stored)
                continue
            elapsed = time.monotonic() - task_started
            if elapsed >= total_limit:
                break
            payload = self._trial_payload(
                trial_id,
                trial_dir,
                candidate,
                x=self._data.x[development],
                y=self._data.y[development],
                sample_ids=self._data.sample_ids[development],
                target_weights=None,
                candidate_index=candidate_index,
                seed=seed,
            )
            payload["split"] = local_split
            payload["anomaly_validation_only"] = True
            results.append(_run_isolated_trial(payload, min(timeout, total_limit - elapsed)))

        leaderboard = self._build_leaderboard(results, plan["primary_metric"], candidates)
        self._write_leaderboards(run_dir, leaderboard)
        _json_dump(run_dir / "trials.json", [result.to_dict() for result in results])
        sealed_test: dict[str, Any]
        remaining = max(0.0, total_limit - (time.monotonic() - task_started))
        completed = self._completed_sealed_test(run_dir)
        if completed is not None:
            sealed_test = completed
        elif leaderboard.empty or leaderboard.iloc[0]["status"] != "completed":
            sealed_test = {"status": "not_run", "reason": "No completed validation trial"}
        elif remaining <= 0:
            sealed_test = {"status": "not_run", "reason": "Task time budget exhausted before sealed test"}
        else:
            best_id = str(leaderboard.iloc[0]["trial_id"])
            candidate_index = int(leaderboard.iloc[0]["candidate_index"])
            candidate = candidates[candidate_index]
            final_seed = self._evaluation_seeds()[0]
            payload = self._trial_payload(
                best_id,
                run_dir / "best_model",
                candidate,
                x=self._data.x,
                y=self._data.y,
                sample_ids=self._data.sample_ids,
                target_weights=None,
                candidate_index=candidate_index,
                seed=final_seed,
            )
            payload["split"] = split
            payload["anomaly_validation_only"] = False
            final_result = _run_isolated_trial(payload, min(timeout, remaining))
            threshold_path = run_dir / "best_model" / "threshold.json"
            threshold = (
                json.loads(threshold_path.read_text(encoding="utf-8"))["value"] if threshold_path.is_file() else None
            )
            test_timestamps = [
                str(pd.Timestamp(self._data.timestamps[split.test].min())),
                str(pd.Timestamp(self._data.timestamps[split.test].max())),
            ]
            sealed_test = {
                "status": final_result.status,
                "selected_trial_id": best_id,
                "selected_pipeline": candidate.trial_name,
                "train_samples": len(split.train),
                "validation_samples": len(split.val),
                "test_samples": len(split.test),
                "test_timestamp_range": test_timestamps,
                "threshold": threshold,
                "threshold_selected_on": "validation",
                "evaluation_seeds": list(self._evaluation_seeds()),
                "final_training_seed": final_seed,
                "test_metrics": final_result.test_metrics,
                "duration_seconds": final_result.duration_seconds,
                "artifact_dir": final_result.artifact_dir,
                "error": final_result.error,
            }
            _json_dump(run_dir / "best_model" / "selection.json", sealed_test)
        _json_dump(run_dir / "sealed_test_metrics.json", sealed_test)
        (run_dir / "final_report.md").write_text(
            self._report(leaderboard, results, plan["primary_metric"], sealed_test=sealed_test),
            encoding="utf-8",
        )
        return LeaderboardResult(run_dir, leaderboard, tuple(results), plan["primary_metric"])

    def _run_cross_validation(
        self,
        run_dir: Path,
        trials_dir: Path,
        plan: dict[str, Any],
        candidates: list[AlgorithmCandidate],
    ) -> LeaderboardResult:
        assert self._data is not None and isinstance(self._split, (GroupCVSplit, RollingOriginSplit))
        split = self._split
        development = split.development
        global_to_local = np.full(len(self._data.x), -1, dtype=np.int64)
        global_to_local[development] = np.arange(len(development))
        local_folds = tuple(
            ValidationFold(global_to_local[fold.train], global_to_local[fold.val]) for fold in split.folds
        )
        target_weights = self._target_weights()
        results: list[TrialResult] = []
        task_started = time.monotonic()
        total_limit = self.config.resources.max_total_minutes * 60
        timeout = self.config.resources.timeout_per_trial_minutes * 60
        for candidate_index, candidate, seed, trial_id in self._planned_trials(candidates):
            trial_dir = trials_dir / trial_id
            stored = _load_trial_result(trial_dir, trial_id, candidate, candidate_index, seed)
            if stored is not None:
                results.append(stored)
                continue
            elapsed = time.monotonic() - task_started
            if elapsed >= total_limit:
                break
            payload = self._trial_payload(
                trial_id,
                trial_dir,
                candidate,
                x=self._data.x[development],
                y=self._data.y[development],
                sample_ids=self._data.sample_ids[development],
                target_weights=target_weights,
                candidate_index=candidate_index,
                seed=seed,
            )
            payload["folds"] = local_folds
            payload["cv_protocol"] = "group_cv" if isinstance(split, GroupCVSplit) else "rolling_origin"
            if isinstance(split, GroupCVSplit):
                assert self._data.groups is not None
                payload["groups"] = self._data.groups[development]
            results.append(_run_isolated_trial(payload, min(timeout, total_limit - elapsed)))

        leaderboard = self._build_leaderboard(results, plan["primary_metric"], candidates)
        self._write_leaderboards(run_dir, leaderboard)
        _json_dump(run_dir / "trials.json", [result.to_dict() for result in results])
        sealed_test = self._evaluate_selected_on_sealed_test(
            run_dir,
            leaderboard,
            results,
            candidates,
            target_weights,
            timeout,
            max(0.0, total_limit - (time.monotonic() - task_started)),
        )
        _json_dump(run_dir / "sealed_test_metrics.json", sealed_test)
        (run_dir / "final_report.md").write_text(
            self._report(leaderboard, results, plan["primary_metric"], sealed_test=sealed_test),
            encoding="utf-8",
        )
        return LeaderboardResult(run_dir, leaderboard, tuple(results), plan["primary_metric"])

    def _evaluate_selected_on_sealed_test(
        self,
        run_dir: Path,
        leaderboard: pd.DataFrame,
        results: list[TrialResult],
        candidates: list[AlgorithmCandidate],
        target_weights: np.ndarray | None,
        timeout: float,
        remaining_time: float,
    ) -> dict[str, Any]:
        assert self._data is not None and isinstance(self._split, (GroupCVSplit, RollingOriginSplit))
        completed = self._completed_sealed_test(run_dir)
        if completed is not None:
            return completed
        if leaderboard.empty or leaderboard.iloc[0]["status"] != "completed":
            return {"status": "not_run", "reason": "No completed validation trial"}
        if remaining_time <= 0:
            return {"status": "not_run", "reason": "Task time budget exhausted before final retraining"}
        best_id = str(leaderboard.iloc[0]["trial_id"])
        candidate_index = int(leaderboard.iloc[0]["candidate_index"])
        candidate = candidates[candidate_index]
        candidate_results = [result for result in results if result.candidate_index == candidate_index]
        selected_epochs = self._selected_training_epochs(candidate_results, candidate)
        final_seed = self._evaluation_seeds()[0]
        split = DataSplit(
            train=self._split.development,
            val=self._split.test,
            test=self._split.test,
        )
        payload = self._trial_payload(
            best_id,
            run_dir / "best_model",
            candidate,
            x=self._data.x,
            y=self._data.y,
            sample_ids=self._data.sample_ids,
            target_weights=target_weights,
            candidate_index=candidate_index,
            seed=final_seed,
        )
        payload["split"] = split
        payload["final_only"] = True
        if selected_epochs is not None:
            payload["torch_epochs"] = selected_epochs
            payload["torch_early_stopping_patience"] = None
            payload["torch_restore_best"] = False
        final_result = _run_isolated_trial(payload, min(timeout, remaining_time))
        test_groups = (
            sorted(np.unique(self._data.groups[self._split.test]).astype(str).tolist())
            if isinstance(self._split, GroupCVSplit) and self._data.groups is not None
            else []
        )
        test_timestamps = (
            [
                str(min(self._data.timestamps[self._split.test].tolist())),
                str(max(self._data.timestamps[self._split.test].tolist())),
            ]
            if isinstance(self._split, RollingOriginSplit) and self._data.timestamps is not None
            else None
        )
        record = {
            "status": final_result.status,
            "selected_trial_id": best_id,
            "selected_pipeline": candidate.trial_name,
            "development_samples": len(self._split.development),
            "test_samples": len(self._split.test),
            "test_groups": test_groups,
            "test_timestamp_range": test_timestamps,
            "test_metrics": final_result.test_metrics,
            "selected_training_epochs": selected_epochs,
            "evaluation_seeds": list(self._evaluation_seeds()),
            "final_training_seed": final_seed,
            "duration_seconds": final_result.duration_seconds,
            "artifact_dir": final_result.artifact_dir,
            "error": final_result.error,
        }
        _json_dump(run_dir / "best_model" / "selection.json", record)
        return record

    @staticmethod
    def _selected_training_epochs(results: list[TrialResult], candidate: AlgorithmCandidate) -> int | None:
        if candidate.backend != "pytorch":
            return None
        best_epochs = []
        for result in results:
            if result.status != "completed":
                continue
            artifact_dir = Path(result.artifact_dir)
            fold_metrics_path = artifact_dir / "fold_metrics.json"
            if fold_metrics_path.is_file():
                fold_records = json.loads(fold_metrics_path.read_text(encoding="utf-8"))
                best_epochs.extend(
                    record["training"]["best_epoch"]
                    for record in fold_records
                    if record.get("training") and record["training"].get("best_epoch")
                )
            else:
                summary_path = artifact_dir / "training_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
                if summary.get("best_epoch"):
                    best_epochs.append(summary["best_epoch"])
        return max(1, int(round(float(np.median(best_epochs))))) if best_epochs else None

    def _trial_payload(
        self,
        trial_id: str,
        trial_dir: Path,
        candidate: AlgorithmCandidate,
        *,
        x: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray,
        target_weights: np.ndarray | None,
        candidate_index: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        resolved_seed = self.config.seed if seed is None else seed
        return {
            "trial_id": trial_id,
            "trial_dir": str(trial_dir),
            "candidate": candidate,
            "task_type": self.config.task.type,
            "seed": resolved_seed,
            "candidate_index": candidate_index,
            "workers": self.config.resources.max_cpu_workers,
            "device": self.config.resources.device,
            "x": x,
            "y": y,
            "sample_ids": sample_ids,
            "save_checkpoint": self.config.artifacts.save_checkpoints,
            "save_predictions": self.config.artifacts.save_predictions,
            "target_weights": target_weights,
            "torch_epochs": self.config.search.torch_epochs,
            "torch_batch_size": self.config.search.torch_batch_size,
            "torch_learning_rate": self.config.search.torch_learning_rate,
            "torch_optimizer": self.config.search.torch_optimizer,
            "torch_weight_decay": self.config.search.torch_weight_decay,
            "torch_loss": self.config.search.torch_loss,
            "torch_grad_clip_norm": self.config.search.torch_grad_clip_norm,
            "torch_lr_scheduler": self.config.search.torch_lr_scheduler,
            "torch_early_stopping_patience": self.config.search.torch_early_stopping_patience,
            "torch_min_delta": self.config.search.torch_min_delta,
            "torch_restore_best": self.config.search.torch_restore_best,
            "torch_target_constraint": self.config.search.torch_target_constraint,
            "forecast_target_indices": (
                [self._data.feature_columns.index(label) for label in self.config.dataset.label_columns]
                if self.config.task.type == TaskType.TIME_SERIES_FORECASTING and self._data is not None
                else []
            ),
            "feature_columns": self._data.feature_columns if self._data is not None else (),
            "anomaly_threshold_candidates": self.config.search.anomaly_threshold_candidates,
        }

    def _target_weights(self) -> np.ndarray | None:
        if not self.config.ranking.target_weights:
            return None
        return np.asarray(
            [self.config.ranking.target_weights[label] for label in self.config.dataset.label_columns],
            dtype=np.float64,
        )

    @staticmethod
    def _family_leaderboard(leaderboard: pd.DataFrame, family: str) -> pd.DataFrame:
        if leaderboard.empty or "model_family" not in leaderboard:
            return leaderboard.iloc[0:0].copy()
        frame = leaderboard[leaderboard["model_family"] == family].copy()
        if frame.empty:
            return frame
        if "rank" in frame:
            frame = frame.rename(columns={"rank": "overall_rank"})
        else:
            frame.insert(0, "overall_rank", None)
        completed = frame["status"] == "completed"
        frame.insert(0, "family_rank", [*range(1, int(completed.sum()) + 1), *([None] * int((~completed).sum()))])
        return frame

    @classmethod
    def _write_leaderboards(cls, run_dir: Path, leaderboard: pd.DataFrame) -> None:
        leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)
        cls._family_leaderboard(leaderboard, "torch_like").to_csv(run_dir / "leaderboard_torch_like.csv", index=False)
        cls._family_leaderboard(leaderboard, "baseline").to_csv(run_dir / "leaderboard_baselines.csv", index=False)

    def _build_leaderboard(
        self,
        results: list[TrialResult],
        primary: str,
        candidates: list[AlgorithmCandidate],
    ) -> pd.DataFrame:
        rows = []
        expected_seeds = self._evaluation_seeds()
        seed_ranks: dict[int, dict[int, int]] = {}
        for seed in expected_seeds:
            seed_results = [
                result
                for result in results
                if result.seed == seed and result.status == "completed" and primary in result.validation_metrics
            ]
            seed_results.sort(
                key=lambda result: result.validation_metrics[primary],
                reverse=metric_direction(primary) == "maximize",
            )
            seed_ranks[seed] = {
                int(result.candidate_index): rank
                for rank, result in enumerate(seed_results, start=1)
                if result.candidate_index is not None
            }
        grouped: dict[int, list[TrialResult]] = {}
        for result in results:
            if result.candidate_index is None:
                raise ValueError(f"Trial lacks candidate_index: {result.trial_id}")
            grouped.setdefault(result.candidate_index, []).append(result)
        for candidate_index, candidate_results in sorted(grouped.items()):
            candidate = candidates[candidate_index]
            candidate_results.sort(
                key=lambda item: expected_seeds.index(item.seed) if item.seed in expected_seeds else -1
            )
            completed_results = [result for result in candidate_results if result.status == "completed"]
            completed_seeds = {result.seed for result in completed_results}
            complete_coverage = completed_seeds == set(expected_seeds)
            representative = completed_results[0] if completed_results else candidate_results[0]
            status = (
                "completed" if complete_coverage else "partial" if completed_results else candidate_results[0].status
            )
            row = {
                "trial_id": representative.trial_id,
                "candidate_index": candidate_index,
                "pipeline": representative.name,
                "backend": candidate.backend,
                "model_family": candidate.model_family,
                "status": status,
                "seed_count": len(expected_seeds),
                "completed_seed_count": len(completed_seeds),
                "evaluation_seeds": ",".join(str(seed) for seed in expected_seeds),
                "duration_seconds": float(sum(result.duration_seconds for result in candidate_results)),
                "parameter_count": next(
                    (result.parameter_count for result in completed_results if result.parameter_count is not None), None
                ),
            }
            ranks = [
                seed_ranks[seed][candidate_index] for seed in expected_seeds if candidate_index in seed_ranks[seed]
            ]
            if len(expected_seeds) > 1:
                row.update(
                    {
                        "seed_ranks": ",".join(str(rank) for rank in ranks),
                        "rank_mean_across_seeds": float(np.mean(ranks)) if ranks else None,
                        "rank_std_across_seeds": float(np.std(ranks)) if ranks else None,
                        "rank_min_across_seeds": min(ranks) if ranks else None,
                        "rank_max_across_seeds": max(ranks) if ranks else None,
                    }
                )
            metric_names = sorted({key for result in completed_results for key in result.validation_metrics})
            for key in metric_names:
                values = [
                    result.validation_metrics[key] for result in completed_results if key in result.validation_metrics
                ]
                row[f"val_{key}"] = float(np.mean(values))
                if len(expected_seeds) > 1:
                    row[f"val_{key}_seed_std"] = float(np.std(values))
            test_metric_names = sorted({key for result in completed_results for key in result.test_metrics})
            for key in test_metric_names:
                values = [result.test_metrics[key] for result in completed_results if key in result.test_metrics]
                row[f"test_{key}"] = float(np.mean(values))
            rows.append(row)
        frame = pd.DataFrame(rows)
        metric_column = f"val_{primary}"
        if not frame.empty and metric_column in frame:
            completed = frame["status"] == "completed"
            ranked = frame[completed].sort_values(
                metric_column,
                ascending=metric_direction(primary) == "minimize",
                kind="stable",
            )
            failed = frame[~completed]
            frame = pd.concat([ranked, failed], ignore_index=True)
            frame.insert(0, "rank", [*range(1, len(ranked) + 1), *([None] * len(failed))])
        return frame

    def _environment(self) -> dict[str, Any]:
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = "unknown"
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "torch_like": importlib.metadata.version("torch_like"),
            "git_commit": git_commit,
            "pid": os.getpid(),
            "seed": self.config.seed,
            "evaluation_seeds": list(self._evaluation_seeds()),
            "dataset_fingerprint": self._data.profile.fingerprint if self._data else None,
        }

    def _report(
        self,
        leaderboard: pd.DataFrame,
        results: list[TrialResult],
        primary: str,
        *,
        sealed_test: dict[str, Any] | None = None,
    ) -> str:
        completed = sum(result.status == "completed" for result in results)
        failures = len(results) - completed
        best = "None / 无" if leaderboard.empty else str(leaderboard.iloc[0].get("pipeline", "None / 无"))
        lines = [
            "# Leaderboard Report / 模型排行榜报告",
            "",
            f"- Task / 任务: `{self.config.task_id}`",
            f"- Task type / 任务类型: `{self.config.task.type.value}`",
            f"- Primary metric / 主指标: `{primary}` (validation ranking / 验证集排名)",
            f"- Evaluation seeds / 评估种子: `{list(self._evaluation_seeds())}`",
            f"- Candidate policy / 候选策略: `{self.config.search.candidate_policy.value}`",
            f"- Completed trials / 完成试验: {completed}",
            f"- Failed trials / 失败试验: {failures}",
            f"- Best pipeline / 最佳管线: `{best}`",
            "",
        ]
        completed_rows = leaderboard[leaderboard["status"] == "completed"].head(5) if not leaderboard.empty else None
        if completed_rows is not None and not completed_rows.empty:
            metric_column = f"val_{primary}"
            std_column = f"{metric_column}_seed_std"
            lines.extend(
                [
                    "## Top Candidates / 前五名候选",
                    "",
                    "| Rank / 排名 | Pipeline / 管线 | Validation / 验证均值 | Seed std / 种子标准差 | "
                    "Mean seed rank / 种子平均名次 |",
                    "|---:|---|---:|---:|---:|",
                ]
            )
            for _, row in completed_rows.iterrows():
                rank = int(row["rank"])
                metric = float(row[metric_column])
                std = row.get(std_column)
                mean_rank = row.get("rank_mean_across_seeds")
                std_text = "n/a" if pd.isna(std) else f"{float(std):.6f}"
                mean_rank_text = "n/a" if pd.isna(mean_rank) else f"{float(mean_rank):.3f}"
                lines.append(f"| {rank} | `{row['pipeline']}` | {metric:.6f} | {std_text} | {mean_rank_text} |")
            lines.append("")
        for family, title in (
            ("torch_like", "Torch-like Candidates / torch_like 模型榜"),
            ("baseline", "Baseline Candidates / 基线模型榜"),
        ):
            family_rows = self._family_leaderboard(leaderboard, family)
            if family_rows.empty:
                continue
            family_rows = family_rows[family_rows["status"] == "completed"].head(5)
            if family_rows.empty:
                continue
            metric_column = f"val_{primary}"
            lines.extend(
                [
                    f"## {title}",
                    "",
                    "| Family rank / 族内排名 | Overall rank / 总排名 | Pipeline / 管线 | Validation / 验证均值 |",
                    "|---:|---:|---|---:|",
                ]
            )
            for _, row in family_rows.iterrows():
                lines.append(
                    f"| {int(row['family_rank'])} | {int(row['overall_rank'])} | "
                    f"`{row['pipeline']}` | {float(row[metric_column]):.6f} |"
                )
            lines.append("")
        if sealed_test is None:
            lines.extend(
                [
                    "This run uses a single split. Test metrics are computed after each pipeline is trained.",
                    "",
                    "本次运行采用单次划分，每条管线训练后均计算测试指标。",
                ]
            )
        else:
            metrics = sealed_test.get("test_metrics", {})
            if isinstance(self._split, RollingOriginSplit):
                protocol = "rolling-origin backtesting"
            elif isinstance(self._split, GroupCVSplit):
                protocol = "group cross-validation"
            elif self.config.task.type == TaskType.ANOMALY_DETECTION:
                protocol = "predefined chronological validation"
            else:
                protocol = "a fixed development split"
            lines.extend(
                [
                    f"This run ranks candidates only by {protocol}. The sealed test data are exposed once, "
                    "after selecting and retraining the best pipeline.",
                    "",
                    "本次运行仅按开发阶段的交叉验证或滚动回测结果选择候选。测试数据在最佳管线选定并重训后才解封一次。",
                    "",
                    f"- Sealed-test status / 封存测试状态: `{sealed_test.get('status', 'unknown')}`",
                    f"- Sealed-test groups / 封存测试组: `{sealed_test.get('test_groups', [])}`",
                    f"- Sealed-test time range / 封存测试时间范围: `{sealed_test.get('test_timestamp_range')}`",
                    f"- Sealed-test metrics / 封存测试指标: `{metrics}`",
                ]
            )
        return "\n".join(lines) + "\n"
