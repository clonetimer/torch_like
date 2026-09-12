"""Builtin backend runners: bind execution handlers to backend specs.

This module is the seam that lets a backend be added without touching the
orchestrator. A runner is a plain callable attached to a
:class:`~torch_like.automl.backends.BackendSpec`; :mod:`torch_like.automl.task`
only ever does ``get_backend(name).runner(...)``, so it carries no knowledge of
sklearn, PyTorch, or anomaly detection.

Runner protocols
----------------
``runner(candidate, payload, split, trial_dir)``
    Single split. Returns ``(validation_metrics, test_metrics, parameter_count,
    val_prediction, test_prediction)``, where the metrics/predictions refer to
    ``split.val`` and ``split.test`` respectively.

``cv_runner(candidate, payload, fold, fold_dir, fold_seed)``
    One cross-validation fold. Returns ``(inner_metrics, validation_metrics,
    parameter_count, val_prediction, inner_val_groups)``. Backends that need an
    inner validation window (PyTorch early stopping) build it themselves; the
    others report ``inner_metrics=None`` and ``inner_val_groups=[]``.

``payload`` is the immutable trial payload built by the orchestrator; runners
read the options they need from it, so adding a knob does not change signatures.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from torch_like.automl.anomaly_backend import AnomalyRunResult, run_anomaly_baseline, run_anomaly_pytorch
from torch_like.automl.artifacts import json_dump, write_sklearn_artifact
from torch_like.automl.backends import attach_runner
from torch_like.automl.events import training_events
from torch_like.automl.data import DataSplit, ValidationFold
from torch_like.automl.sklearn_backend import (
    AlgorithmCandidate,
    build_pipeline,
    estimate_parameter_count,
    evaluate_predictions,
)
from torch_like.automl.specs import TaskType
from torch_like.automl.torch_backend import run_torch_candidate


def _fit_targets(task_type: TaskType, targets: np.ndarray) -> np.ndarray:
    return targets.reshape(len(targets), -1) if task_type == TaskType.TIME_SERIES_FORECASTING else targets


def _reshape_predictions(task_type: TaskType, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return predictions.reshape(targets.shape) if task_type == TaskType.TIME_SERIES_FORECASTING else predictions


def _last_value_forecast(x: np.ndarray, target_indices: list[int], target_shape: tuple[int, ...]) -> np.ndarray:
    last_values = np.stack([np.asarray(sequence)[-1, target_indices] for sequence in x])
    return np.repeat(last_values[:, None, :], target_shape[1], axis=1)


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


def run_sklearn_single(
    candidate: AlgorithmCandidate, payload: dict[str, Any], split: DataSplit, trial_dir: Path
) -> tuple[dict[str, float], dict[str, float], int | None, np.ndarray, np.ndarray]:
    task_type, x, y = payload["task_type"], payload["x"], payload["y"]
    training_events.emit("training_start", candidate=candidate.name, backend=candidate.backend, task_type=task_type.value)
    pipeline = build_pipeline(candidate, task_type, seed=payload["seed"], workers=payload["workers"])
    pipeline.fit(x[split.train], _fit_targets(task_type, y[split.train]))
    final_only = payload.get("final_only", False)
    val_prediction = (
        np.asarray(pipeline.predict(x[split.test]))
        if final_only
        else np.asarray(pipeline.predict(x[split.val]))
    )
    test_prediction = val_prediction if final_only else np.asarray(pipeline.predict(x[split.test]))
    val_prediction = _reshape_predictions(task_type, val_prediction, y[split.test] if final_only else y[split.val])
    test_prediction = _reshape_predictions(task_type, test_prediction, y[split.test])
    validation_metrics = (
        {}
        if final_only
        else evaluate_predictions(task_type, y[split.val], val_prediction, payload["target_weights"])
    )
    test_metrics = evaluate_predictions(task_type, y[split.test], test_prediction, payload["target_weights"])
    parameter_count = estimate_parameter_count(pipeline)
    if payload["save_checkpoint"]:
        write_sklearn_artifact(trial_dir / "pipeline.joblib", pipeline, candidate)
    training_events.emit("training_end", candidate=candidate.name, backend=candidate.backend, parameter_count=parameter_count)
    return validation_metrics, test_metrics, parameter_count, val_prediction, test_prediction


def run_baseline_single(
    candidate: AlgorithmCandidate, payload: dict[str, Any], split: DataSplit, trial_dir: Path
) -> tuple[dict[str, float], dict[str, float], int, np.ndarray, np.ndarray]:
    task_type, x, y = payload["task_type"], payload["x"], payload["y"]
    training_events.emit("training_start", candidate=candidate.name, backend=candidate.backend, task_type=task_type.value)
    final_only = payload.get("final_only", False)
    val_indices = split.test if final_only else split.val
    val_prediction = _last_value_forecast(x[val_indices], payload["forecast_target_indices"], y[val_indices].shape)
    test_prediction = (
        val_prediction
        if final_only
        else _last_value_forecast(x[split.test], payload["forecast_target_indices"], y[split.test].shape)
    )
    validation_metrics = (
        {}
        if final_only
        else evaluate_predictions(task_type, y[split.val], val_prediction, payload["target_weights"])
    )
    test_metrics = evaluate_predictions(task_type, y[split.test], test_prediction, payload["target_weights"])
    json_dump(
        trial_dir / "artifact_manifest.json",
        {"backend": "baseline", "name": candidate.name, "parameter_count": 0},
    )
    training_events.emit("training_end", candidate=candidate.name, backend=candidate.backend, parameter_count=0)
    return validation_metrics, test_metrics, 0, val_prediction, test_prediction


def run_torch_single(
    candidate: AlgorithmCandidate, payload: dict[str, Any], split: DataSplit, trial_dir: Path
) -> tuple[dict[str, float], dict[str, float], int, np.ndarray, np.ndarray]:
    return run_torch_candidate(
        candidate,
        payload["task_type"],
        payload["x"],
        payload["y"],
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


def _anomaly_tuple(
    result: AnomalyRunResult,
) -> tuple[dict[str, float], dict[str, float], int | None, np.ndarray, np.ndarray]:
    return (
        result.validation_metrics,
        result.test_metrics,
        result.parameter_count,
        result.validation_scores,
        result.validation_scores if result.test_scores is None else result.test_scores,
    )


def run_anomaly_baseline_single(
    candidate: AlgorithmCandidate, payload: dict[str, Any], split: DataSplit, trial_dir: Path
) -> tuple[dict[str, float], dict[str, float], int | None, np.ndarray, np.ndarray]:
    result = run_anomaly_baseline(
        candidate,
        payload["x"],
        payload["y"],
        split,
        sample_ids=payload["sample_ids"],
        channels=payload["feature_columns"],
        seed=payload["seed"],
        workers=payload["workers"],
        threshold_candidates=payload["anomaly_threshold_candidates"],
        trial_dir=trial_dir,
        save_predictions=payload["save_predictions"],
        validation_only=payload.get("anomaly_validation_only", False),
    )
    return _anomaly_tuple(result)


def run_anomaly_pytorch_single(
    candidate: AlgorithmCandidate, payload: dict[str, Any], split: DataSplit, trial_dir: Path
) -> tuple[dict[str, float], dict[str, float], int | None, np.ndarray, np.ndarray]:
    result = run_anomaly_pytorch(
        candidate,
        payload["x"],
        payload["y"],
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
    return _anomaly_tuple(result)


def run_sklearn_cv(
    candidate: AlgorithmCandidate, payload: dict[str, Any], fold: ValidationFold, fold_dir: Path, fold_seed: int
) -> tuple[dict[str, float] | None, dict[str, float], int | None, np.ndarray, list[str]]:
    task_type, x, y = payload["task_type"], payload["x"], payload["y"]
    training_events.emit("training_start", candidate=candidate.name, backend=candidate.backend, task_type=task_type.value)
    pipeline = build_pipeline(candidate, task_type, seed=fold_seed, workers=payload["workers"])
    pipeline.fit(x[fold.train], _fit_targets(task_type, y[fold.train]))
    val_prediction = _reshape_predictions(
        task_type, np.asarray(pipeline.predict(x[fold.val])), y[fold.val]
    )
    validation_metrics = evaluate_predictions(
        task_type, y[fold.val], val_prediction, payload["target_weights"]
    )
    parameter_count = estimate_parameter_count(pipeline)
    if payload["save_checkpoint"]:
        write_sklearn_artifact(fold_dir / "pipeline.joblib", pipeline, candidate)
    training_events.emit("training_end", candidate=candidate.name, backend=candidate.backend, parameter_count=parameter_count)
    return None, validation_metrics, parameter_count, val_prediction, []


def run_baseline_cv(
    candidate: AlgorithmCandidate, payload: dict[str, Any], fold: ValidationFold, fold_dir: Path, fold_seed: int
) -> tuple[None, dict[str, float], int, np.ndarray, list[str]]:
    task_type, x, y = payload["task_type"], payload["x"], payload["y"]
    training_events.emit("training_start", candidate=candidate.name, backend=candidate.backend, task_type=task_type.value)
    val_prediction = _last_value_forecast(x[fold.val], payload["forecast_target_indices"], y[fold.val].shape)
    validation_metrics = evaluate_predictions(task_type, y[fold.val], val_prediction, payload["target_weights"])
    json_dump(
        fold_dir / "artifact_manifest.json",
        {"backend": "baseline", "name": candidate.name, "parameter_count": 0},
    )
    training_events.emit("training_end", candidate=candidate.name, backend=candidate.backend, parameter_count=0)
    return None, validation_metrics, 0, val_prediction, []


def run_torch_cv(
    candidate: AlgorithmCandidate, payload: dict[str, Any], fold: ValidationFold, fold_dir: Path, fold_seed: int
) -> tuple[dict[str, float] | None, dict[str, float], int, np.ndarray, list[str]]:
    if payload["cv_protocol"] == "group_cv":
        fold_split, inner_val_groups = _inner_group_split(fold, payload["groups"], fold_seed)
    else:
        fold_split = _inner_time_split(fold)
        inner_val_groups = []
    inner_metrics, validation_metrics, parameter_count, _, val_prediction = run_torch_candidate(
        candidate,
        payload["task_type"],
        payload["x"],
        payload["y"],
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
    return inner_metrics, validation_metrics, parameter_count, val_prediction, inner_val_groups


_ATTACH_LOCK = RLock()
_ATTACHED = False


def attach_builtin_runners(scope: Any = None) -> None:
    """Bind the builtin runners to the builtin backend specs (idempotent).

    Attaching is a Cordis-style effect: pass ``scope`` and disposing it restores
    the runner-free specs, which is how tests can swap in fake backends.
    """
    global _ATTACHED
    with _ATTACH_LOCK:
        if _ATTACHED:
            return
        attach_runner("sklearn", runner=run_sklearn_single, cv_runner=run_sklearn_cv, scope=scope)
        attach_runner("baseline", runner=run_baseline_single, cv_runner=run_baseline_cv, scope=scope)
        attach_runner("pytorch", runner=run_torch_single, cv_runner=run_torch_cv, scope=scope)
        attach_runner("anomaly_baseline", runner=run_anomaly_baseline_single, scope=scope)
        attach_runner("anomaly_pytorch", runner=run_anomaly_pytorch_single, scope=scope)
        _ATTACHED = True


__all__ = [
    "attach_builtin_runners",
    "run_anomaly_baseline_single",
    "run_anomaly_pytorch_single",
    "run_baseline_cv",
    "run_baseline_single",
    "run_sklearn_cv",
    "run_sklearn_single",
    "run_torch_cv",
    "run_torch_single",
]
