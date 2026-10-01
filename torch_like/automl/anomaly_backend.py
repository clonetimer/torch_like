"""Classical and PyTorch execution backend for anomaly leaderboard candidates."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from torch_like.automl.anomaly_metrics import evaluate_anomaly_scores, select_anomaly_threshold
from torch_like.automl.data import DataSplit
from torch_like.automl.events import training_events
from torch_like.automl.sklearn_backend import AlgorithmCandidate
from torch_like.automl.specs import TaskType
from torch_like.models.detection import DetectionBatch
from torch_like.models.registry import create_model


@dataclass(frozen=True)
class AnomalyRunResult:
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    parameter_count: int
    threshold: float
    validation_scores: np.ndarray
    test_scores: np.ndarray | None


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "none":
        return None
    raise ValueError(f"Unsupported anomaly preprocessing: {name}")


def _preprocess(x: np.ndarray, train_indices: np.ndarray, name: str, trial_dir: Path) -> np.ndarray:
    scaler = _scaler(name)
    if scaler is None:
        joblib.dump({"name": "none", "scaler": None}, trial_dir / "preprocessor.joblib")
        return np.asarray(x, dtype=np.float32)
    channels = x.shape[-1]
    scaler.fit(x[train_indices].reshape(-1, channels))
    transformed = scaler.transform(x.reshape(-1, channels)).reshape(x.shape).astype(np.float32)
    joblib.dump({"name": name, "scaler": scaler}, trial_dir / "preprocessor.joblib")
    return transformed


def _window_features(x: np.ndarray) -> np.ndarray:
    positions = np.arange(x.shape[1], dtype=np.float64)
    centered = positions - positions.mean()
    denominator = float(np.square(centered).sum())
    slope = np.einsum("l,nlc->nc", centered, x) / max(denominator, np.finfo(np.float64).eps)
    return np.stack((x[:, -1], x.mean(axis=1), x.std(axis=1), slope), axis=-1)


def _run_baseline(
    candidate: AlgorithmCandidate,
    x: np.ndarray,
    split: DataSplit,
    seed: int,
    workers: int,
    trial_dir: Path,
    validation_only: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    train = x[split.train]
    if candidate.name == "GlobalSTD":
        mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
        std = train.reshape(-1, train.shape[-1]).std(axis=0)
        std = np.maximum(std, np.finfo(np.float32).eps)

        def score(values: np.ndarray) -> np.ndarray:
            return np.abs(values[:, -1] - mean) / std

        model = {"name": candidate.name, "mean": mean, "std": std}
        parameter_count = int(mean.size + std.size)
    elif candidate.name == "IsolationForest":
        train_features = _window_features(train)
        models = []
        for channel in range(train.shape[-1]):
            model = IsolationForest(
                n_estimators=int(candidate.parameters.get("n_estimators", 50)),
                max_samples=min(int(candidate.parameters.get("max_samples", 4096)), len(train)),
                contamination="auto",
                random_state=seed + channel,
                n_jobs=workers,
            )
            model.fit(train_features[:, channel])
            models.append(model)

        def score(values: np.ndarray) -> np.ndarray:
            features = _window_features(values)
            return np.stack(
                [-model.decision_function(features[:, channel]) for channel, model in enumerate(models)], axis=1
            )

        model = {"name": candidate.name, "models": models}
        parameter_count = int(sum(tree.tree_.node_count for item in models for tree in item.estimators_))
    else:
        raise ValueError(f"Unsupported anomaly baseline: {candidate.name}")
    validation_scores = score(x[split.val]).astype(np.float64)
    test_scores = None if validation_only else score(x[split.test]).astype(np.float64)
    pipeline_path = trial_dir / "pipeline.joblib"
    joblib.dump(model, pipeline_path)
    _json_dump(
        trial_dir / "artifact_manifest.json",
        {
            "backend": "sklearn",
            "format": "joblib",
            "trusted_local_only": True,
            "sha256": _sha256(pipeline_path),
            "parameter_count": parameter_count,
        },
    )
    return validation_scores, test_scores, parameter_count


def _build_model(candidate: AlgorithmCandidate, sequence_length: int, channels: int):
    return create_model(
        candidate.name,
        domain="detection",
        input_dim=channels,
        seq_len=sequence_length,
        **dict(candidate.parameters),
    )


def _run_pytorch(
    candidate: AlgorithmCandidate,
    x: np.ndarray,
    split: DataSplit,
    *,
    seed: int,
    workers: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    grad_clip_norm: float | None,
    trial_dir: Path,
    save_checkpoint: bool,
    validation_only: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, workers))
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    model = _build_model(candidate, x.shape[1], x.shape[2]).to(resolved_device)
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_tensor = torch.from_numpy(np.ascontiguousarray(x[split.train]))
    generator = torch.Generator().manual_seed(seed)
    # Trials already execute in an isolated subprocess. Spawning nested
    # DataLoader workers from a process that has initialized PyTorch/BLAS
    # threads is fragile on Python 3.13+ and adds little for the small window
    # tensors used here. CPU parallelism is controlled by torch.set_num_threads.
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=resolved_device.type == "cuda",
        generator=generator,
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.fit_step(DetectionBatch(x=batch_x))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN or infinite anomaly training loss at epoch {epoch}")
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    _json_dump(trial_dir / "training_history.json", history)

    def score(indices: np.ndarray) -> np.ndarray:
        model.eval()
        values = torch.from_numpy(np.ascontiguousarray(x[indices]))
        score_loader = DataLoader(
            TensorDataset(values),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=resolved_device.type == "cuda",
        )
        outputs = []
        with torch.inference_mode():
            for (batch_x,) in score_loader:
                batch_x = batch_x.to(resolved_device, non_blocking=True)
                outputs.append(model.score_batch(DetectionBatch(x=batch_x), reduction="none").cpu())
        return torch.cat(outputs).numpy().astype(np.float64)

    validation_scores = score(split.val)
    test_scores = None if validation_only else score(split.test)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if save_checkpoint:
        checkpoint_path = trial_dir / "checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "candidate": candidate.to_dict(),
                "sequence_length": x.shape[1],
                "channels": x.shape[2],
                "seed": seed,
                "epochs": epochs,
            },
            checkpoint_path,
        )
        _json_dump(
            trial_dir / "artifact_manifest.json",
            {
                "backend": "pytorch",
                "format": "state_dict",
                "sha256": _sha256(checkpoint_path),
                "device": str(resolved_device),
                "parameter_count": parameter_count,
            },
        )
    return validation_scores, test_scores, parameter_count


def _score_frame(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    channels: tuple[str, ...],
    threshold: float,
) -> pd.DataFrame:
    payload: dict[str, Any] = {"timestamp": sample_ids}
    for index, channel in enumerate(channels):
        payload[f"label_{channel}"] = labels[:, index]
        payload[f"score_{channel}"] = scores[:, index]
        payload[f"detection_{channel}"] = (scores[:, index] >= threshold).astype(np.uint8)
    return pd.DataFrame(payload)


def _finalize_anomaly_run(
    y: np.ndarray,
    split: DataSplit,
    val_scores: np.ndarray,
    test_scores: np.ndarray | None,
    parameter_count: int,
    *,
    sample_ids: np.ndarray,
    channels: tuple[str, ...],
    threshold_candidates: int,
    trial_dir: Path,
    save_predictions: bool,
) -> AnomalyRunResult:
    """Shared tail: pick the threshold, score, persist artifacts, package the result."""
    threshold, validation_metrics = select_anomaly_threshold(y[split.val], val_scores, candidates=threshold_candidates)
    test_metrics = {} if test_scores is None else evaluate_anomaly_scores(y[split.test], test_scores, threshold)
    _json_dump(
        trial_dir / "threshold.json",
        {
            "value": threshold,
            "selected_on": "validation",
            "metric": "event_f0_5_corrected",
            "candidates": threshold_candidates,
        },
    )
    if save_predictions:
        _score_frame(sample_ids[split.val], y[split.val], val_scores, channels, threshold).to_csv(
            trial_dir / "validation_scores.csv", index=False
        )
        if test_scores is not None:
            _score_frame(sample_ids[split.test], y[split.test], test_scores, channels, threshold).to_csv(
                trial_dir / "predictions.csv", index=False
            )
    return AnomalyRunResult(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        parameter_count=parameter_count,
        threshold=threshold,
        validation_scores=val_scores,
        test_scores=test_scores,
    )


def run_anomaly_baseline(
    candidate: AlgorithmCandidate,
    x: np.ndarray,
    y: np.ndarray,
    split: DataSplit,
    *,
    sample_ids: np.ndarray,
    channels: tuple[str, ...],
    seed: int,
    workers: int,
    threshold_candidates: int,
    trial_dir: Path,
    save_predictions: bool,
    validation_only: bool,
) -> AnomalyRunResult:
    """Score with a classical detector -- bound to the ``anomaly_baseline`` backend."""
    training_events.emit(
        "training_start", candidate=candidate.name, backend=candidate.backend, task_type=TaskType.ANOMALY_DETECTION.value
    )
    transformed = _preprocess(x, split.train, candidate.preprocessing, trial_dir)
    val_scores, test_scores, parameter_count = _run_baseline(
        candidate, transformed, split, seed, workers, trial_dir, validation_only
    )
    training_events.emit("training_end", candidate=candidate.name, backend=candidate.backend, parameter_count=parameter_count)
    return _finalize_anomaly_run(
        y,
        split,
        val_scores,
        test_scores,
        parameter_count,
        sample_ids=sample_ids,
        channels=channels,
        threshold_candidates=threshold_candidates,
        trial_dir=trial_dir,
        save_predictions=save_predictions,
    )


def run_anomaly_pytorch(
    candidate: AlgorithmCandidate,
    x: np.ndarray,
    y: np.ndarray,
    split: DataSplit,
    *,
    sample_ids: np.ndarray,
    channels: tuple[str, ...],
    seed: int,
    workers: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    grad_clip_norm: float | None,
    threshold_candidates: int,
    trial_dir: Path,
    save_checkpoint: bool,
    save_predictions: bool,
    validation_only: bool,
) -> AnomalyRunResult:
    """Score with a torch detection model -- bound to the ``anomaly_pytorch`` backend."""
    training_events.emit(
        "training_start", candidate=candidate.name, backend=candidate.backend, task_type=TaskType.ANOMALY_DETECTION.value
    )
    transformed = _preprocess(x, split.train, candidate.preprocessing, trial_dir)
    val_scores, test_scores, parameter_count = _run_pytorch(
        candidate,
        transformed,
        split,
        seed=seed,
        workers=workers,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        trial_dir=trial_dir,
        save_checkpoint=save_checkpoint,
        validation_only=validation_only,
    )
    training_events.emit("training_end", candidate=candidate.name, backend=candidate.backend, parameter_count=parameter_count)
    return _finalize_anomaly_run(
        y,
        split,
        val_scores,
        test_scores,
        parameter_count,
        sample_ids=sample_ids,
        channels=channels,
        threshold_candidates=threshold_candidates,
        trial_dir=trial_dir,
        save_predictions=save_predictions,
    )


__all__ = ["AnomalyRunResult", "run_anomaly_baseline", "run_anomaly_pytorch"]
