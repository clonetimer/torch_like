"""Load a selected leaderboard artifact as a reusable inference bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from torch_like.automl.specs import TaskType
from torch_like.models.detection import DetectionBatch
from torch_like.models.registry import create_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str | None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    if expected is not None and _sha256(path) != expected:
        raise ValueError(f"Model artifact checksum mismatch: {path}")


def _stack_sequences(x: np.ndarray) -> np.ndarray:
    values = [np.asarray(sequence, dtype=np.float32) for sequence in x]
    if not values or len({len(sequence) for sequence in values}) != 1:
        raise ValueError("PyTorch model bundles require non-empty equal-length sequences")
    return np.stack(values)


def _transform_sequences(x: np.ndarray, preprocessor: dict[str, Any]) -> np.ndarray:
    values = _stack_sequences(x)
    feature_dim = values.shape[-1]
    flat = preprocessor["imputer"].transform(values.reshape(-1, feature_dim))
    scaler = preprocessor.get("scaler")
    if scaler is not None:
        flat = scaler.transform(flat)
    return flat.reshape(values.shape).astype(np.float32)


def _transform_tabular(x: np.ndarray, preprocessor: dict[str, Any]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("PyTorch tabular model bundles require a 2D feature matrix")
    transformed = preprocessor["imputer"].transform(values)
    scaler = preprocessor.get("scaler")
    if scaler is not None:
        transformed = scaler.transform(transformed)
    return transformed.astype(np.float32)


def _anomaly_window_features(x: np.ndarray) -> np.ndarray:
    positions = np.arange(x.shape[1], dtype=np.float64)
    centered = positions - positions.mean()
    denominator = max(float(np.square(centered).sum()), np.finfo(np.float64).eps)
    slope = np.einsum("l,nlc->nc", centered, x) / denominator
    return np.stack((x[:, -1], x.mean(axis=1), x.std(axis=1), slope), axis=-1)


@dataclass
class ModelBundle:
    """Trusted-local inference wrapper around one selected leaderboard pipeline."""

    run_dir: Path
    task_type: TaskType
    backend: str
    candidate: dict[str, Any]
    model: Any
    preprocessor: Any = None
    threshold: float | None = None
    device: torch.device = torch.device("cpu")
    target_constraint: str = "none"

    def _torch_output(self, x: np.ndarray) -> np.ndarray:
        transformed = (
            _transform_tabular(x, self.preprocessor)
            if self.task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION}
            else _transform_sequences(x, self.preprocessor)
        )
        values = torch.from_numpy(transformed).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            if self.task_type == TaskType.TIME_SERIES_FORECASTING:
                output = self.model(values, None, None, None)
            else:
                output = self.model(values)
            if self.task_type == TaskType.RUL_REGRESSION and self.target_constraint == "unit_interval":
                output = output.sigmoid()
        prediction = output.cpu().numpy()
        if self.task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.TIME_SERIES_CLASSIFICATION}:
            encoder = self.preprocessor.get("label_encoder")
            return encoder.inverse_transform(prediction.argmax(axis=-1))
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return (1.0 / (1.0 + np.exp(-prediction)) >= 0.5).astype(np.int64)
        target_scaler = self.preprocessor.get("target_scaler")
        if target_scaler is not None:
            shape = prediction.shape
            prediction = target_scaler.inverse_transform(prediction.reshape(-1, shape[-1])).reshape(shape)
        return prediction

    def score(self, x: np.ndarray) -> np.ndarray:
        """Return continuous anomaly scores for an anomaly-detection bundle."""

        if self.task_type != TaskType.ANOMALY_DETECTION:
            raise TypeError("score() is available only for anomaly-detection bundles")
        values = np.asarray(x, dtype=np.float32)
        scaler = self.preprocessor.get("scaler") if self.preprocessor is not None else None
        if scaler is not None:
            shape = values.shape
            values = scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
        if self.backend == "anomaly_pytorch":
            self.model.eval()
            with torch.inference_mode():
                batch = torch.from_numpy(np.ascontiguousarray(values)).to(self.device)
                return self.model.score_batch(DetectionBatch(x=batch), reduction="none").cpu().numpy()
        if self.candidate["name"] == "GlobalSTD":
            return np.abs(values[:, -1] - self.model["mean"]) / self.model["std"]
        if self.candidate["name"] == "IsolationForest":
            features = _anomaly_window_features(values)
            return np.stack(
                [-model.decision_function(features[:, channel]) for channel, model in enumerate(self.model["models"])],
                axis=1,
            )
        raise ValueError(f"Unsupported anomaly bundle: {self.candidate['name']}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels/targets, or thresholded detections for anomaly tasks."""

        if self.task_type == TaskType.ANOMALY_DETECTION:
            if self.threshold is None:
                raise ValueError("Anomaly bundle does not contain a selected threshold")
            return self.score(x) >= self.threshold
        if self.backend == "sklearn":
            prediction = np.asarray(self.model.predict(x))
            if self.task_type == TaskType.TIME_SERIES_FORECASTING:
                prediction_length = int(self.candidate["target_shape"][0])
                target_dim = int(self.candidate["target_shape"][1])
                return prediction.reshape(len(prediction), prediction_length, target_dim)
            return prediction
        if self.backend == "baseline" and self.candidate["name"] == "LastValue":
            target_indices = self.candidate["forecast_target_indices"]
            target_shape = self.candidate["target_shape"]
            last_values = np.stack([np.asarray(sequence)[-1, target_indices] for sequence in x])
            return np.repeat(last_values[:, None, :], int(target_shape[0]), axis=1)
        if self.backend == "pytorch":
            return self._torch_output(x)
        raise ValueError(f"Unsupported model bundle backend: {self.backend}")


def _domain(task_type: TaskType) -> str:
    if task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION}:
        return "tabular"
    if task_type == TaskType.TIME_SERIES_FORECASTING:
        return "forecasting"
    if task_type == TaskType.TIME_SERIES_CLASSIFICATION:
        return "diagnosis"
    if task_type == TaskType.RUL_REGRESSION:
        return "rul"
    if task_type == TaskType.ANOMALY_DETECTION:
        return "detection"
    raise ValueError(f"Task {task_type.value!r} does not use a torch_like model domain")


def load_best(run_dir: str | Path, *, device: str = "cpu") -> ModelBundle:
    """Load the selected artifact from a completed leaderboard run."""

    root = Path(run_dir)
    best_dir = root / "best_model"
    root_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    task_type = TaskType(root_config["task"]["type"])
    candidate = json.loads((best_dir / "config.json").read_text(encoding="utf-8"))["candidate"]
    manifest = json.loads((best_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    backend = candidate["backend"]
    torch_device = torch.device(device)
    if backend in {"sklearn", "anomaly_baseline"}:
        path = best_dir / "pipeline.joblib"
        _verify(path, manifest.get("sha256"))
        model = joblib.load(path)
    elif backend == "baseline":
        model = None
    elif backend in {"pytorch", "anomaly_pytorch"}:
        checkpoint_info = manifest.get("checkpoint")
        if isinstance(checkpoint_info, dict):
            path = best_dir / checkpoint_info["path"]
            expected = checkpoint_info.get("sha256")
        else:
            path = best_dir / "checkpoint.pt"
            expected = manifest.get("sha256")
        _verify(path, expected)
        checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
        if backend == "anomaly_pytorch":
            model = create_model(
                candidate["name"],
                domain="detection",
                input_dim=int(checkpoint["channels"]),
                seq_len=int(checkpoint["sequence_length"]),
                **candidate["parameters"],
            )
            model.load_state_dict(checkpoint["model"])
        else:
            model = create_model(
                checkpoint["model_name"],
                domain=_domain(task_type),
                **checkpoint["model_kwargs"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(torch_device)
    else:
        raise ValueError(f"Unsupported model bundle backend: {backend}")
    preprocessor_path = best_dir / "preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_path) if preprocessor_path.is_file() else None
    threshold_path = best_dir / "threshold.json"
    threshold = (
        float(json.loads(threshold_path.read_text(encoding="utf-8"))["value"]) if threshold_path.is_file() else None
    )
    dataset = root_config["dataset"]
    enriched_candidate = dict(candidate)
    if task_type == TaskType.TIME_SERIES_FORECASTING:
        features = dataset.get("feature_columns") or dataset["label_columns"]
        enriched_candidate["forecast_target_indices"] = [features.index(name) for name in dataset["label_columns"]]
        enriched_candidate["target_shape"] = [dataset["prediction_length"], len(dataset["label_columns"])]
    return ModelBundle(
        run_dir=root,
        task_type=task_type,
        backend=backend,
        candidate=enriched_candidate,
        model=model,
        preprocessor=preprocessor,
        threshold=threshold,
        device=torch_device,
        target_constraint=root_config["search"].get("torch_target_constraint", "none"),
    )


__all__ = ["ModelBundle", "load_best"]
