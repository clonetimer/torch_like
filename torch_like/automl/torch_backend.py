"""Portable PyTorch trial adapter for selected torch_like task models."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from torch_like.automl.events import training_events as training_events
from torch_like.automl.sklearn_backend import AlgorithmCandidate, evaluate_predictions
from torch_like.automl.specs import TaskType
from torch_like.contracts import TaskKind, get_model_spec
from torch_like.data import DataLoaderConfig, TensorDataAdapter
from torch_like.models.registry import create_model
from torch_like.training import Trainer, TrainerConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stack_sequences(x: np.ndarray) -> np.ndarray:
    lengths = {len(sequence) for sequence in x}
    if len(lengths) != 1:
        raise ValueError("PyTorch v1 candidates require equal-length sequences")
    return np.stack([np.asarray(sequence, dtype=np.float32) for sequence in x])


def _fit_transform_sequences(x: np.ndarray, train_indices: np.ndarray, preprocessing: str):
    values = _stack_sequences(x)
    feature_dim = values.shape[-1]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(values[train_indices].reshape(-1, feature_dim))
    flat = imputer.transform(values.reshape(-1, feature_dim))
    if preprocessing == "standard":
        scaler: Any = StandardScaler()
    elif preprocessing == "robust":
        scaler = RobustScaler()
    elif preprocessing == "minmax":
        scaler = MinMaxScaler()
    elif preprocessing == "none":
        scaler = None
    else:
        raise ValueError(f"Unsupported preprocessing: {preprocessing}")
    if scaler is not None:
        scaler.fit(imputer.transform(values[train_indices].reshape(-1, feature_dim)))
        flat = scaler.transform(flat)
    transformed = flat.reshape(values.shape).astype(np.float32)
    return transformed, {"imputer": imputer, "scaler": scaler}


def _fit_transform_tabular(x: np.ndarray, train_indices: np.ndarray, preprocessing: str):
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("PyTorch tabular candidates require a 2D feature matrix")
    imputer = SimpleImputer(strategy="median").fit(values[train_indices])
    transformed = imputer.transform(values)
    if preprocessing == "standard":
        scaler: Any = StandardScaler()
    elif preprocessing == "robust":
        scaler = RobustScaler()
    elif preprocessing == "minmax":
        scaler = MinMaxScaler()
    elif preprocessing == "none":
        scaler = None
    else:
        raise ValueError(f"Unsupported preprocessing: {preprocessing}")
    if scaler is not None:
        scaler.fit(imputer.transform(values[train_indices]))
        transformed = scaler.transform(transformed)
    return transformed.astype(np.float32), {"imputer": imputer, "scaler": scaler}


def _build_model(candidate: AlgorithmCandidate, task_type: TaskType, sequence_length: int, feature_dim: int, y):
    """Build a torch_like AutoML candidate through the shared registry/factory contract.

    Task adapters still own task-derived constructor arguments (for example
    ``num_classes`` or ``pred_len``), but class discovery no longer maintains a
    second hard-coded import table.
    """

    parameters = dict(candidate.parameters)
    spec = get_model_spec(candidate.name, backend="torch_like")
    compatible_task_kinds = {
        TaskType.TABULAR_CLASSIFICATION: {TaskKind.TABULAR_PREDICTION},
        TaskType.MULTILABEL_CLASSIFICATION: {TaskKind.TABULAR_PREDICTION},
        TaskType.TIME_SERIES_FORECASTING: {TaskKind.TIME_SERIES_FORECASTING, TaskKind.PROBABILISTIC_FORECASTING},
        TaskType.TIME_SERIES_CLASSIFICATION: {TaskKind.TIME_SERIES_CLASSIFICATION},
        TaskType.RUL_REGRESSION: {TaskKind.RUL_REGRESSION},
    }
    expected = compatible_task_kinds.get(task_type)
    if expected is None or spec.task.kind not in expected:
        expected_values = [] if expected is None else sorted(kind.value for kind in expected)
        raise ValueError(
            f"AutoML task {task_type.value!r} is incompatible with {candidate.name!r} contract "
            f"{spec.task.kind.value!r}; expected one of {expected_values}"
        )

    if task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION}:
        kwargs = {
            "input_dim": feature_dim,
            "output_dim": int(y.shape[1])
            if task_type == TaskType.MULTILABEL_CLASSIFICATION
            else int(len(np.unique(y))),
            **parameters,
        }
    elif task_type == TaskType.TIME_SERIES_FORECASTING:
        target_dim = int(y.shape[-1])
        kwargs = {
            "enc_in": feature_dim,
            "dec_in": target_dim,
            "c_out": target_dim,
            "seq_len": sequence_length,
            "label_len": 0,
            "pred_len": int(y.shape[1]),
            **parameters,
        }
    elif task_type == TaskType.TIME_SERIES_CLASSIFICATION:
        kwargs = {
            "input_dim": feature_dim,
            "num_classes": int(len(np.unique(y))),
            "seq_len": sequence_length,
            **parameters,
        }
    elif task_type == TaskType.RUL_REGRESSION:
        kwargs = {"enc_in": feature_dim, "seq_len": sequence_length, **parameters}
    else:  # guarded by compatible_task_kinds above
        raise ValueError(f"Unsupported PyTorch task: {task_type.value}")

    return create_model(candidate.name, domain=spec.domain, **kwargs), kwargs


def run_torch_candidate(
    candidate: AlgorithmCandidate,
    task_type: TaskType,
    x: np.ndarray,
    y: np.ndarray,
    split,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    workers: int,
    trial_dir: Path,
    save_checkpoint: bool,
    device: str = "cpu",
    final_only: bool = False,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    loss_name: str = "mse",
    grad_clip_norm: float | None = None,
    lr_scheduler_name: str = "none",
    early_stopping_patience: int | None = None,
    min_delta: float = 0.0,
    restore_best: bool = False,
    target_constraint: str = "none",
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    torch_device = torch.device(resolved_device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {resolved_device}")
    if task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION}:
        transformed, preprocessor = _fit_transform_tabular(x, split.train, candidate.preprocessing)
        sequence_length, feature_dim = 1, transformed.shape[1]
    else:
        transformed, preprocessor = _fit_transform_sequences(x, split.train, candidate.preprocessing)
        sequence_length, feature_dim = transformed.shape[1], transformed.shape[2]
    labels = np.asarray(y)
    label_encoder = None
    target_scaler = None
    if task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.TIME_SERIES_CLASSIFICATION}:
        label_encoder = LabelEncoder().fit(labels[split.train])
        encoded = label_encoder.transform(labels).astype(np.int64)
        target_tensor = torch.from_numpy(encoded)
    elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
        encoded = labels.astype(np.float32)
        target_tensor = torch.from_numpy(encoded)
    elif task_type == TaskType.TIME_SERIES_FORECASTING:
        target_scaler = StandardScaler().fit(labels[split.train].reshape(-1, labels.shape[-1]))
        encoded = target_scaler.transform(labels.reshape(-1, labels.shape[-1])).reshape(labels.shape).astype(np.float32)
        target_tensor = torch.from_numpy(encoded)
    else:
        encoded = labels.astype(np.float32)
        target_tensor = torch.from_numpy(encoded)
    input_tensor = torch.from_numpy(transformed)
    model, model_kwargs = _build_model(
        candidate,
        task_type,
        sequence_length,
        feature_dim,
        labels[split.train],
    )
    model = model.to(torch_device)

    def forward_model(batch_x: torch.Tensor) -> torch.Tensor:
        if task_type == TaskType.TIME_SERIES_FORECASTING:
            return model(batch_x, None, None, None)
        return model(batch_x)

    data_adapter = TensorDataAdapter()
    train_loader = data_adapter.loader(
        input_tensor,
        target_tensor,
        indices=split.train,
        config=DataLoaderConfig(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch_device.type == "cuda",
            seed=seed,
        ),
    )

    def constrain(output: torch.Tensor) -> torch.Tensor:
        if task_type == TaskType.RUL_REGRESSION and target_constraint == "unit_interval":
            return output.sigmoid()
        return output

    def loss_step(output: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        if task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.TIME_SERIES_CLASSIFICATION}:
            return F.cross_entropy(output, batch_y)
        if task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return F.binary_cross_entropy_with_logits(output, batch_y.to(device=output.device, dtype=output.dtype))
        output = constrain(output)
        if loss_name == "mae":
            return F.l1_loss(output, batch_y)
        if loss_name == "huber":
            return F.huber_loss(output, batch_y)
        return F.mse_loss(output, batch_y)

    def forward_loss(batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        if task_type == TaskType.TABULAR_CLASSIFICATION:
            return model.loss_step(model.make_tabular_batch(batch_x, y=batch_y))
        return loss_step(forward_model(batch_x), batch_y)

    validation_loader = None
    if not final_only:
        validation_loader = data_adapter.loader(
            input_tensor,
            target_tensor,
            indices=split.val,
            config=DataLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch_device.type == "cuda",
            ),
        )

    trainer = Trainer(
        model,
        TrainerConfig(
            epochs=epochs,
            device=resolved_device,
            optimizer=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler_name,
            grad_clip_norm=grad_clip_norm,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            restore_best=restore_best,
            checkpoint_dir=trial_dir if save_checkpoint else None,
            save_last=save_checkpoint,
            save_best=save_checkpoint,
        ),
        loss_fn=lambda _model, batch: forward_loss(batch[0], batch[1]),
        event_context={
            "candidate": candidate.name,
            "backend": candidate.backend,
            "task_type": task_type.value,
        },
    )
    training_result = trainer.fit(train_loader, validation_loader)
    history = [item.to_dict() for item in training_result.history]
    training_summary = {
        "epochs_requested": epochs,
        "epochs_ran": training_result.epochs_ran,
        "best_epoch": training_result.best_epoch,
        "best_monitored_loss": training_result.best_monitored_loss,
        "stopped_early": training_result.stopped_early,
        "monitor": training_result.monitor,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "loss": (
            "binary_cross_entropy_with_logits"
            if task_type == TaskType.MULTILABEL_CLASSIFICATION
            else "cross_entropy"
            if task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.TIME_SERIES_CLASSIFICATION}
            else loss_name
        ),
        "grad_clip_norm": grad_clip_norm,
        "lr_scheduler": lr_scheduler_name,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
        "restore_best": restore_best,
        "target_constraint": target_constraint,
        "training_runtime": "torch_like.training.Trainer",
    }
    (trial_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (trial_dir / "training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def predict_indices(indices: np.ndarray) -> torch.Tensor:
        loader = data_adapter.loader(
            input_tensor,
            indices=indices,
            config=DataLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch_device.type == "cuda",
            ),
        )
        outputs = []
        with torch.inference_mode():
            for (batch_x,) in loader:
                outputs.append(constrain(forward_model(batch_x.to(torch_device, non_blocking=True))).cpu())
        return torch.cat(outputs)

    model.eval()
    test_output = predict_indices(split.test)
    val_output = test_output if final_only else predict_indices(split.val)
    if task_type in {TaskType.TABULAR_CLASSIFICATION, TaskType.TIME_SERIES_CLASSIFICATION}:
        val_prediction = label_encoder.inverse_transform(val_output.argmax(dim=-1).numpy())
        test_prediction = label_encoder.inverse_transform(test_output.argmax(dim=-1).numpy())
    elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
        val_prediction = (torch.sigmoid(val_output).numpy() >= 0.5).astype(np.int64)
        test_prediction = (torch.sigmoid(test_output).numpy() >= 0.5).astype(np.int64)
    else:
        val_prediction = val_output.numpy()
        test_prediction = test_output.numpy()
        if target_scaler is not None:
            val_prediction = target_scaler.inverse_transform(
                val_prediction.reshape(-1, val_prediction.shape[-1])
            ).reshape(val_prediction.shape)
            test_prediction = target_scaler.inverse_transform(
                test_prediction.reshape(-1, test_prediction.shape[-1])
            ).reshape(test_prediction.shape)
    validation_metrics = (
        {} if final_only else evaluate_predictions(task_type, labels[split.val], np.asarray(val_prediction))
    )
    test_metrics = evaluate_predictions(task_type, labels[split.test], np.asarray(test_prediction))

    preprocessor_path = trial_dir / "preprocessor.joblib"
    joblib.dump({**preprocessor, "label_encoder": label_encoder, "target_scaler": target_scaler}, preprocessor_path)
    checkpoint_path = trial_dir / "last_checkpoint.pt"
    manifest = {
        "backend": "pytorch",
        "model_name": candidate.name,
        "model_kwargs": model_kwargs,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(torch_device),
        "seed": seed,
        "scikit_learn": sklearn.__version__,
        "preprocessor": {"path": preprocessor_path.name, "sha256": _sha256(preprocessor_path)},
        "checkpoint": (
            {"path": checkpoint_path.name, "sha256": _sha256(checkpoint_path)} if checkpoint_path.exists() else None
        ),
        "best_checkpoint": (
            {
                "path": "best_checkpoint.pt",
                "sha256": _sha256(trial_dir / "best_checkpoint.pt"),
            }
            if (trial_dir / "best_checkpoint.pt").exists()
            else None
        ),
        "training_summary": training_summary,
    }
    (trial_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return validation_metrics, test_metrics, int(parameter_count), val_prediction, test_prediction
