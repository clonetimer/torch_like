"""CPU-first native PyTorch fault classifier for the sealed diagnostic workflows.

The underlying InceptionTime, ROCKET or caller-provided nn.Module retains its
native class identity. Training labels are never supplied to prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from torch_like.diagnostics.classification import FaultLabelSchema
from torch_like.diagnostics.types import DiagnosticAlgorithmSpec, DiagnosticResult, SignalBatch


class TorchWindowFaultClassifier:
    """Train native time-series logits on development windows only."""

    def __init__(
        self,
        model_name: str = "InceptionTime",
        *,
        model: torch.nn.Module | None = None,
        model_kwargs: dict[str, Any] | None = None,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        if model_name not in {"InceptionTime", "ROCKET"}:
            raise ValueError("model_name must be InceptionTime or ROCKET")
        if model is not None and not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a native torch.nn.Module")
        if model is not None and model_kwargs:
            raise ValueError("model_kwargs cannot be combined with a provided model")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        self.model_name = model_name
        self.model = model
        self.model_kwargs = dict(model_kwargs or {})
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self._started = False
        self._schema: FaultLabelSchema | None = None
        self._channel_names: tuple[str, ...] | None = None
        self._units: tuple[str, ...] | None = None
        self._rate: float | None = None
        self._window_length: int | None = None

    @property
    def schema_(self) -> FaultLabelSchema:
        if self._schema is None:
            raise RuntimeError("fit on development labels before inference")
        return self._schema

    @staticmethod
    def _tensor(batch: SignalBatch) -> torch.Tensor:
        if not isinstance(batch, SignalBatch):
            raise TypeError("expected SignalBatch")
        if not np.isfinite(batch.values).all():
            raise ValueError("missing or nonfinite signal values require upstream preprocessing")
        values = np.asarray(batch.values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("float32 conversion produced nonfinite signal values")
        return torch.from_numpy(np.array(values, copy=True))

    @staticmethod
    def _checked_logits(logits: torch.Tensor, size: int, classes: int) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or logits.shape != (size, classes):
            raise ValueError("native model must return (batch, declared_classes) logits")
        if not torch.isfinite(logits).all().item():
            raise ValueError("native model emitted nonfinite logits")
        return logits

    def fit(
        self, reference: SignalBatch, labels: tuple[str, ...] | list[str], schema: FaultLabelSchema
    ) -> "TorchWindowFaultClassifier":
        if self._started:
            raise RuntimeError("classifier instance cannot be fitted a second time")
        if not isinstance(reference, SignalBatch) or not isinstance(schema, FaultLabelSchema):
            raise TypeError("fit requires SignalBatch and FaultLabelSchema")
        encoded = schema.encode(labels)
        if len(encoded) != reference.shape[0]:
            raise ValueError("training labels must align with input windows")
        if set(encoded.tolist()) != set(range(len(schema.labels))):
            raise ValueError("training must represent all declared fault classes")
        x = self._tensor(reference)
        y = torch.as_tensor(encoded.copy(), dtype=torch.long)
        self._started = True
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            if self.model is None:
                from torch_like.models.diagnosis import ROCKET, InceptionTime

                model_class = {"InceptionTime": InceptionTime, "ROCKET": ROCKET}[self.model_name]
                self.model = model_class(
                    input_dim=reference.shape[2],
                    num_classes=len(schema.labels),
                    seq_len=reference.shape[1],
                    **self.model_kwargs,
                )
            model = self.model.cpu()
            parameters = [param for param in model.parameters() if param.requires_grad]
            if not parameters:
                raise ValueError("native model has no trainable parameters")
            optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
            model.train()
            for _ in range(self.epochs):
                indices = torch.randperm(x.shape[0])
                for start in range(0, len(indices), self.batch_size):
                    selected = indices[start : start + self.batch_size]
                    logits = self._checked_logits(model(x[selected]), len(selected), len(schema.labels))
                    loss = F.cross_entropy(logits, y[selected])
                    if not torch.isfinite(loss).item():
                        raise ValueError("training loss is nonfinite")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            model.eval()
        self._schema = schema
        self._channel_names = reference.channel_names
        self._units = reference.units
        self._rate = reference.sample_rate_hz
        self._window_length = reference.shape[1]
        return self

    def predict_proba(self, batch: SignalBatch) -> DiagnosticResult:
        schema = self.schema_
        if not isinstance(batch, SignalBatch):
            raise TypeError("inference requires SignalBatch")
        if (batch.channel_names, batch.units, batch.sample_rate_hz, batch.shape[1]) != (
            self._channel_names,
            self._units,
            self._rate,
            self._window_length,
        ):
            raise ValueError("channel order, units, sample rate and window length must match training")
        x = self._tensor(batch)
        if self.model is None:
            raise RuntimeError("missing fitted native model")
        self.model.eval()
        probabilities = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], self.batch_size):
                piece = x[start : start + self.batch_size]
                logits = self._checked_logits(self.model(piece), len(piece), len(schema.labels))
                probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        values = np.concatenate(probabilities, axis=0).astype(np.float64)
        return DiagnosticResult(
            task="fault_classification",
            output_kind="class_probability",
            values=values,
            algorithm_name=type(self.model).__name__,
            label_names=schema.labels,
        )

    def predict(self, batch: SignalBatch) -> DiagnosticResult:
        result = self.predict_proba(batch)
        names = np.asarray(self.schema_.labels)
        return DiagnosticResult(
            task="fault_classification",
            output_kind="fault_label",
            values=names[np.argmax(result.values, axis=1)],
            algorithm_name=result.algorithm_name,
            label_names=self.schema_.labels,
        )


def register_torch_fault_classifiers(registry=None) -> None:
    """Opt-in registration; never mutate the native PyTorch model registry."""
    from torch_like.diagnostics.registry import diagnostic_registry

    target = diagnostic_registry if registry is None else registry
    target.register(
        DiagnosticAlgorithmSpec(
            name="TorchWindowFaultClassifier",
            backend="torch_like",
            task="fault_classification",
            output_kind="class_probability",
            operations=("fit", "predict"),
        ),
        TorchWindowFaultClassifier,
    )
