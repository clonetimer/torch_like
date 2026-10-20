"""Uniform user-data adapter for existing native PyTorch RUL models.

The original RULModel and torch.nn.Module interfaces remain untouched.
Models are resolved lazily so an algorithm-only wheel can ship a strict subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, util
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from torch_like.diagnostics.types import DiagnosticResult, SignalBatch

_RUL_MODELS = {
    "TCNRUL": "torch_like.models.rul_backbones",
    "PatchTSTRUL": "torch_like.models.rul_backbones",
    "TransformerRUL": "torch_like.models.rul_backbones",
    "THGNN": "torch_like.models.thgnn",
    "InformerRUL": "torch_like.models.forecasting.rul_adapters",
    "AutoformerRUL": "torch_like.models.forecasting.rul_adapters",
    "LogTransRUL": "torch_like.models.forecasting.rul_adapters",
}


def list_rul_models() -> tuple[str, ...]:
    """Native models physically available in this installed distribution."""
    available = []
    for name, module_path in _RUL_MODELS.items():
        try:
            found = util.find_spec(module_path)
        except (ImportError, ModuleNotFoundError):
            found = None
        if found is not None:
            available.append(name)
    return tuple(available)


@dataclass(frozen=True)
class RULForecast:
    """Nonnegative diagnostic RUL values with an explicit caller-declared unit."""

    result: DiagnosticResult
    unit: str

    def __post_init__(self) -> None:
        if self.result.task.value != "rul_regression" or self.result.output_kind.value != "rul_value":
            raise ValueError("RULForecast requires rul_regression/rul_value output")
        if self.unit not in {"cycles", "hours", "seconds"}:
            raise ValueError("unit must be cycles, hours or seconds")

    @property
    def values(self) -> np.ndarray:
        return self.result.values


class RULRegressor:
    """Direct fit/predict interface around the existing native RUL network.

    The caller supplies RUL targets and their unit, not a failure threshold
    inferred by the library. Training is a minimal supervised Adam loop, not
    an evaluation pipeline. Negative raw predictions must be handled explicitly.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        unit: str,
        negative_policy: str = "reject",
        ready: bool = False,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a native torch.nn.Module")
        if unit not in {"cycles", "hours", "seconds"}:
            raise ValueError("unit must be cycles, hours or seconds")
        if negative_policy not in {"reject", "clip"}:
            raise ValueError("negative_policy must be reject or clip")
        if not isinstance(ready, bool):
            raise TypeError("ready must be a bool")
        self.model = model
        self.unit = unit
        self.negative_policy = negative_policy
        self._ready = ready
        self._input_metadata: tuple[tuple[str, ...], tuple[str, ...] | None, float] | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    def mark_ready(self) -> "RULRegressor":
        """Explicitly acknowledge externally trained/restored native weights."""
        self._ready = True
        return self

    def _inputs(
        self, signals: SignalBatch, x_mark: np.ndarray | torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(signals, SignalBatch):
            raise TypeError("RUL inference/training requires SignalBatch")
        if not np.isfinite(signals.values).all():
            raise ValueError("RUL inputs must be finite; caller handles imputation")
        expected_channels = getattr(self.model, "enc_in", None)
        expected_length = getattr(self.model, "seq_len", None)
        if expected_channels is not None and signals.shape[2] != expected_channels:
            raise ValueError("RUL channel count does not match native model enc_in")
        if expected_length is not None and signals.shape[1] != expected_length:
            raise ValueError("RUL window length does not match native model seq_len")
        if self._input_metadata is not None:
            if (signals.channel_names, signals.units, signals.sample_rate_hz) != self._input_metadata:
                raise ValueError("channel names, units and sampling rate must match fitted input")
        try:
            parameter = next(self.model.parameters())
        except StopIteration as exc:
            raise ValueError("native RUL model must contain trainable parameters") from exc
        data = np.asarray(signals.values, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("RUL float32 conversion overflowed")
        x = torch.from_numpy(np.array(data, copy=True)).to(device=parameter.device, dtype=parameter.dtype)
        markers = None
        if x_mark is not None:
            markers = torch.as_tensor(x_mark, device=parameter.device, dtype=parameter.dtype)
            if markers.ndim != 3 or markers.shape[:2] != x.shape[:2] or markers.shape[2] < 1:
                raise ValueError("x_mark requires shape (samples, time, marker_features)")
            if not torch.isfinite(markers).all().item():
                raise ValueError("x_mark requires finite numeric values")
        return x, markers

    def fit(
        self,
        signals: SignalBatch,
        targets: np.ndarray | torch.Tensor,
        *,
        x_mark: np.ndarray | torch.Tensor | None = None,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        loss: str = "mse",
        seed: int = 0,
    ) -> "RULRegressor":
        if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive integers")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if loss not in {"mse", "mae", "huber"}:
            raise ValueError("loss must be mse, mae or huber")
        x, marks = self._inputs(signals, x_mark)
        labels = np.asarray(targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets)
        if labels.shape != (x.shape[0],) or not np.isfinite(labels).all() or np.any(labels < 0):
            raise ValueError("RUL targets must be finite nonnegative shape (samples,)")
        labels = torch.as_tensor(labels.copy(), device=x.device, dtype=x.dtype)
        params = [param for param in self.model.parameters() if param.requires_grad]
        if not params:
            raise ValueError("RUL model has no trainable parameters")
        optimizer = torch.optim.Adam(params, lr=float(learning_rate))
        self._ready = False
        self.model.train()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for _ in range(epochs):
                indices = torch.randperm(len(x))
                for start in range(0, len(indices), batch_size):
                    part = indices[start : start + batch_size].to(x.device)
                    raw = self.model(x[part], None if marks is None else marks[part])
                    if raw.shape != (len(part),) or not torch.isfinite(raw).all().item():
                        raise ValueError("native RUL model must output finite (samples,) values")
                    if loss == "mse":
                        objective = F.mse_loss(raw, labels[part])
                    elif loss == "mae":
                        objective = F.l1_loss(raw, labels[part])
                    else:
                        objective = F.huber_loss(raw, labels[part])
                    if not torch.isfinite(objective).item():
                        raise ValueError("RUL training loss is nonfinite")
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    optimizer.step()
        self.model.eval()
        self._input_metadata = (signals.channel_names, signals.units, signals.sample_rate_hz)
        self._ready = True
        return self

    def predict(self, signals: SignalBatch, *, x_mark: np.ndarray | torch.Tensor | None = None) -> DiagnosticResult:
        if not self._ready:
            raise RuntimeError("RUL model must be fitted or explicitly marked ready")
        x, markers = self._inputs(signals, x_mark)
        self.model.eval()
        with torch.inference_mode():
            raw = self.model(x, markers)
            if not isinstance(raw, torch.Tensor) or raw.shape != (len(x),):
                raise ValueError("native RUL model must produce (samples,) predictions")
            values = raw.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("native RUL model produced nonfinite estimates")
        if np.any(values < 0):
            if self.negative_policy == "reject":
                raise ValueError("negative RUL estimates; select negative_policy='clip' explicitly")
            values = np.maximum(values, 0)
        return DiagnosticResult(
            task="rul_regression",
            output_kind="rul_value",
            values=values,
            algorithm_name=type(self.model).__name__,
        )

    def predict_with_unit(
        self, signals: SignalBatch, *, x_mark: np.ndarray | torch.Tensor | None = None
    ) -> RULForecast:
        return RULForecast(self.predict(signals, x_mark=x_mark), self.unit)


def create_rul_regressor(name: str, *, unit: str, negative_policy: str = "reject", **model_kwargs: Any) -> RULRegressor:
    """Factory for installed native model classes; never creates a surrogate."""
    if name not in _RUL_MODELS:
        raise KeyError(f"unknown RUL model {name!r}")
    module_path = _RUL_MODELS[name]
    try:
        module = import_module(module_path)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValueError(f"{name} is not included in this algorithm distribution") from exc
    return RULRegressor(
        getattr(module, name)(**model_kwargs),
        unit=unit,
        negative_policy=negative_policy,
    )


__all__ = ["RULForecast", "RULRegressor", "create_rul_regressor", "list_rul_models"]
