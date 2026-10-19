"""Batch adaptation between generic loader output and domain model batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from torch_like.contracts import TaskKind, get_model_spec
from torch_like.training.data import infer_batch_size


@dataclass(frozen=True)
class AdaptedBatch:
    """One training/evaluation batch after domain adaptation.

    ``model_batch`` is passed to the model/runtime. ``targets`` is kept separate
    so evaluation and supervised losses do not need to overload domain batch
    fields. ``raw_batch`` remains available for task-specific code.
    """

    model_batch: Any
    targets: Any = None
    sample_count: int | None = None
    raw_batch: Any = None

    def resolved_sample_count(self) -> int:
        if self.sample_count is not None:
            return int(self.sample_count)
        for candidate in (self.targets, self.model_batch, self.raw_batch):
            if candidate is None:
                continue
            try:
                return infer_batch_size(candidate)
            except ValueError:
                continue
        raise ValueError("cannot infer sample count from adapted batch")


@runtime_checkable
class BatchAdapter(Protocol):
    """Convert one loader batch into the model's domain batch contract."""

    def adapt(self, model: Any, raw_batch: Any) -> AdaptedBatch: ...


class IdentityBatchAdapter:
    """Pass loader output through unchanged while preserving a common wrapper."""

    def adapt(self, model: Any, raw_batch: Any) -> AdaptedBatch:
        del model
        targets = None
        if isinstance(raw_batch, (tuple, list)) and len(raw_batch) >= 2:
            targets = raw_batch[1]
        return AdaptedBatch(model_batch=raw_batch, targets=targets, raw_batch=raw_batch)


class CallableBatchAdapter:
    """Compatibility wrapper for the 0.6 callable ``batch_adapter`` API."""

    def __init__(self, function: Callable[[Any], Any]):
        self.function = function

    def adapt(self, model: Any, raw_batch: Any) -> AdaptedBatch:
        del model
        model_batch = self.function(raw_batch)
        targets = _target_from_model_batch(model_batch)
        if targets is None and isinstance(raw_batch, (tuple, list)) and len(raw_batch) >= 2:
            targets = raw_batch[1]
        return AdaptedBatch(model_batch=model_batch, targets=targets, raw_batch=raw_batch)


class ForecastingBatchAdapter:
    """Adapt tuple or task-data dictionary batches to ``ForecastingBatch``.

    Dict batches produced by :mod:`torch_like.data.task_data` keep decoder
    targets separate from decoder inputs. This adapter constructs the decoder
    input without leaking future targets and exposes the future target through
    :class:`AdaptedBatch` for loss/metric evaluation.

    ``label_len`` and ``pred_len`` normally come from the model. Explicit
    overrides are useful for runner compatibility and compiled module wrappers
    whose attributes may not be statically visible.
    """

    def __init__(self, *, label_len: int | None = None, pred_len: int | None = None):
        if label_len is not None and label_len < 0:
            raise ValueError("label_len must be non-negative")
        if pred_len is not None and pred_len <= 0:
            raise ValueError("pred_len must be positive")
        self.label_len = label_len
        self.pred_len = pred_len

    @staticmethod
    def _decoder_input(target: torch.Tensor, label_len: int) -> torch.Tensor:
        if not 0 <= label_len <= target.shape[1]:
            raise ValueError(f"label_len must be in [0, {target.shape[1]}]")
        known = target[:, :label_len, :]
        future = torch.zeros_like(target[:, label_len:, :])
        return torch.cat([known, future], dim=1)

    def adapt(self, model: Any, raw_batch: Any) -> AdaptedBatch:
        if isinstance(raw_batch, Mapping):
            if "x_enc" not in raw_batch:
                raise ValueError("forecasting mapping batch requires 'x_enc'")
            x_enc = raw_batch["x_enc"]
            decoder_target = raw_batch.get("x_dec_target")
            if decoder_target is not None:
                label_len = self.label_len
                if label_len is None:
                    label_len = int(getattr(model, "label_len", 0))
                x_dec = self._decoder_input(decoder_target, label_len)
                pred_len = self.pred_len
                if pred_len is None:
                    pred_len = int(getattr(model, "pred_len", decoder_target.shape[1] - label_len))
                if pred_len <= 0 or pred_len > decoder_target.shape[1]:
                    raise ValueError(f"pred_len must be in [1, {decoder_target.shape[1]}], got {pred_len}")
                targets = decoder_target[:, -pred_len:, :]
            else:
                x_dec = raw_batch.get("x_dec")
                targets = raw_batch.get("targets")
            model_batch = model.make_forecast_batch(
                x_enc,
                x_mark_enc=raw_batch.get("x_mark_enc"),
                x_dec=x_dec,
                x_mark_dec=raw_batch.get("x_mark_dec"),
                x_target_enc=raw_batch.get("x_target_enc"),
                x_exog_enc=raw_batch.get("x_exog_enc"),
                x_target_dec=raw_batch.get("x_target_dec"),
                x_exog_dec=raw_batch.get("x_exog_dec"),
            )
            return AdaptedBatch(model_batch=model_batch, targets=targets, raw_batch=raw_batch)

        if isinstance(raw_batch, (tuple, list)):
            if len(raw_batch) == 4:
                x_enc, decoder_target, x_mark_enc, x_mark_dec = raw_batch
                return self.adapt(
                    model,
                    {
                        "x_enc": x_enc,
                        "x_dec_target": decoder_target,
                        "x_mark_enc": x_mark_enc,
                        "x_mark_dec": x_mark_dec,
                    },
                )
            if len(raw_batch) != 2:
                raise ValueError("forecasting sequence batch must contain either 2 or 4 values")
            x_enc, targets = raw_batch
            model_batch = model.make_forecast_batch(x_enc)
            return AdaptedBatch(model_batch=model_batch, targets=targets, raw_batch=raw_batch)

        if is_dataclass(raw_batch):
            return AdaptedBatch(model_batch=raw_batch, targets=_target_from_model_batch(raw_batch), raw_batch=raw_batch)
        raise TypeError(f"unsupported forecasting batch type: {type(raw_batch).__name__}")


class DefaultBatchAdapter:
    """Conservative model-aware adapter for common supervised torch_like tasks.

    Ambiguous domains (graph, recommendation, multimodal, RL, generative) are
    deliberately left unchanged unless the loader already returns their domain
    dataclass. Callers can provide an explicit adapter for those tasks.
    """

    def adapt(self, model: Any, raw_batch: Any) -> AdaptedBatch:
        if isinstance(raw_batch, AdaptedBatch):
            return raw_batch
        spec = get_model_spec(model)
        if is_dataclass(raw_batch):
            return AdaptedBatch(
                model_batch=raw_batch,
                targets=_target_from_model_batch(raw_batch, spec.task.target_field),
                raw_batch=raw_batch,
            )
        if spec.task.kind == TaskKind.TIME_SERIES_FORECASTING:
            return ForecastingBatchAdapter().adapt(model, raw_batch)

        values = _supervised_pair(raw_batch)
        if values is None:
            return IdentityBatchAdapter().adapt(model, raw_batch)
        x, targets = values
        kind = spec.task.kind
        if kind == TaskKind.TABULAR_PREDICTION and hasattr(model, "make_tabular_batch"):
            batch = model.make_tabular_batch(x, y=targets)
        elif kind == TaskKind.RUL_REGRESSION and hasattr(model, "make_rul_batch"):
            batch = model.make_rul_batch(x, targets=targets)
        elif kind == TaskKind.TIME_SERIES_CLASSIFICATION and hasattr(model, "make_diagnosis_batch"):
            batch = model.make_diagnosis_batch(x, labels=targets)
        elif kind == TaskKind.VISION and hasattr(model, "make_vision_batch"):
            batch = model.make_vision_batch(x, y=targets)
        elif kind == TaskKind.AUDIO and hasattr(model, "make_audio_batch"):
            batch = model.make_audio_batch(x, y=targets)
        elif kind == TaskKind.CAUSAL_LANGUAGE_MODELING and hasattr(model, "make_sequence_batch"):
            batch = model.make_sequence_batch(x, labels=targets)
        elif kind == TaskKind.PROBABILISTIC_FORECASTING and hasattr(model, "make_probabilistic_forecast_batch"):
            batch = model.make_probabilistic_forecast_batch(x, y_true=targets)
        else:
            return IdentityBatchAdapter().adapt(model, raw_batch)
        return AdaptedBatch(model_batch=batch, targets=targets, raw_batch=raw_batch)


def _supervised_pair(raw_batch: Any) -> tuple[Any, Any] | None:
    if isinstance(raw_batch, Sequence) and not isinstance(raw_batch, (str, bytes)) and len(raw_batch) == 2:
        return raw_batch[0], raw_batch[1]
    return None


def _target_from_model_batch(batch: Any, target_field: str | None = None) -> Any:
    fields = [target_field] if target_field else []
    fields.extend(["y", "labels", "targets", "y_true"])
    for field in fields:
        if field and hasattr(batch, field):
            value = getattr(batch, field)
            if value is not None:
                return value
    return None


def resolve_batch_adapter(
    model: Any,
    adapter: BatchAdapter | Callable[[Any], Any] | None,
    *,
    auto: bool,
) -> BatchAdapter:
    if adapter is None:
        return DefaultBatchAdapter() if auto else IdentityBatchAdapter()
    if isinstance(adapter, BatchAdapter):
        return adapter
    if callable(adapter):
        return CallableBatchAdapter(adapter)
    raise TypeError("batch_adapter must implement adapt(model, raw_batch) or be callable")


__all__ = [
    "AdaptedBatch",
    "BatchAdapter",
    "CallableBatchAdapter",
    "DefaultBatchAdapter",
    "ForecastingBatchAdapter",
    "IdentityBatchAdapter",
    "resolve_batch_adapter",
]
