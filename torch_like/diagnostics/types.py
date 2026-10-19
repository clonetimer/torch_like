"""Contracts for diagnostic telemetry, algorithm capabilities and predictions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class DiagnosticTask(str, Enum):
    ANOMALY_DETECTION = "anomaly_detection"
    FAULT_CLASSIFICATION = "fault_classification"
    FAULT_ISOLATION = "fault_isolation"
    HEALTH_ASSESSMENT = "health_assessment"
    RUL_REGRESSION = "rul_regression"


class DiagnosticOutputKind(str, Enum):
    ANOMALY_SCORE = "anomaly_score"
    CLASS_PROBABILITY = "class_probability"
    FAULT_LABEL = "fault_label"
    FAULT_SOURCE_SCORE = "fault_source_score"
    HEALTH_INDEX = "health_index"
    RUL_VALUE = "rul_value"


@dataclass(frozen=True)
class SignalBatch:
    """Signals have shape (samples, time, channels) and explicit sample rate in Hz."""

    values: np.ndarray
    sample_rate_hz: float
    channel_names: tuple[str, ...] | None = None
    units: tuple[str, ...] | None = None
    sample_ids: tuple[str, ...] | None = None
    timestamps_s: np.ndarray | None = None
    allow_missing: bool = False

    def __post_init__(self) -> None:
        x = np.asarray(self.values, dtype=np.float64)
        if x.ndim != 3 or min(x.shape) <= 0:
            raise ValueError("values must have nonempty shape (samples, time, channels)")
        if np.isinf(x).any() or (not self.allow_missing and not np.isfinite(x).all()):
            raise ValueError("nonfinite data violates the missing-value policy")
        rate = float(self.sample_rate_hz)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        n, length, channels = x.shape
        names = (
            tuple(f"channel_{i}" for i in range(channels)) if self.channel_names is None else tuple(self.channel_names)
        )
        if (
            len(names) != channels
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name.strip() for name in names)
        ):
            raise ValueError("channel_names must be unique nonempty strings, one per channel")
        if self.units is not None and (
            len(self.units) != channels or any(not isinstance(unit, str) or not unit.strip() for unit in self.units)
        ):
            raise ValueError("units must have one nonempty unit per channel")
        if self.sample_ids is not None and (
            len(self.sample_ids) != n
            or len(set(self.sample_ids)) != n
            or any(not isinstance(item, str) or not item.strip() for item in self.sample_ids)
        ):
            raise ValueError("sample_ids must be unique nonempty strings, one per sample")
        if self.timestamps_s is not None:
            times = np.asarray(self.timestamps_s, dtype=np.float64)
            if times.ndim == 1 and times.shape == (length,):
                times = np.broadcast_to(times, (n, length))
            if times.shape != (n, length):
                raise ValueError("timestamps_s must have shape (time,) or (samples, time)")
            if not np.isfinite(times).all() or np.any(np.diff(times, axis=1) <= 0):
                raise ValueError("timestamps_s must be finite and strictly increasing")
            times = np.array(times, copy=True)
            times.setflags(write=False)
            object.__setattr__(self, "timestamps_s", times)
        x = np.array(x, copy=True)
        x.setflags(write=False)
        object.__setattr__(self, "values", x)
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "channel_names", names)
        if self.units is not None:
            object.__setattr__(self, "units", tuple(self.units))
        if self.sample_ids is not None:
            object.__setattr__(self, "sample_ids", tuple(self.sample_ids))

    @classmethod
    def from_sequence(cls, values: Any, *, sample_rate_hz: float, **kwargs: Any) -> "SignalBatch":
        """Explicitly promote (time, channels) into a single-sample batch."""
        x = np.asarray(values)
        if x.ndim != 2:
            raise ValueError("from_sequence expects (time, channels)")
        return cls(x[np.newaxis, :, :], sample_rate_hz=sample_rate_hz, **kwargs)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape

    @property
    def duration_s(self) -> float:
        return self.values.shape[1] / self.sample_rate_hz


@dataclass(frozen=True)
class DiagnosticResult:
    """Prediction-only result; input ground truth must not be stored here."""

    task: DiagnosticTask
    output_kind: DiagnosticOutputKind
    values: np.ndarray
    algorithm_name: str
    threshold: float | None = None
    label_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        task = DiagnosticTask(self.task)
        kind = DiagnosticOutputKind(self.output_kind)
        if not isinstance(self.algorithm_name, str) or not self.algorithm_name.strip():
            raise ValueError("algorithm_name must be nonempty")
        values = np.asarray(self.values)
        if values.ndim not in {1, 2, 3} or min(values.shape) <= 0:
            raise ValueError("values must have nonempty sample-leading shape")
        if kind == DiagnosticOutputKind.FAULT_LABEL:
            if values.ndim != 1:
                raise ValueError("fault_label must be one-dimensional")
        else:
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError("scores and estimates must be numeric")
            values = np.asarray(values, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("scores and estimates must be finite")
        if self.threshold is not None and (
            kind not in {DiagnosticOutputKind.ANOMALY_SCORE, DiagnosticOutputKind.HEALTH_INDEX}
            or not np.isfinite(float(self.threshold))
        ):
            raise ValueError("threshold must be finite and only set for score/index outputs")
        if self.label_names is not None and (
            not self.label_names
            or len(set(self.label_names)) != len(self.label_names)
            or any(not isinstance(label, str) or not label.strip() for label in self.label_names)
        ):
            raise ValueError("label_names must be nonempty and unique")
        if kind == DiagnosticOutputKind.CLASS_PROBABILITY:
            if values.ndim != 2 or np.any(values < 0) or np.any(values > 1):
                raise ValueError("class_probability requires (samples, classes) in [0, 1]")
            if self.label_names is not None and len(self.label_names) != values.shape[1]:
                raise ValueError("label_names must match the classes axis")
        if kind == DiagnosticOutputKind.RUL_VALUE and (values.ndim != 1 or np.any(values < 0)):
            raise ValueError("rul_value must be a nonnegative estimate per sample")
        if task == DiagnosticTask.RUL_REGRESSION and kind != DiagnosticOutputKind.RUL_VALUE:
            raise ValueError("RUL task requires rul_value")
        if task == DiagnosticTask.ANOMALY_DETECTION and kind != DiagnosticOutputKind.ANOMALY_SCORE:
            raise ValueError("anomaly detection requires anomaly_score")
        if task == DiagnosticTask.FAULT_CLASSIFICATION and kind not in {
            DiagnosticOutputKind.CLASS_PROBABILITY,
            DiagnosticOutputKind.FAULT_LABEL,
        }:
            raise ValueError("fault classification requires probabilities or fault labels")
        values = np.array(values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "output_kind", kind)
        if self.label_names is not None:
            object.__setattr__(self, "label_names", tuple(self.label_names))

    @property
    def sample_count(self) -> int:
        return self.values.shape[0]


@dataclass(frozen=True)
class DiagnosticAlgorithmSpec:
    """Backend-independent metadata, with no change to native estimator types."""

    name: str
    backend: str
    task: DiagnosticTask
    output_kind: DiagnosticOutputKind
    operations: tuple[str, ...] = ("predict",)
    input_kind: str = "signal_batch"
    supports_online: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")
        if self.backend not in {"statistical", "sklearn", "torch_like", "external"}:
            raise ValueError("unknown backend")
        if self.input_kind not in {"signal_batch", "feature_matrix"}:
            raise ValueError("unknown input_kind")
        operations = tuple(self.operations)
        if (
            not operations
            or len(set(operations)) != len(operations)
            or not set(operations).issubset({"fit", "predict", "score", "partial_fit"})
        ):
            raise ValueError("operations must be unique supported operations")
        object.__setattr__(self, "task", DiagnosticTask(self.task))
        object.__setattr__(self, "output_kind", DiagnosticOutputKind(self.output_kind))
        object.__setattr__(self, "operations", operations)
