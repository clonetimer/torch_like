"""Contracts for diagnostic telemetry, algorithm capabilities and predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
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
        if np.iscomplexobj(self.values):
            raise ValueError("complex-valued signals need an explicit real-valued representation")
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
            if np.iscomplexobj(values):
                raise ValueError("complex-valued estimates require explicit conversion")
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
        if task == DiagnosticTask.FAULT_ISOLATION and kind != DiagnosticOutputKind.FAULT_SOURCE_SCORE:
            raise ValueError("fault isolation requires fault_source_score")
        if task == DiagnosticTask.HEALTH_ASSESSMENT and kind != DiagnosticOutputKind.HEALTH_INDEX:
            raise ValueError("health assessment requires health_index")
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
class DiagnosticParameterSpec:
    """JSON-compatible constructor parameter descriptor, not a UI schema."""

    name: str
    kind: str
    default: Any = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    choices: tuple[Any, ...] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be nonempty")
        if self.kind not in {"integer", "number", "boolean", "string"}:
            raise ValueError("parameter kind must be integer, number, boolean or string")
        if not isinstance(self.required, bool) or (self.required and self.default is not None):
            raise ValueError("required parameters cannot declare a default")
        if self.kind not in {"integer", "number"} and (self.minimum is not None or self.maximum is not None):
            raise ValueError("minimum/maximum require a numeric parameter")
        if not isinstance(self.exclusive_minimum, bool) or not isinstance(self.exclusive_maximum, bool):
            raise TypeError("exclusive bounds must be booleans")
        if (self.exclusive_minimum and self.minimum is None) or (self.exclusive_maximum and self.maximum is None):
            raise ValueError("exclusive bounds require minimum/maximum")
        for bound in (self.minimum, self.maximum):
            if bound is not None and (
                isinstance(bound, bool) or not isinstance(bound, Real) or not math.isfinite(bound)
            ):
                raise ValueError("parameter bounds must be finite numbers")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("parameter minimum exceeds maximum")
        if self.choices is not None:
            choices = tuple(self.choices)
            if not choices or any(not self._accepts(value, check_range=False) for value in choices):
                raise ValueError("parameter choices must be nonempty, finite and match kind")
            if len(set(choices)) != len(choices):
                raise ValueError("parameter choices must be unique")
            if any(not self._accepts(value, check_choices=False) for value in choices):
                raise ValueError("parameter choice violates bounds")
            object.__setattr__(self, "choices", choices)
        if self.default is not None:
            self.validate(self.default)

    def _accepts(self, value: Any, *, check_range: bool = True, check_choices: bool = True) -> bool:
        if self.kind == "boolean":
            valid = isinstance(value, bool)
        elif self.kind == "string":
            valid = isinstance(value, str)
        elif self.kind == "integer":
            valid = isinstance(value, Integral) and not isinstance(value, bool)
        else:
            valid = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
        if not valid:
            return False
        if self.kind in {"integer", "number"} and check_range:
            if self.minimum is not None and value < self.minimum:
                return False
            if self.maximum is not None and value > self.maximum:
                return False
            if self.minimum is not None and self.exclusive_minimum and value == self.minimum:
                return False
            if self.maximum is not None and self.exclusive_maximum and value == self.maximum:
                return False
        return not (check_choices and self.choices is not None and value not in self.choices)

    def validate(self, value: Any) -> Any:
        if not self._accepts(value):
            raise ValueError(f"invalid value for diagnostic parameter {self.name!r}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "exclusive_minimum": self.exclusive_minimum,
            "exclusive_maximum": self.exclusive_maximum,
            "choices": None if self.choices is None else list(self.choices),
            "description": self.description,
        }


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
    parameters: tuple[DiagnosticParameterSpec, ...] = ()

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
        task = DiagnosticTask(self.task)
        output = DiagnosticOutputKind(self.output_kind)
        if task == DiagnosticTask.ANOMALY_DETECTION and output != DiagnosticOutputKind.ANOMALY_SCORE:
            raise ValueError("anomaly detection spec requires anomaly_score")
        if task == DiagnosticTask.RUL_REGRESSION and output != DiagnosticOutputKind.RUL_VALUE:
            raise ValueError("RUL spec requires rul_value")
        if task == DiagnosticTask.FAULT_CLASSIFICATION and output not in {
            DiagnosticOutputKind.CLASS_PROBABILITY,
            DiagnosticOutputKind.FAULT_LABEL,
        }:
            raise ValueError("fault classification spec requires probabilities or fault labels")
        if task == DiagnosticTask.FAULT_ISOLATION and output != DiagnosticOutputKind.FAULT_SOURCE_SCORE:
            raise ValueError("fault isolation spec requires fault_source_score")
        if task == DiagnosticTask.HEALTH_ASSESSMENT and output != DiagnosticOutputKind.HEALTH_INDEX:
            raise ValueError("health assessment spec requires health_index")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "output_kind", output)
        object.__setattr__(self, "operations", operations)
        parameters = tuple(self.parameters)
        if any(not isinstance(item, DiagnosticParameterSpec) for item in parameters):
            raise TypeError("parameters must be DiagnosticParameterSpec instances")
        if len({item.name for item in parameters}) != len(parameters):
            raise ValueError("duplicate diagnostic parameter names")
        object.__setattr__(self, "parameters", parameters)

    def validate_parameters(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Validate explicit, opt-in constructor metadata.

        Existing unannotated factories continue receiving arbitrary kwargs.
        No implicit casting of strings/bools into numeric hyperparameters.
        """
        supplied = dict(values)
        if not self.parameters:
            return supplied
        declared = {parameter.name: parameter for parameter in self.parameters}
        unknown = set(supplied) - set(declared)
        if unknown:
            raise ValueError(f"unknown diagnostic parameters: {sorted(unknown)}")
        resolved: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.name in supplied:
                resolved[parameter.name] = parameter.validate(supplied[parameter.name])
            elif parameter.required:
                raise ValueError(f"missing required diagnostic parameter {parameter.name!r}")
            elif parameter.default is not None:
                resolved[parameter.name] = parameter.default
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "task": self.task.value,
            "output_kind": self.output_kind.value,
            "operations": list(self.operations),
            "input_kind": self.input_kind,
            "supports_online": self.supports_online,
            "description": self.description,
            "parameters": [item.to_dict() for item in self.parameters],
        }
