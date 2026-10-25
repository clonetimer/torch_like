"""Optional, explicit final-decision results for industrial diagnostic algorithms.

No fit, implicit threshold selection, score pooling, label remapping or evaluator
policy lives here. Decisions remain separate from the legacy DiagnosticResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from torch_like.diagnostics.types import DiagnosticOutputKind, DiagnosticResult, DiagnosticTask


class DecisionMode(str, Enum):
    SCORE_THRESHOLD = "score_threshold"
    NATIVE = "native"
    LABEL = "label"


@dataclass(frozen=True)
class DiagnosticDecision:
    """Algorithm-owned labels, preserving score resolution and decision authority.

    Anomaly labels are binary, with shape (N,), (N,L), or (N,L,C).
    Fault-classification labels are one-dimensional, allowing multiple classes.
    A native algorithm may additionally expose its DiagnosticResult, but must not
    present a separate generic threshold as an alternative decision authority.
    """

    task: DiagnosticTask
    decision_mode: DecisionMode
    labels: np.ndarray
    algorithm_name: str
    source_result: DiagnosticResult | None = None
    thresholds: np.ndarray | None = None
    label_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        task = DiagnosticTask(self.task)
        mode = DecisionMode(self.decision_mode)
        if task not in {DiagnosticTask.ANOMALY_DETECTION, DiagnosticTask.FAULT_CLASSIFICATION}:
            raise ValueError("decisions support anomaly_detection and fault_classification only")
        if not isinstance(self.algorithm_name, str) or not self.algorithm_name.strip():
            raise ValueError("algorithm_name must be nonempty")

        labels = np.asarray(self.labels)
        if labels.ndim not in {1, 2, 3} or not labels.size or any(dim < 1 for dim in labels.shape):
            raise ValueError("labels must be a nonempty sample-leading array")
        if task == DiagnosticTask.ANOMALY_DETECTION:
            if labels.dtype.kind not in {"b", "i", "u"} or np.any((labels != 0) & (labels != 1)):
                raise ValueError("anomaly labels must be binary (0 or 1)")
            labels = labels.astype(np.int64, copy=True)
        else:
            if labels.ndim != 1 or labels.dtype.kind not in {"i", "u", "U", "S"}:
                raise ValueError("fault classification labels require one-dimensional integer or string labels")
            labels = np.array(labels, copy=True)

        names = None if self.label_names is None else tuple(self.label_names)
        if names is not None:
            if task != DiagnosticTask.FAULT_CLASSIFICATION:
                raise ValueError("label_names only apply to fault classification")
            if not names or len(set(names)) != len(names) or any(
                not isinstance(name, str) or not name.strip() for name in names
            ):
                raise ValueError("label_names must contain unique nonempty strings")
            if labels.dtype.kind in {"i", "u"}:
                if np.any(labels < 0) or np.any(labels >= len(names)):
                    raise ValueError("classification label index outside label_names")
            elif any(str(label) not in names for label in labels):
                raise ValueError("classification labels not declared in label_names")

        source = self.source_result
        if source is not None:
            if not isinstance(source, DiagnosticResult) or source.task != task:
                raise ValueError("source_result task must match the decision")
            if source.sample_count != labels.shape[0]:
                raise ValueError("source_result and labels differ on sample axis")

        if mode == DecisionMode.SCORE_THRESHOLD:
            if (
                task != DiagnosticTask.ANOMALY_DETECTION
                or source is None
                or source.output_kind != DiagnosticOutputKind.ANOMALY_SCORE
                or source.algorithm_name != self.algorithm_name
                or self.thresholds is None
            ):
                raise ValueError("score_threshold requires a matching anomaly score and explicit thresholds")
            threshold = _expand_thresholds(self.thresholds, source.values.shape)
            if labels.shape != source.values.shape:
                raise ValueError("score_threshold labels must preserve source score shape")
            if not np.array_equal(labels, (source.values > threshold).astype(np.int64)):
                raise ValueError("score_threshold labels disagree with the threshold comparison")
        else:
            if self.thresholds is not None:
                raise ValueError("native/label decisions cannot declare generic thresholds")
            if mode == DecisionMode.LABEL and source is not None:
                raise ValueError("label mode cannot declare a competing score source")
            threshold = None

        labels.setflags(write=False)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "decision_mode", mode)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "thresholds", threshold)
        object.__setattr__(self, "label_names", names)

    @property
    def sample_count(self) -> int:
        return self.labels.shape[0]

    @classmethod
    def from_threshold(
        cls, result: DiagnosticResult, threshold: float | np.ndarray
    ) -> DiagnosticDecision:
        """Apply a caller-selected fixed/online threshold without fitting it.

        A scalar cutoff expands to the exact score shape. For a time-varying
        threshold, supply an array with precisely the same shape as the scores.
        """
        if not isinstance(result, DiagnosticResult) or result.task != DiagnosticTask.ANOMALY_DETECTION:
            raise ValueError("from_threshold requires anomaly DiagnosticResult")
        expanded = _expand_thresholds(threshold, result.values.shape)
        return cls(
            task=result.task,
            decision_mode=DecisionMode.SCORE_THRESHOLD,
            labels=(result.values > expanded).astype(np.int64),
            algorithm_name=result.algorithm_name,
            source_result=result,
            thresholds=expanded,
        )

    @classmethod
    def from_labels(
        cls,
        *,
        task: DiagnosticTask | str,
        labels: np.ndarray,
        algorithm_name: str,
        decision_mode: DecisionMode | str = DecisionMode.LABEL,
        source_result: DiagnosticResult | None = None,
        label_names: tuple[str, ...] | None = None,
    ) -> DiagnosticDecision:
        mode = DecisionMode(decision_mode)
        if mode == DecisionMode.SCORE_THRESHOLD:
            raise ValueError("use from_threshold for score-based decisions")
        return cls(
            task=task,
            decision_mode=mode,
            labels=labels,
            algorithm_name=algorithm_name,
            source_result=source_result,
            label_names=label_names,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "decision_mode": self.decision_mode.value,
            "algorithm_name": self.algorithm_name,
            "labels": self.labels.tolist(),
            "label_names": None if self.label_names is None else list(self.label_names),
            "thresholds": None if self.thresholds is None else self.thresholds.tolist(),
            "source_result": None if self.source_result is None else {
                "algorithm_name": self.source_result.algorithm_name,
                "output_kind": self.source_result.output_kind.value,
                "values": self.source_result.values.tolist(),
            },
        }


def _expand_thresholds(threshold: float | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(threshold)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("thresholds must be finite real numbers, not booleans or strings")
    if raw.ndim == 0:
        values = np.full(shape, float(raw), dtype=np.float64)
    elif raw.shape == shape:
        values = raw.astype(np.float64, copy=True)
    else:
        raise ValueError("thresholds must be scalar or exactly match score shape")
    if not np.isfinite(values).all():
        raise ValueError("thresholds must be finite")
    values.setflags(write=False)
    return values


__all__ = ["DecisionMode", "DiagnosticDecision"]
