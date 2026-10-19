"""One-shot, recording-disjoint supervised fault classification and reporting.

Training labels may only be passed to model.fit; sealed test labels are read
after the classifier has produced probabilities and are used only for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from torch_like.diagnostics.classification import FaultLabelSchema
from torch_like.diagnostics.types import DiagnosticOutputKind, DiagnosticResult, DiagnosticTask, SignalBatch


@dataclass(frozen=True)
class FaultClassificationPartition:
    """Each window carries stable original recording provenance and a class."""

    signals: SignalBatch
    recording_ids: tuple[str, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.signals, SignalBatch):
            raise TypeError("signals must be SignalBatch")
        n = self.signals.shape[0]
        if self.signals.sample_ids is None:
            raise ValueError("sample_ids are required for classification partitions")
        ids = tuple(self.recording_ids)
        labels = tuple(self.labels)
        if len(ids) != n or any(not isinstance(x, str) or not x.strip() for x in ids):
            raise ValueError("recording_ids must contain one original run name per window")
        if len(labels) != n or any(not isinstance(x, str) or not x.strip() for x in labels):
            raise ValueError("labels must contain one class name per window")
        object.__setattr__(self, "recording_ids", ids)
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class FaultClassificationSplit:
    """Explicit, recording- and sample-disjoint train / dev / sealed-test."""

    schema: FaultLabelSchema
    train: FaultClassificationPartition
    test: FaultClassificationPartition
    validation: FaultClassificationPartition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema, FaultLabelSchema):
            raise TypeError("schema must be FaultLabelSchema")
        parts = [("train", self.train), ("test", self.test)]
        if self.validation is not None:
            parts.append(("validation", self.validation))
        for name, part in parts:
            if not isinstance(part, FaultClassificationPartition):
                raise TypeError(f"{name} must be FaultClassificationPartition")
            self.schema.encode(part.labels)
        if set(self.train.labels) != set(self.schema.labels):
            raise ValueError("training partition must represent all declared fault classes")
        for i, (left_name, left) in enumerate(parts):
            for right_name, right in parts[i + 1 :]:
                if set(left.recording_ids) & set(right.recording_ids):
                    raise ValueError(f"{left_name} and {right_name} overlap in original recording_ids")
                if set(left.signals.sample_ids) & set(right.signals.sample_ids):
                    raise ValueError(f"{left_name} and {right_name} overlap in sample_ids")


@dataclass(frozen=True)
class FaultClassificationEvaluation:
    """Confusion matrix: true-class rows, predicted-class columns."""

    algorithm_name: str
    class_labels: tuple[str, ...]
    sample_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    predictions: tuple[str, ...]
    probabilities: np.ndarray
    confusion: np.ndarray

    def __post_init__(self) -> None:
        labels = tuple(self.class_labels)
        probs = np.asarray(self.probabilities, dtype=np.float64).copy()
        counts = np.asarray(self.confusion, dtype=np.int64).copy()
        n = len(self.sample_ids)
        if len(labels) < 2 or len(set(labels)) != len(labels) or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ):
            raise ValueError("class_labels must form a nonempty, unique multi-class vocabulary")
        if probs.shape != (n, len(labels)) or not np.isfinite(probs).all():
            raise ValueError("probabilities must match samples and class count")
        if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(
            probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5
        ):
            raise ValueError("probabilities must be normalized to one for each test window")
        if counts.shape != (len(labels), len(labels)) or np.any(counts < 0) or counts.sum() != n:
            raise ValueError("confusion must contain one nonnegative count per test sample")
        if (
            len(self.recording_ids) != n
            or len(self.predictions) != n
            or any(label not in labels for label in self.predictions)
        ):
            raise ValueError("prediction labels and recording IDs must align with samples")
        if len(set(self.sample_ids)) != n or any(
            not isinstance(item, str) or not item.strip() for item in self.sample_ids
        ):
            raise ValueError("sample_ids must be nonempty and unique")
        if any(not isinstance(item, str) or not item.strip() for item in self.recording_ids):
            raise ValueError("recording_ids must be nonempty source identifiers")
        predicted_columns = np.argmax(probs, axis=1)
        expected_names = tuple(labels[index] for index in predicted_columns)
        if tuple(self.predictions) != expected_names:
            raise ValueError("predictions must match the corresponding class-probability argmax")
        if not np.array_equal(counts.sum(axis=0), np.bincount(predicted_columns, minlength=len(labels))):
            raise ValueError("confusion predicted-class totals must match the reported predictions")
        probs.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "probabilities", probs)
        object.__setattr__(self, "confusion", counts)
        object.__setattr__(self, "class_labels", labels)
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "recording_ids", tuple(self.recording_ids))
        object.__setattr__(self, "predictions", tuple(self.predictions))

    @property
    def accuracy(self) -> float:
        return float(np.trace(self.confusion) / self.confusion.sum())

    @property
    def macro_f1(self) -> float:
        diagonal = np.diag(self.confusion)
        denominator = self.confusion.sum(axis=0) + self.confusion.sum(axis=1)
        scores = np.divide(2 * diagonal, denominator, out=np.zeros(len(diagonal), dtype=float), where=denominator > 0)
        return float(scores.mean())

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible evidence, omitting individual held-out truth labels."""
        return {
            "algorithm": self.algorithm_name,
            "class_labels": list(self.class_labels),
            "num_windows": len(self.sample_ids),
            "sample_ids": list(self.sample_ids),
            "recording_ids": list(self.recording_ids),
            "predictions": list(self.predictions),
            "probabilities": self.probabilities.tolist(),
            "confusion": self.confusion.tolist(),
            "support": self.confusion.sum(axis=1).tolist(),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "metric_level": "window",
        }


class FaultClassificationWorkflow:
    """Single-use sealed test: one supervised fit, one prediction batch."""

    def __init__(self, classifier: Any) -> None:
        if not callable(getattr(classifier, "fit", None)) or not callable(getattr(classifier, "predict_proba", None)):
            raise TypeError("classifier must implement fit and predict_proba")
        self.classifier = classifier
        self._consumed = False

    def run(self, split: FaultClassificationSplit) -> FaultClassificationEvaluation:
        if self._consumed:
            raise RuntimeError("sealed fault-classification evaluation already consumed")
        if not isinstance(split, FaultClassificationSplit):
            raise TypeError("run expects FaultClassificationSplit")
        self._consumed = True
        self.classifier.fit(split.train.signals, split.train.labels, split.schema)
        result = self.classifier.predict_proba(split.test.signals)
        if not isinstance(result, DiagnosticResult):
            raise TypeError("classifier must return DiagnosticResult")
        if (
            result.task != DiagnosticTask.FAULT_CLASSIFICATION
            or result.output_kind != DiagnosticOutputKind.CLASS_PROBABILITY
        ):
            raise ValueError("classifier must return fault_classification/class_probability")
        if result.label_names != split.schema.labels:
            raise ValueError("prediction class ordering must match fixed schema")
        if result.values.shape != (split.test.signals.shape[0], len(split.schema.labels)):
            raise ValueError("classifier must emit one class-probability row per window")
        if not np.allclose(result.values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError("classifier probabilities must sum to one")

        expected = split.schema.encode(split.test.labels)
        predicted = np.argmax(result.values, axis=1)
        confusion = np.zeros((len(split.schema.labels), len(split.schema.labels)), dtype=np.int64)
        np.add.at(confusion, (expected, predicted), 1)
        names = tuple(split.schema.labels[index] for index in predicted)
        return FaultClassificationEvaluation(
            algorithm_name=result.algorithm_name,
            class_labels=split.schema.labels,
            sample_ids=split.test.signals.sample_ids,
            recording_ids=split.test.recording_ids,
            predictions=names,
            probabilities=result.values,
            confusion=confusion,
        )
