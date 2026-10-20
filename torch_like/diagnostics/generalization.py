"""Closed-world fault classification under explicit device/condition holdouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from torch_like.diagnostics.classification import FaultLabelSchema
from torch_like.diagnostics.classification_workflow import (
    FaultClassificationEvaluation,
    FaultClassificationPartition,
    FaultClassificationSplit,
    FaultClassificationWorkflow,
)


@dataclass(frozen=True)
class FaultDomainPartition:
    """Each original recording belongs to one device and one operating condition."""

    classification: FaultClassificationPartition
    device_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.classification, FaultClassificationPartition):
            raise TypeError("classification must be FaultClassificationPartition")
        n = self.classification.signals.shape[0]
        for field in ("device_ids", "condition_ids"):
            ids = tuple(getattr(self, field))
            if len(ids) != n or any(not isinstance(x, str) or not x.strip() for x in ids):
                raise ValueError(f"{field} must identify every window with a nonempty string")
            object.__setattr__(self, field, ids)
        context_by_run: dict[str, tuple[str, str]] = {}
        for run, device, condition in zip(self.classification.recording_ids, self.device_ids, self.condition_ids, strict=True):
            current = (device, condition)
            if run in context_by_run and current != context_by_run[run]:
                raise ValueError("one original recording_id cannot span multiple physical contexts")
            context_by_run[run] = current

    @property
    def context_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.device_ids, self.condition_ids, strict=True))


@dataclass(frozen=True)
class FaultGeneralizationSplit:
    """Hold out device, condition, both individual IDs, or a joint domain pair."""

    schema: FaultLabelSchema
    train: FaultDomainPartition
    test: FaultDomainPartition
    validation: FaultDomainPartition | None = None
    holdout: str = "device"
    min_domain_windows: int = 5

    def __post_init__(self) -> None:
        if self.holdout not in {"device", "condition", "both", "pair"}:
            raise ValueError("holdout must be device, condition, both or pair")
        if (
            isinstance(self.min_domain_windows, bool)
            or not isinstance(self.min_domain_windows, int)
            or self.min_domain_windows < 1
        ):
            raise ValueError("min_domain_windows must be a positive integer")
        groups = [("train", self.train), ("test", self.test)]
        if self.validation is not None:
            groups.append(("validation", self.validation))
        for name, part in groups:
            if not isinstance(part, FaultDomainPartition):
                raise TypeError(f"{name} must be FaultDomainPartition")
        # Retain the R4a closed label vocabulary, train-class coverage and
        # sample/recording disjointness even when domain IDs are different.
        _ = self.as_classification_split()
        axes = ("device", "condition") if self.holdout == "both" else (self.holdout,)
        for index, (name, first) in enumerate(groups):
            for second_name, second in groups[index + 1 :]:
                for axis in axes:
                    overlap = self._identities(first, axis) & self._identities(second, axis)
                    if overlap:
                        raise ValueError(
                            f"{self.holdout} holdout leaks {axis} between {name} and {second_name}: {sorted(overlap)!r}"
                        )

    @staticmethod
    def _identities(part: FaultDomainPartition, axis: str) -> set[str] | set[tuple[str, str]]:
        if axis == "device":
            return set(part.device_ids)
        if axis == "condition":
            return set(part.condition_ids)
        return set(part.context_pairs)

    def as_classification_split(self) -> FaultClassificationSplit:
        return FaultClassificationSplit(
            self.schema,
            self.train.classification,
            self.test.classification,
            self.validation.classification if self.validation is not None else None,
        )


@dataclass(frozen=True)
class FaultDomainMetrics:
    """Auditable slice with a fixed vocabulary and explicit absent true classes."""

    axis: str
    identity: tuple[str, ...]
    window_count: int
    recording_count: int
    confusion: np.ndarray
    class_labels: tuple[str, ...]
    min_domain_windows: int

    def __post_init__(self) -> None:
        counts = np.asarray(self.confusion, dtype=np.int64).copy()
        size = len(self.class_labels)
        if counts.shape != (size, size) or np.any(counts < 0) or int(counts.sum()) != self.window_count:
            raise ValueError("domain confusion must count every sealed sample exactly once")
        if self.window_count < 1 or self.recording_count < 1:
            raise ValueError("domain slice must have windows and source recordings")
        counts.setflags(write=False)
        object.__setattr__(self, "confusion", counts)

    @property
    def support(self) -> np.ndarray:
        return self.confusion.sum(axis=1)

    @property
    def accuracy(self) -> float:
        return float(np.trace(self.confusion) / self.window_count)

    def _class_f1(self) -> np.ndarray:
        true_positive = np.diag(self.confusion)
        denominator = self.confusion.sum(axis=0) + self.support
        return np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros(len(true_positive), dtype=np.float64),
            where=denominator > 0,
        )

    @property
    def macro_f1_all_classes(self) -> float:
        return float(self._class_f1().mean())

    @property
    def macro_f1_present_classes(self) -> float:
        """Not comparable across slices with different observed class sets."""
        return float(self._class_f1()[self.support > 0].mean())

    def to_dict(self) -> dict[str, Any]:
        support = self.support
        return {
            "axis": self.axis,
            "identity": list(self.identity),
            "num_windows": self.window_count,
            "num_recordings": self.recording_count,
            "confusion": self.confusion.tolist(),
            "support": support.tolist(),
            "absent_true_classes": [name for name, count in zip(self.class_labels, support, strict=True) if count == 0],
            "accuracy": self.accuracy,
            "macro_f1_all_classes": self.macro_f1_all_classes,
            "macro_f1_present_classes": self.macro_f1_present_classes,
            "sufficient_window_support": self.window_count >= self.min_domain_windows,
        }


@dataclass(frozen=True)
class FaultGeneralizationEvaluation:
    """One sealed report, sliced by actual physical-domain provenance."""

    overall: FaultClassificationEvaluation
    holdout: str
    min_domain_windows: int
    train_devices: tuple[str, ...]
    train_conditions: tuple[str, ...]
    test_devices: tuple[str, ...]
    test_conditions: tuple[str, ...]
    slices: tuple[FaultDomainMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "recording_disjoint_closed_world_domain_holdout",
            "holdout": self.holdout,
            "min_domain_windows": self.min_domain_windows,
            "train_devices": list(self.train_devices),
            "train_conditions": list(self.train_conditions),
            "test_devices": list(self.test_devices),
            "test_conditions": list(self.test_conditions),
            "overall": self.overall.to_dict(),
            "domain_slices": [part.to_dict() for part in self.slices],
            "underpowered_domain_count": sum(
                part.window_count < self.min_domain_windows for part in self.slices
            ),
        }


class FaultGeneralizationWorkflow:
    """Fit on train labels; score sealed test once; slice only after scoring."""

    def __init__(self, classifier: Any) -> None:
        if not callable(getattr(classifier, "fit", None)) or not callable(
            getattr(classifier, "predict_proba", None)
        ):
            raise TypeError("classifier must implement fit and predict_proba")
        self.classifier = classifier
        self._consumed = False

    def run(self, split: FaultGeneralizationSplit) -> FaultGeneralizationEvaluation:
        if self._consumed:
            raise RuntimeError("sealed domain generalization evaluation already consumed")
        if not isinstance(split, FaultGeneralizationSplit):
            raise TypeError("run expects FaultGeneralizationSplit")
        self._consumed = True
        # Domain identity and sealed labels never enter estimator fitting or inference.
        overall = FaultClassificationWorkflow(self.classifier).run(split.as_classification_split())
        test = split.test
        labels = split.schema.labels
        true_index = split.schema.encode(test.classification.labels)
        pred_index = np.array([labels.index(name) for name in overall.predictions], dtype=np.int64)
        axes = (
            ("device", tuple((name,) for name in test.device_ids)),
            ("condition", tuple((name,) for name in test.condition_ids)),
            ("pair", test.context_pairs),
            ("recording", tuple((name,) for name in test.classification.recording_ids)),
        )
        slices = []
        for axis, identities in axes:
            for identity in sorted(set(identities)):
                rows = [i for i, key in enumerate(identities) if key == identity]
                confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
                np.add.at(confusion, (true_index[rows], pred_index[rows]), 1)
                slices.append(
                    FaultDomainMetrics(
                        axis=axis,
                        identity=identity,
                        window_count=len(rows),
                        recording_count=len({test.classification.recording_ids[i] for i in rows}),
                        confusion=confusion,
                        class_labels=labels,
                        min_domain_windows=split.min_domain_windows,
                    )
                )
        return FaultGeneralizationEvaluation(
            overall=overall,
            holdout=split.holdout,
            min_domain_windows=split.min_domain_windows,
            train_devices=tuple(sorted(set(split.train.device_ids))),
            train_conditions=tuple(sorted(set(split.train.condition_ids))),
            test_devices=tuple(sorted(set(test.device_ids))),
            test_conditions=tuple(sorted(set(test.condition_ids))),
            slices=tuple(slices),
        )
