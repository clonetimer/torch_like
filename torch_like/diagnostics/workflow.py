"""One-shot anomaly workflow with recording-disjoint development and evaluation.

This module deliberately handles one bounded task: window-level anomaly
detection. It is not a general diagnostic, fault-isolation, or RUL evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from torch_like.diagnostics.types import DiagnosticOutputKind, DiagnosticResult, DiagnosticTask, SignalBatch


@dataclass(frozen=True)
class AnomalyPartition:
    """A named set of windows from explicitly identified source recordings.

    Sample IDs are mandatory: there must be no untraceable windows. Group IDs
    must identify the *original recording/device run* and remain stable across
    partitions, even if windows were reindexed or reordered upstream.
    """

    signals: SignalBatch
    recording_ids: tuple[str, ...]
    labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.signals, SignalBatch):
            raise TypeError("signals must be SignalBatch")
        n = self.signals.shape[0]
        if self.signals.sample_ids is None:
            raise ValueError("each signal window must provide a unique sample_id for split auditing")
        ids = tuple(self.recording_ids)
        if len(ids) != n or any(not isinstance(x, str) or not x.strip() for x in ids):
            raise ValueError("recording_ids must contain a nonempty source id for every window")
        object.__setattr__(self, "recording_ids", ids)
        if self.labels is not None:
            values = np.asarray(self.labels)
            if values.shape != (n,) or values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                raise ValueError("labels must be finite binary sample-level values with shape (windows,)")
            if not np.logical_or(values == 0, values == 1).all():
                raise ValueError("labels must be binary 0/1 without unknown classes")
            labels = np.array(values, dtype=np.uint8, copy=True)
            labels.setflags(write=False)
            object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class AnomalySplit:
    """Recording- and sample-disjoint fit, calibration and sealed evaluation."""

    reference: AnomalyPartition
    test: AnomalyPartition
    validation: AnomalyPartition | None = None

    def __post_init__(self) -> None:
        partitions = [("reference", self.reference), ("test", self.test)]
        if self.validation is not None:
            partitions.append(("validation", self.validation))
        for name, part in partitions:
            if not isinstance(part, AnomalyPartition):
                raise TypeError(f"{name} must be AnomalyPartition")
        if self.reference.labels is not None and np.any(self.reference.labels != 0):
            raise ValueError("reference fitting requires healthy-only windows")
        for index, (name, left) in enumerate(partitions):
            for other_name, right in partitions[index + 1 :]:
                recordings = set(left.recording_ids) & set(right.recording_ids)
                if recordings:
                    raise ValueError(f"{name} and {other_name} share original recording_ids: {sorted(recordings)!r}")
                samples = set(left.signals.sample_ids) & set(right.signals.sample_ids)
                if samples:
                    raise ValueError(f"{name} and {other_name} share sample_ids: {sorted(samples)!r}")


@dataclass(frozen=True)
class AnomalyEvaluation:
    """Auditable test-only window predictions and confusion-matrix metrics."""

    algorithm_name: str
    threshold: float
    threshold_source: str
    sample_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    scores: np.ndarray
    predictions: np.ndarray
    tp: int
    tn: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float64).copy()
        predictions = np.asarray(self.predictions, dtype=bool).copy()
        if scores.shape != predictions.shape or scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("evaluation requires one finite anomaly score per window")
        if len(self.sample_ids) != scores.size or len(self.recording_ids) != scores.size:
            raise ValueError("prediction provenance length must match sample count")
        scores.setflags(write=False)
        predictions.setflags(write=False)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "predictions", predictions)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0

    @property
    def false_alarm_rate(self) -> float | None:
        """Undefined when the sealed test subset has no healthy windows."""
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly report (never includes ground-truth label arrays)."""
        return {
            "algorithm": self.algorithm_name,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "num_windows": int(self.scores.size),
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_alarm_rate": self.false_alarm_rate,
            "sample_ids": list(self.sample_ids),
            "recording_ids": list(self.recording_ids),
            "scores": self.scores.tolist(),
            "predictions": self.predictions.tolist(),
        }


class AnomalyDiagnosticWorkflow:
    """Fit -> optional labeled validation calibration -> evaluate exactly once.

    Modes:
      native: use the detector's own threshold, no validation labels required.
      validation_f1: learn a threshold only from labeled development validation
                     using per-window F1, never held-out test labels.
    """

    def __init__(
        self,
        detector: Any,
        *,
        threshold_policy: str = "native",
        threshold_candidates: int = 64,
    ) -> None:
        if threshold_policy not in {"native", "validation_f1"}:
            raise ValueError("threshold_policy must be native or validation_f1")
        if (
            isinstance(threshold_candidates, bool)
            or not isinstance(threshold_candidates, int)
            or not 2 <= threshold_candidates <= 1000
        ):
            raise ValueError("threshold_candidates must be an integer in [2, 1000]")
        if not callable(getattr(detector, "fit", None)) or not callable(getattr(detector, "score", None)):
            raise TypeError("detector must implement fit and score")
        self.detector = detector
        self.threshold_policy = threshold_policy
        self.threshold_candidates = threshold_candidates
        self._used = False

    def _score_windows(self, signals: SignalBatch) -> tuple[np.ndarray, DiagnosticResult]:
        result = self.detector.score(signals)
        if not isinstance(result, DiagnosticResult):
            raise TypeError("detector.score must return DiagnosticResult")
        if result.task != DiagnosticTask.ANOMALY_DETECTION or result.output_kind != DiagnosticOutputKind.ANOMALY_SCORE:
            raise ValueError("workflow requires anomaly_detection/anomaly_score output")
        if result.sample_count != signals.shape[0]:
            raise ValueError("detector scores must preserve the leading window dimension")
        x = result.values
        if x.ndim > 1:
            x = np.max(x, axis=tuple(range(1, x.ndim)))
        if x.shape != (signals.shape[0],) or not np.isfinite(x).all():
            raise ValueError("reduced anomaly scores must be finite and window-level")
        return np.asarray(x, dtype=np.float64), result

    @staticmethod
    def _confusion(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
        truth = labels.astype(bool)
        tp = int(np.count_nonzero(truth & predictions))
        tn = int(np.count_nonzero(~truth & ~predictions))
        fp = int(np.count_nonzero(~truth & predictions))
        fn = int(np.count_nonzero(truth & ~predictions))
        return tp, tn, fp, fn

    def _validation_threshold(self, validation: AnomalyPartition) -> float:
        if validation.labels is None:
            raise ValueError("validation_f1 requires validation labels")
        if np.unique(validation.labels).size != 2:
            raise ValueError("validation_f1 requires both healthy and anomalous validation windows")
        scores, _ = self._score_windows(validation.signals)
        unique = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, self.threshold_candidates)))
        low = float(np.nextafter(float(np.min(scores)), -np.inf))
        if not np.isfinite(low):
            raise ValueError("validation anomaly scores cannot admit a finite lower decision threshold")
        candidates = np.unique(np.concatenate(([low], unique)))
        threshold, max_f1 = None, -1.0
        for candidate in candidates:
            predicted = scores > candidate
            tp, _, fp, fn = self._confusion(validation.labels, predicted)
            denominator = 2 * tp + fp + fn
            value = 2 * tp / denominator if denominator else 0.0
            # Conservative tie break: prefer fewer alerts (higher threshold).
            if value > max_f1 or (value == max_f1 and (threshold is None or candidate > threshold)):
                threshold, max_f1 = float(candidate), value
        assert threshold is not None
        return threshold

    def run(self, split: AnomalySplit) -> AnomalyEvaluation:
        """Evaluate sealed test once; no train/test overlap by recording or sample ID."""
        if self._used:
            raise RuntimeError("this workflow has already consumed a sealed test evaluation")
        if not isinstance(split, AnomalySplit):
            raise TypeError("run expects an AnomalySplit")
        if split.test.labels is None:
            raise ValueError("sealed test labels are required only for final evaluation")
        if self.threshold_policy == "validation_f1" and split.validation is None:
            raise ValueError("validation_f1 requires a distinct validation partition")

        # Mark the evaluation as consumed before any fallible model operation.
        # A failed run cannot be retried against the same test partition.
        self._used = True
        self.detector.fit(split.reference.signals)

        if self.threshold_policy == "validation_f1":
            assert split.validation is not None
            threshold = self._validation_threshold(split.validation)
        else:
            threshold = None

        test_scores, native_result = self._score_windows(split.test.signals)
        if self.threshold_policy == "native":
            if native_result.threshold is None or not np.isfinite(native_result.threshold):
                raise ValueError("native threshold policy requires a finite detector threshold")
            threshold = float(native_result.threshold)
        assert threshold is not None
        predictions = test_scores > threshold
        tp, tn, fp, fn = self._confusion(split.test.labels, predictions)
        return AnomalyEvaluation(
            algorithm_name=native_result.algorithm_name,
            threshold=float(threshold),
            threshold_source=self.threshold_policy,
            sample_ids=split.test.signals.sample_ids,
            recording_ids=split.test.recording_ids,
            scores=test_scores,
            predictions=predictions,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
        )
