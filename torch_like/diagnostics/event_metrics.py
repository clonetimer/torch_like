"""Robust event-wise anomaly metrics used by industrial TSAD evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EventWiseMetrics:
    """Paper-style event F1 and its point/event components."""

    precision: float
    recall: float
    f1: float
    true_positive_events: int
    false_positive_events: int
    false_negative_events: int
    point_precision: float
    normal_specificity: float


def _binary_series(values: Iterable[int], name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional binary series")
    if not np.isin(raw, (0, 1, False, True)).all():
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return raw.astype(bool)


def _intervals(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True))


def _validate_intervals(
    intervals: Iterable[tuple[int, int]],
    *,
    length: int,
) -> tuple[tuple[int, int], ...]:
    result = []
    previous_end = 0
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise ValueError("ground-truth intervals must be (start, end_exclusive) pairs")
        start, end = interval
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, np.integer))
            or not isinstance(end, (int, np.integer))
            or start < 0
            or end <= start
            or end > length
            or start < previous_end
        ):
            raise ValueError("ground-truth intervals must be sorted, disjoint and within the series")
        result.append((int(start), int(end)))
        previous_end = int(end)
    return tuple(result)


def compute_event_wise_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    gt_intervals: Iterable[tuple[int, int]] | None = None,
) -> EventWiseMetrics:
    """Compute the ALDM reference event-wise index with safe empty-set handling.

    Event recall counts a ground-truth event as detected when any predicted
    point overlaps it. Event precision follows the reference implementation:
    point precision multiplied by normal-point specificity. A predicted
    segment that overlaps no ground-truth event is counted as a false-positive
    event. This metric is intended for a single contiguous time series after
    window scores have been aligned and aggregated by the caller.
    """
    truth = _binary_series(y_true, "y_true")
    prediction = _binary_series(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical lengths")

    true_events = _intervals(truth) if gt_intervals is None else _validate_intervals(gt_intervals, length=truth.size)
    predicted_events = _intervals(prediction)

    detected = 0
    for start, end in true_events:
        if prediction[start:end].any():
            detected += 1

    false_positive_events = sum(not truth[start:end].any() for start, end in predicted_events)
    false_negative_events = len(true_events) - detected

    predicted_points = int(prediction.sum())
    true_positive_points = int(np.logical_and(truth, prediction).sum())
    normal_points = int((~truth).sum())
    false_positive_points = predicted_points - true_positive_points

    point_precision = true_positive_points / predicted_points if predicted_points else 0.0
    normal_specificity = 1.0 - false_positive_points / normal_points if normal_points else (1.0 if false_positive_points == 0 else 0.0)
    precision = point_precision * normal_specificity
    recall = detected / len(true_events) if true_events else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return EventWiseMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positive_events=int(detected),
        false_positive_events=int(false_positive_events),
        false_negative_events=int(false_negative_events),
        point_precision=float(point_precision),
        normal_specificity=float(normal_specificity),
    )


__all__ = ["EventWiseMetrics", "compute_event_wise_metrics"]
