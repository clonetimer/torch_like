"""Leakage-safe threshold selection and event-oriented anomaly metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def _segments(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return 0.0 if denominator == 0 else (1 + beta_squared) * precision * recall / denominator


def _event_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    true_segments = _segments(y_true)
    predicted_segments = _segments(y_pred)
    true_positive = sum(any(_overlaps(actual, detected) for detected in predicted_segments) for actual in true_segments)
    false_negative = len(true_segments) - true_positive
    false_positive = sum(
        not any(_overlaps(detected, actual) for actual in true_segments) for detected in predicted_segments
    )
    redundant = sum(
        max(0, sum(_overlaps(actual, detected) for detected in predicted_segments) - 1) for actual in true_segments
    )
    return true_positive, false_positive, false_negative, redundant


def _corrected_event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_positive, false_positive, false_negative, redundant = _event_counts(y_true, y_pred)
    event_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    event_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    nominal = ~np.asarray(y_true, dtype=bool)
    true_negative_rate = float((~np.asarray(y_pred, dtype=bool))[nominal].mean()) if nominal.any() else 0.0
    corrected_precision = event_precision * true_negative_rate
    alarming_precision = true_positive / (true_positive + redundant) if true_positive + redundant else 0.0
    return {
        "event_precision_corrected": float(corrected_precision),
        "event_recall": float(event_recall),
        "event_f0_5_corrected": float(_fbeta(corrected_precision, event_recall)),
        "alarming_precision": float(alarming_precision),
        "event_true_positives": float(true_positive),
        "event_false_positives": float(false_positive),
        "event_false_negatives": float(false_negative),
    }


def evaluate_anomaly_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Evaluate channel scores on a uniform time grid using binary and event metrics."""

    truth = np.asarray(y_true, dtype=np.uint8)
    score_values = np.asarray(scores, dtype=np.float64)
    if truth.shape != score_values.shape or truth.ndim != 2:
        raise ValueError("Anomaly labels and scores must have matching shape (samples, channels)")
    if not np.isfinite(score_values).all():
        raise FloatingPointError("Anomaly scores contain NaN or infinite values")
    prediction = score_values >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth.reshape(-1), prediction.reshape(-1), average="binary", zero_division=0
    )
    aggregate_truth = truth.any(axis=1)
    aggregate_prediction = prediction.any(axis=1)
    metrics = {
        "point_precision": float(precision),
        "point_recall": float(recall),
        "point_f1": float(f1),
        "false_alarm_rate": float(aggregate_prediction[~aggregate_truth].mean()) if (~aggregate_truth).any() else 0.0,
        **_corrected_event_metrics(aggregate_truth, aggregate_prediction),
    }
    channel_counts = np.zeros(4, dtype=np.int64)
    for channel in range(truth.shape[1]):
        channel_counts += np.asarray(_event_counts(truth[:, channel], prediction[:, channel]), dtype=np.int64)
    channel_tp, channel_fp, channel_fn, _ = channel_counts.tolist()
    channel_precision = channel_tp / (channel_tp + channel_fp) if channel_tp + channel_fp else 0.0
    channel_recall = channel_tp / (channel_tp + channel_fn) if channel_tp + channel_fn else 0.0
    metrics["channel_event_precision"] = float(channel_precision)
    metrics["channel_event_recall"] = float(channel_recall)
    metrics["channel_event_f0_5"] = float(_fbeta(channel_precision, channel_recall))
    flat_truth = truth.reshape(-1)
    flat_scores = score_values.reshape(-1)
    if np.unique(flat_truth).size == 2:
        metrics["average_precision"] = float(average_precision_score(flat_truth, flat_scores))
        metrics["roc_auc"] = float(roc_auc_score(flat_truth, flat_scores))
    else:
        metrics["average_precision"] = 0.0
        metrics["roc_auc"] = 0.5
    return metrics


def select_anomaly_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric: str = "event_f0_5_corrected",
    candidates: int = 100,
) -> tuple[float, dict[str, float]]:
    """Select one global threshold using validation labels only."""

    score_values = np.asarray(scores, dtype=np.float64)
    quantiles = np.linspace(0.5, 0.9995, candidates)
    thresholds = np.unique(np.quantile(score_values, quantiles))
    best_threshold = float(thresholds[0])
    best_metrics = evaluate_anomaly_scores(y_true, score_values, best_threshold)
    for threshold in thresholds[1:]:
        metrics = evaluate_anomaly_scores(y_true, score_values, float(threshold))
        current_key = (metrics[metric], metrics["point_precision"], float(threshold))
        best_key = (best_metrics[metric], best_metrics["point_precision"], best_threshold)
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


__all__ = ["evaluate_anomaly_scores", "select_anomaly_threshold"]
