"""Post-hoc cluster-bootstrap uncertainty from already sealed R4c evidence.

Resampling is at whole-recording or whole-device level, never at window level.
No model, signals, individual test labels or inference calls are accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from torch_like.diagnostics.generalization import FaultDomainMetrics, FaultGeneralizationEvaluation


@dataclass(frozen=True)
class FaultMetricInterval:
    """Window-weighted point estimate and percentile interval over whole clusters."""

    estimate: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class FaultClusterUncertainty:
    """A frozen, JSON-ready summary; no resampled window predictions are retained."""

    cluster_by: str
    class_labels: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    device_count: int
    recording_count: int
    window_count: int
    seed: int
    n_resamples: int
    confidence: float
    min_class_clusters: int
    class_cluster_support: tuple[int, ...]
    missing_class_resamples: tuple[int, ...]
    resamples_with_any_missing_class: int
    accuracy: FaultMetricInterval
    macro_f1_all_classes: FaultMetricInterval

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "sealed_posthoc_cluster_percentile_bootstrap",
            "cluster_by": self.cluster_by,
            "metric_weighting": "windows_with_whole_cluster_resampling",
            "class_labels": list(self.class_labels),
            "cluster_ids": list(self.cluster_ids),
            "num_clusters": len(self.cluster_ids),
            "num_recordings": self.recording_count,
            "num_devices": self.device_count,
            "num_windows": self.window_count,
            "seed": self.seed,
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
            "min_class_clusters": self.min_class_clusters,
            "class_cluster_support": list(self.class_cluster_support),
            "missing_class_resamples": dict(zip(self.class_labels, self.missing_class_resamples, strict=True)),
            "resamples_with_any_missing_class": self.resamples_with_any_missing_class,
            "accuracy": self.accuracy.to_dict(),
            "macro_f1_all_classes": self.macro_f1_all_classes.to_dict(),
            "assumptions": [
                "Original recording and device IDs faithfully represent independent physical sources.",
                "Whole clusters are sampled with replacement; windows within clusters remain correlated.",
                "Metrics remain window-weighted, not a mean of per-recording or per-device metrics.",
                "The fixed full-class macro-F1 assigns zero to classes absent in a bootstrap replicate.",
                "Percentile intervals are descriptive conditional on observed source clusters.",
                "Neither external-domain generalization nor causal fault-source probability is established.",
            ],
        }


def _audited_slices(evaluation: FaultGeneralizationEvaluation) -> dict[str, tuple[FaultDomainMetrics, ...]]:
    """Check marginal evidence against the sealed report before touching the RNG."""
    if not isinstance(evaluation, FaultGeneralizationEvaluation):
        raise TypeError("expected an already sealed FaultGeneralizationEvaluation")
    overall = evaluation.overall
    labels = tuple(overall.class_labels)
    n = len(overall.recording_ids)
    counts = np.asarray(overall.confusion)
    if counts.shape != (len(labels), len(labels)) or counts.sum() != n or n < 1:
        raise ValueError("invalid sealed overall confusion/counts")
    if len(overall.predictions) != n or len(overall.sample_ids) != n or len(set(overall.sample_ids)) != n:
        raise ValueError("sealed sample or prediction provenance is incomplete")
    if len(set(overall.recording_ids)) < 1 or any(not value for value in overall.recording_ids):
        raise ValueError("sealed original recording provenance is incomplete")
    prediction_columns = np.array([labels.index(name) for name in overall.predictions], dtype=int)
    if not np.array_equal(np.bincount(prediction_columns, minlength=len(labels)), counts.sum(axis=0)):
        raise ValueError("sealed predictions do not match confusion column totals")
    groups: dict[str, list[FaultDomainMetrics]] = {axis: [] for axis in ("recording", "device", "condition", "pair")}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for part in evaluation.slices:
        if not isinstance(part, FaultDomainMetrics) or part.axis not in groups:
            raise ValueError("unexpected or untyped sealed domain slice")
        identity = tuple(part.identity)
        if not identity or any(not isinstance(x, str) or not x.strip() for x in identity):
            raise ValueError("domain identity must contain nonempty strings")
        if len(identity) != (2 if part.axis == "pair" else 1):
            raise ValueError("domain identity does not match its axis")
        key = (part.axis, identity)
        if key in seen:
            raise ValueError("duplicate sealed source identity")
        seen.add(key)
        if tuple(part.class_labels) != labels:
            raise ValueError("domain class ordering disagrees with sealed vocabulary")
        matrix = np.asarray(part.confusion)
        if (
            matrix.shape != counts.shape
            or not np.issubdtype(matrix.dtype, np.integer)
            or np.any(matrix < 0)
            or int(matrix.sum()) != part.window_count
            or part.recording_count < 1
            or part.recording_count > part.window_count
        ):
            raise ValueError("invalid domain confusion or source counts")
        groups[part.axis].append(part)
    for axis, parts in groups.items():
        if not parts:
            raise ValueError(f"missing {axis} domain provenance")
        if not np.array_equal(np.sum([part.confusion for part in parts], axis=0), counts):
            raise ValueError(f"{axis} domain confusion fails sealed overall reconciliation")
        if sum(part.window_count for part in parts) != n:
            raise ValueError(f"{axis} domain counts do not cover the sealed set")
    recordings = groups["recording"]
    actual = set(overall.recording_ids)
    if {part.identity[0] for part in recordings} != actual:
        raise ValueError("per-recording identities do not match sealed source IDs")
    for part in recordings:
        positions = [i for i, value in enumerate(overall.recording_ids) if value == part.identity[0]]
        expected = np.bincount(prediction_columns[positions], minlength=len(labels))
        if part.recording_count != 1 or part.window_count != len(positions):
            raise ValueError("per-recording counts contradict sealed sample provenance")
        if not np.array_equal(np.asarray(part.confusion).sum(axis=0), expected):
            raise ValueError("per-recording predicted counts contradict sealed predictions")
    devices = groups["device"]
    if {part.identity[0] for part in devices} != set(evaluation.test_devices):
        raise ValueError("device identities disagree with sealed evaluation metadata")
    if {part.identity[0] for part in groups["condition"]} != set(evaluation.test_conditions):
        raise ValueError("condition identities disagree with sealed evaluation metadata")
    if sum(part.recording_count for part in devices) != len(recordings):
        raise ValueError("device recording counts do not reconcile with independent recording count")
    if sum(part.recording_count for part in groups["condition"]) != len(recordings):
        raise ValueError("condition recording counts do not reconcile with independent recording count")
    if sum(part.recording_count for part in groups["pair"]) != len(recordings):
        raise ValueError("pair recording counts do not reconcile with independent recording count")
    for axis, parts in groups.items():
        groups[axis] = sorted(parts, key=lambda part: part.identity)
    return {axis: tuple(parts) for axis, parts in groups.items()}


def bootstrap_fault_generalization(
    evaluation: FaultGeneralizationEvaluation,
    *,
    cluster_by: str = "recording",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    min_clusters: int = 3,
    min_class_clusters: int = 2,
) -> FaultClusterUncertainty:
    """Compute fixed-label, window-weighted percentile CIs on sealed confusion.

    Device intervals require a true R4c unseen-device holdout and independent
    device clusters. Recording intervals do not imply independent devices.
    """
    if cluster_by not in {"recording", "device"}:
        raise ValueError("cluster_by must be recording or device")
    if type(n_resamples) is not int or n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if type(confidence) not in {float, int} or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be a finite value strictly between zero and one")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(min_clusters) is not int or min_clusters < 3:
        raise ValueError("min_clusters must be at least 3 independent physical sources")
    if type(min_class_clusters) is not int or min_class_clusters < 2:
        raise ValueError("min_class_clusters must be at least 2 independent sources per class")
    groups = _audited_slices(evaluation)
    if cluster_by == "device" and evaluation.holdout not in {"device", "both"}:
        raise ValueError("device-level uncertainty requires an unseen-device holdout")
    selected = groups[cluster_by]
    if len(selected) < min_clusters:
        raise ValueError("insufficient independent source clusters for bootstrap")
    confusions = np.stack([part.confusion for part in selected])
    class_presence = np.sum(confusions.sum(axis=2) > 0, axis=0)
    if np.any(class_presence < min_class_clusters):
        raise ValueError("insufficient independent class-bearing clusters for bootstrap")
    if not np.array_equal(confusions.sum(axis=0), evaluation.overall.confusion):
        raise ValueError("selected clusters do not reconcile with sealed confusion")
    rng = np.random.default_rng(seed)
    repeats = rng.multinomial(
        len(selected), np.full(len(selected), 1 / len(selected)), size=n_resamples
    )
    aggregated = np.tensordot(repeats, confusions, axes=(1, 0))
    windows = aggregated.sum(axis=(1, 2))
    if np.any(windows == 0):
        raise ValueError("bootstrap created a zero-window sample")
    accuracy_samples = np.trace(aggregated, axis1=1, axis2=2) / windows
    true_support = aggregated.sum(axis=2)
    predicted_support = aggregated.sum(axis=1)
    tp = np.diagonal(aggregated, axis1=1, axis2=2)
    denominator = true_support + predicted_support
    f1_per_class = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator > 0,
    )
    f1_samples = f1_per_class.mean(axis=1)
    tail = (1 - float(confidence)) / 2

    def interval(point: float, samples: np.ndarray) -> FaultMetricInterval:
        lower, upper = np.quantile(samples, [tail, 1 - tail])
        return FaultMetricInterval(float(point), float(lower), float(upper))

    return FaultClusterUncertainty(
        cluster_by=cluster_by,
        class_labels=tuple(evaluation.overall.class_labels),
        cluster_ids=tuple("|".join(part.identity) for part in selected),
        device_count=len(groups["device"]),
        recording_count=len(groups["recording"]),
        window_count=len(evaluation.overall.sample_ids),
        seed=seed,
        n_resamples=n_resamples,
        confidence=float(confidence),
        min_class_clusters=min_class_clusters,
        class_cluster_support=tuple(int(x) for x in class_presence),
        missing_class_resamples=tuple(int(x) for x in (true_support == 0).sum(axis=0)),
        resamples_with_any_missing_class=int(np.any(true_support == 0, axis=1).sum()),
        accuracy=interval(evaluation.overall.accuracy, accuracy_samples),
        macro_f1_all_classes=interval(evaluation.overall.macro_f1, f1_samples),
    )
