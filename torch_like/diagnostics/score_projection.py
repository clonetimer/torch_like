"""AAS-informed score-axis projection, independent of platform detector wrappers.

Every dimension is explicitly declared: never silently join independent
windows into a continuous stream, average incompatible scores, or calibrate
thresholds from test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from torch_like.diagnostics.types import DiagnosticOutputKind, DiagnosticResult, DiagnosticTask


@dataclass(frozen=True)
class AnomalyScoreProjection:
    """A score vector whose resolution is explicit and cannot be misread."""

    values: np.ndarray
    resolution: str
    algorithm_name: str
    source_shape: tuple[int, ...]
    score_layout: str
    channel_reduction: str | None

    def __post_init__(self) -> None:
        if self.resolution not in {"time", "window"}:
            raise ValueError("resolution must be time or window")
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 1 or not arr.size or not np.isfinite(arr).all():
            raise ValueError("projected scores must be a nonempty finite vector")
        arr = np.array(arr, copy=True)
        arr.setflags(write=False)
        object.__setattr__(self, "values", arr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "resolution": self.resolution,
            "algorithm_name": self.algorithm_name,
            "source_shape": list(self.source_shape),
            "score_layout": self.score_layout,
            "channel_reduction": self.channel_reduction,
        }


def project_anomaly_scores(
    result: DiagnosticResult,
    *,
    score_layout: str,
    sequence_mode: str,
    channel_reduction: str | None = None,
) -> AnomalyScoreProjection:
    """Turn typed anomaly scores into explicit AAS-like observation scores.

    Input layouts:
    - window: (N,) per independent window
    - window_channel: (N,C) one score per channel/window
    - time: (N,L) a time-resolved score per independent sample
    - time_channel: (N,L,C) a time/channel-resolved score

    sequence_mode="continuous" requires N=1 and time-resolved scores,
    returning L values. "independent_windows" returns N values using the
    *last* timestep for time-resolved scores. Channel reduction is explicitly
    declared as max or mean where applicable; unneeded reductions are rejected.

    This is a numerical view only: score thresholds and alarm decisions do not
    transfer implicitly, and no information from one window is used to
    construct another.
    """
    if not isinstance(result, DiagnosticResult):
        raise TypeError("result must be DiagnosticResult")
    if result.task != DiagnosticTask.ANOMALY_DETECTION or result.output_kind != DiagnosticOutputKind.ANOMALY_SCORE:
        raise ValueError("projection requires anomaly_detection/anomaly_score")
    ranks = {"window": 1, "window_channel": 2, "time": 2, "time_channel": 3}
    if score_layout not in ranks:
        raise ValueError("score_layout must be window, window_channel, time or time_channel")
    scores = result.values
    if scores.ndim != ranks[score_layout]:
        raise ValueError("score_layout does not match result score shape")
    if sequence_mode not in {"continuous", "independent_windows"}:
        raise ValueError("sequence_mode must be continuous or independent_windows")

    has_channels = score_layout in {"window_channel", "time_channel"}
    if has_channels and channel_reduction not in {"max", "mean"}:
        raise ValueError("channel_reduction must be explicitly max or mean for channel scores")
    if not has_channels and channel_reduction is not None:
        raise ValueError("channel_reduction is not applicable to channel-free scores")

    if sequence_mode == "continuous":
        if scores.shape[0] != 1 or score_layout not in {"time", "time_channel"}:
            raise ValueError("continuous sequence requires one time-resolved source sample")
        projected = scores[0]  # (L,) or (L,C)
        resolution = "time"
    else:
        if score_layout in {"time", "time_channel"}:
            projected = scores[:, -1]  # (N,) or (N,C)
        else:
            projected = scores
        resolution = "window"

    if has_channels:
        projected = np.max(projected, axis=-1) if channel_reduction == "max" else np.mean(projected, axis=-1)

    return AnomalyScoreProjection(
        values=projected,
        resolution=resolution,
        algorithm_name=result.algorithm_name,
        source_shape=tuple(scores.shape),
        score_layout=score_layout,
        channel_reduction=channel_reduction,
    )


__all__ = ["AnomalyScoreProjection", "project_anomaly_scores"]
