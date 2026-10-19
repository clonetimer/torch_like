"""Turn diagnostic anomaly scores into auditable, non-overlapping window events.

Time resolution is explicitly declared; per-window scores never acquire invented
onset/offset times. This module does not inspect ground-truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torch_like.diagnostics.types import DiagnosticOutputKind, DiagnosticResult, DiagnosticTask, SignalBatch


@dataclass(frozen=True)
class DiagnosticEvent:
    """One threshold crossing episode within an individual input sample/window."""

    algorithm_name: str
    sample_index: int
    sample_id: str | None
    resolution: str
    channel_name: str | None
    start_index: int | None
    end_index_exclusive: int | None
    first_time_s: float | None
    last_time_s: float | None
    active_points: int | None
    peak_score: float
    threshold: float


def _episodes(mask: np.ndarray, *, min_active_points: int, gap_tolerance: int):
    hits = np.flatnonzero(mask)
    if hits.size == 0:
        return
    begin = int(hits[0])
    last = begin
    active = 1
    for next_hit in hits[1:]:
        index = int(next_hit)
        if index - last - 1 <= gap_tolerance:
            last = index
            active += 1
        else:
            if active >= min_active_points:
                yield begin, last + 1, active
            begin = index
            last = index
            active = 1
    if active >= min_active_points:
        yield begin, last + 1, active


def extract_anomaly_events(
    result: DiagnosticResult,
    batch: SignalBatch,
    *,
    score_layout: str | None = None,
    min_active_points: int = 1,
    gap_tolerance: int = 0,
) -> tuple[DiagnosticEvent, ...]:
    """Extract threshold episodes while preserving the source score resolution.

    Layouts:
      - window: (N,) one score per entire window, time indices unavailable
      - window_channel: (N,C), a separate window-level score per channel
      - time: (N,L), one time-resolved score per window
      - time_channel: (N,L,C), time-resolved scores for each channel

    The ambiguous 2-D shape requires an explicit layout. Time-resolved
    durations count *active points*, not the included tolerated gaps.
    """
    if not isinstance(result, DiagnosticResult) or not isinstance(batch, SignalBatch):
        raise TypeError("result must be DiagnosticResult and batch must be SignalBatch")
    if result.task != DiagnosticTask.ANOMALY_DETECTION or result.output_kind != DiagnosticOutputKind.ANOMALY_SCORE:
        raise ValueError("only anomaly_detection/anomaly_score outputs support event extraction")
    if result.threshold is None:
        raise ValueError("anomaly event extraction requires an explicit fixed threshold")
    if isinstance(min_active_points, bool) or not isinstance(min_active_points, int) or min_active_points < 1:
        raise ValueError("min_active_points must be a positive integer")
    if isinstance(gap_tolerance, bool) or not isinstance(gap_tolerance, int) or gap_tolerance < 0:
        raise ValueError("gap_tolerance must be a nonnegative integer")
    scores = result.values
    if scores.shape[0] != batch.shape[0]:
        raise ValueError("score sample count must match the input batch")
    expected = {1: "window", 2: None, 3: "time_channel"}[scores.ndim]
    if scores.ndim == 2:
        if score_layout not in {"time", "window_channel"}:
            raise ValueError("2-D scores require explicit score_layout='time' or 'window_channel'")
        layout = score_layout
    else:
        if score_layout is not None and score_layout != expected:
            raise ValueError("score_layout conflicts with unambiguous score dimensions")
        layout = expected
    if layout in {"time", "time_channel"} and scores.shape[1] != batch.shape[1]:
        raise ValueError("time-resolved scores must match the input time axis")
    if layout in {"window_channel", "time_channel"} and scores.shape[-1] != batch.shape[2]:
        raise ValueError("channel-resolved scores must match the input channel axis")
    if layout in {"window", "window_channel"} and (min_active_points != 1 or gap_tolerance != 0):
        raise ValueError("window-level scores do not support temporal debounce or gap merging")

    threshold = float(result.threshold)
    events: list[DiagnosticEvent] = []

    def identify(sample: int) -> str | None:
        return None if batch.sample_ids is None else batch.sample_ids[sample]

    def append_window(sample: int, score: float, channel: str | None) -> None:
        if score > threshold:
            events.append(
                DiagnosticEvent(
                    result.algorithm_name, sample, identify(sample), "window", channel,
                    None, None, None, None, None, float(score), threshold,
                )
            )

    if layout == "window":
        for sample, score in enumerate(scores):
            append_window(sample, float(score), None)
    elif layout == "window_channel":
        for sample in range(scores.shape[0]):
            for channel in range(scores.shape[1]):
                append_window(sample, float(scores[sample, channel]), batch.channel_names[channel])
    else:
        channels = range(scores.shape[2]) if layout == "time_channel" else (None,)
        for sample in range(scores.shape[0]):
            for channel in channels:
                values = scores[sample] if channel is None else scores[sample, :, channel]
                for begin, end, active in _episodes(
                    values > threshold,
                    min_active_points=min_active_points,
                    gap_tolerance=gap_tolerance,
                ):
                    if batch.timestamps_s is None:
                        first_s = begin / batch.sample_rate_hz
                        last_s = (end - 1) / batch.sample_rate_hz
                    else:
                        first_s = float(batch.timestamps_s[sample, begin])
                        last_s = float(batch.timestamps_s[sample, end - 1])
                    confirmed = values[begin:end]
                    peak = float(np.max(confirmed))
                    events.append(
                        DiagnosticEvent(
                            result.algorithm_name,
                            sample,
                            identify(sample),
                            "time",
                            None if channel is None else batch.channel_names[channel],
                            begin,
                            end,
                            first_s,
                            last_s,
                            active,
                            peak,
                            threshold,
                        )
                    )
    return tuple(events)
