"""Explicit streaming anomaly-threshold state, independent of datasets and runners.

This opt-in threshold algorithm consumes *already computed* anomaly scores
at time/channel resolution. It does not fit, infer sample rate, select healthy
data, tune thresholds on inference labels, or alter upstream score algorithms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral, Real
from typing import Any

import numpy as np

from torch_like.diagnostics.decisions import DiagnosticDecision
from torch_like.diagnostics.types import DiagnosticOutputKind, DiagnosticResult, DiagnosticTask

_STREAM_STATE_VERSION = 1


def _vector(values: float | Sequence[float], name: str, *, strictly_positive: bool = False) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.ndim > 1:
        raise ValueError(f"{name} must be a finite real scalar or one-dimensional vector")
    result = np.atleast_1d(raw).astype(np.float64, copy=True)
    if not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    if strictly_positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return result


def _scalar(value: Real, name: str, *, positive: bool = False, at_most_one: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if at_most_one and result > 1:
        raise ValueError(f"{name} must be at most 1")
    return result


class StreamingThreshold:
    """Turn a single continuous score stream into an explicit decision stream.

    A chunk must be one sample with shape (1, time, channels) and carry an
    explicit absolute `start_index`. A source model's score windows are *not*
    automatically joined or assumed continuous. For an independent stream,
    instantiate a fresh object or explicitly call reset().

    In adaptive mode the caller supplies development-time mean/std. At time t
    the threshold is mean[t-1] + sigma * std[t-1]; state updates only *after*
    the decision, optionally freezing on detected anomalies. Thus no current
    score is used to choose its own threshold. No labels are accessed.
    """

    def __init__(
        self,
        *,
        algorithm_name: str,
        mode: str,
        thresholds: float | Sequence[float] | None = None,
        initial_mean: float | Sequence[float] | None = None,
        initial_std: float | Sequence[float] | None = None,
        alpha: float = 0.2,
        sigma: float = 3.0,
        min_std: float = 1e-8,
        freeze_on_alarm: bool = True,
        channel_names: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(algorithm_name, str) or not algorithm_name.strip():
            raise ValueError("algorithm_name must be a nonempty string")
        if mode not in {"fixed", "adaptive_ewma"}:
            raise ValueError("mode must be fixed or adaptive_ewma")
        if not isinstance(freeze_on_alarm, bool):
            raise TypeError("freeze_on_alarm must be boolean")
        alpha = _scalar(alpha, "alpha", positive=True, at_most_one=True)
        sigma = _scalar(sigma, "sigma", positive=True)
        min_std = _scalar(min_std, "min_std", positive=True)

        if mode == "fixed":
            if thresholds is None or initial_mean is not None or initial_std is not None:
                raise ValueError("fixed mode requires thresholds and forbids adaptive reference")
            cutoffs = _vector(thresholds, "thresholds")
            mean = variance = None
        else:
            if thresholds is not None or initial_mean is None or initial_std is None:
                raise ValueError("adaptive_ewma requires initial_mean and initial_std, not thresholds")
            mean = _vector(initial_mean, "initial_mean")
            std = _vector(initial_std, "initial_std", strictly_positive=True)
            if mean.shape != std.shape:
                raise ValueError("initial_mean and initial_std channel dimensions differ")
            if np.any(std < min_std):
                raise ValueError("initial_std must be at least min_std")
            variance = std**2
            if not np.isfinite(variance).all() or np.any(variance <= 0):
                raise ValueError("initial_std squared must be finite and positive")
            cutoffs = None

        channels = len(cutoffs) if mode == "fixed" else len(mean)
        if channel_names is not None:
            channel_names = tuple(channel_names)
            if (
                len(channel_names) != channels
                or len(set(channel_names)) != channels
                or any(not isinstance(name, str) or not name.strip() for name in channel_names)
            ):
                raise ValueError("channel_names must uniquely identify each threshold channel")
        self._initial_mean = mean
        self._initial_variance = variance
        self._initial_cutoffs = cutoffs
        self._config: dict[str, Any] = {
            "algorithm_name": algorithm_name,
            "mode": mode,
            "initial_thresholds": None if cutoffs is None else cutoffs.tolist(),
            "initial_mean": None if mean is None else mean.tolist(),
            "initial_std": None if variance is None else np.sqrt(variance).tolist(),
            "alpha": alpha,
            "sigma": sigma,
            "min_std": min_std,
            "freeze_on_alarm": freeze_on_alarm,
            "channel_names": None if channel_names is None else list(channel_names),
        }
        self.reset()

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def config(self) -> dict[str, Any]:
        return deepcopy(self._config)

    def reset(self) -> None:
        """Explicitly start a new, independent continuous score stream."""
        self._observations = 0
        self._mean = None if self._initial_mean is None else self._initial_mean.copy()
        self._variance = None if self._initial_variance is None else self._initial_variance.copy()

    def decide_chunk(
        self,
        result: DiagnosticResult,
        *,
        start_index: int,
        channel_names: tuple[str, ...] | None = None,
    ) -> DiagnosticDecision:
        """Decide before adaptation; commit mutable state only on full success."""
        if not isinstance(result, DiagnosticResult):
            raise TypeError("result must be a DiagnosticResult")
        if result.task != DiagnosticTask.ANOMALY_DETECTION or result.output_kind != DiagnosticOutputKind.ANOMALY_SCORE:
            raise ValueError("streaming threshold requires anomaly score output")
        if result.algorithm_name != self._config["algorithm_name"]:
            raise ValueError("algorithm_name differs from fitted streaming threshold configuration")
        if result.values.ndim != 3 or result.values.shape[0] != 1:
            raise ValueError("streaming scores must have shape (1, time, channels)")
        expected_channels = len(self._initial_cutoffs) if self._initial_cutoffs is not None else len(self._initial_mean)
        if result.values.shape[2] != expected_channels:
            raise ValueError("streaming score channel count changed")
        if self._config["channel_names"] is not None:
            if channel_names is None or tuple(channel_names) != tuple(self._config["channel_names"]):
                raise ValueError("channel_names must match the declared order on every chunk")
        if isinstance(start_index, bool) or not isinstance(start_index, Integral) or start_index != self._observations:
            raise ValueError("start_index must equal next expected absolute stream index")

        x = result.values[0]
        step_count = x.shape[0]
        thresholds = np.empty_like(x)
        mean = None if self._mean is None else self._mean.copy()
        variance = None if self._variance is None else self._variance.copy()
        for time in range(step_count):
            if self._config["mode"] == "fixed":
                current = self._initial_cutoffs
            else:
                current = mean + self._config["sigma"] * np.sqrt(variance)
            if not np.isfinite(current).all():
                raise ValueError("streaming threshold became nonfinite; state not committed")
            thresholds[time] = current
            if mean is not None:
                alarm = x[time] > current
                eligible = ~alarm if self._config["freeze_on_alarm"] else np.ones_like(alarm, dtype=bool)
                delta = x[time, eligible] - mean[eligible]
                a = self._config["alpha"]
                new_variance = (1 - a) * (variance[eligible] + a * delta**2)
                new_mean = mean[eligible] + a * delta
                if not np.isfinite(new_mean).all() or not np.isfinite(new_variance).all():
                    raise ValueError("streaming adaptation overflowed; state not committed")
                mean[eligible] = new_mean
                variance[eligible] = np.maximum(new_variance, self._config["min_std"] ** 2)

        decision = DiagnosticDecision.from_threshold(result, thresholds[np.newaxis])
        self._mean = mean
        self._variance = variance
        self._observations += step_count
        return decision

    def state_dict(self) -> dict[str, Any]:
        """Detached, JSON-compatible checkpoint with version and full config."""
        return {
            "schema_version": _STREAM_STATE_VERSION,
            "config": deepcopy(self._config),
            "observations": self._observations,
            "mean": None if self._mean is None else self._mean.tolist(),
            "variance": None if self._variance is None else self._variance.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate entirely before atomic restoration. Only load trusted state."""
        if not isinstance(state, Mapping) or set(state) != {
            "schema_version",
            "config",
            "observations",
            "mean",
            "variance",
        }:
            raise ValueError("invalid streaming checkpoint structure")
        if type(state["schema_version"]) is not int or state["schema_version"] != _STREAM_STATE_VERSION:
            raise ValueError("unknown streaming checkpoint schema version")
        if state["config"] != self._config:
            raise ValueError("streaming checkpoint configuration mismatch")
        count = state["observations"]
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("invalid streaming checkpoint observations")

        if self._config["mode"] == "fixed":
            if state["mean"] is not None or state["variance"] is not None:
                raise ValueError("fixed stream must not store adaptive state")
            mean = variance = None
        else:
            if not isinstance(state["mean"], list) or not isinstance(state["variance"], list):
                raise ValueError("adaptive checkpoint requires numeric mean and variance")
            mean = _vector(state["mean"], "checkpoint mean")
            variance = _vector(state["variance"], "checkpoint variance", strictly_positive=True)
            expected = len(self._initial_mean)
            if len(mean) != expected or len(variance) != expected:
                raise ValueError("checkpoint channel count changed")
            if np.any(variance < self._config["min_std"] ** 2):
                raise ValueError("checkpoint variance below configured minimum")
        self._observations = int(count)
        self._mean = mean
        self._variance = variance


__all__ = ["StreamingThreshold"]
