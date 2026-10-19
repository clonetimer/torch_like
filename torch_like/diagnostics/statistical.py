"""Train-only Gaussian baseline statistical detectors for multichannel signals.

These algorithms are deterministic baselines, not trained neural networks.
Scoring never changes fitted parameters and never consults evaluation labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from torch_like.diagnostics.types import DiagnosticResult, SignalBatch


class _GaussianBaseline:
    """Fit per-channel reference moments using development/healthy data only."""

    def __init__(self, *, min_std: float = 1e-8) -> None:
        if not np.isfinite(min_std) or min_std <= 0:
            raise ValueError("min_std must be finite and positive")
        self.min_std = float(min_std)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._names: tuple[str, ...] | None = None
        self._units: tuple[str, ...] | None = None
        self._rate: float | None = None

    def fit(self, reference: SignalBatch) -> "_GaussianBaseline":
        """Fit once on an explicitly supplied reference set, not inference data."""
        if not isinstance(reference, SignalBatch):
            raise TypeError("reference must be SignalBatch")
        if not np.isfinite(reference.values).all():
            raise ValueError("baseline fitting requires finite reference signals")
        if reference.shape[0] * reference.shape[1] < 2:
            raise ValueError("baseline fitting requires at least two temporal observations")
        flattened = reference.values.reshape(-1, reference.shape[2])
        mean = np.mean(flattened, axis=0)
        scale = np.maximum(np.std(flattened, axis=0, ddof=0), self.min_std)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("baseline moments overflowed; normalize the reference data")
        mean.setflags(write=False)
        scale.setflags(write=False)
        self._mean = mean
        self._scale = scale
        self._names = reference.channel_names
        self._units = reference.units
        self._rate = reference.sample_rate_hz
        return self

    @property
    def mean_(self) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("detector must be fitted before accessing baseline parameters")
        return self._mean.copy()

    @property
    def std_(self) -> np.ndarray:
        if self._scale is None:
            raise RuntimeError("detector must be fitted before accessing baseline parameters")
        return self._scale.copy()

    def _standardized(self, batch: SignalBatch) -> np.ndarray:
        if self._mean is None or self._scale is None:
            raise RuntimeError("fit a healthy reference batch before scoring")
        if not isinstance(batch, SignalBatch):
            raise TypeError("batch must be SignalBatch")
        if not np.isfinite(batch.values).all():
            raise ValueError("scoring requires finite inputs; impute or reject missing data upstream")
        if batch.channel_names != self._names:
            raise ValueError("channel_names must match the fitted reference, including ordering")
        if batch.units != self._units:
            raise ValueError("units must match the fitted reference")
        if batch.sample_rate_hz != self._rate:
            raise ValueError("sample_rate_hz differs from the fitted reference")
        return (batch.values - self._mean) / self._scale

    @staticmethod
    def _result(name: str, scores: np.ndarray, threshold: float) -> DiagnosticResult:
        return DiagnosticResult(
            task="anomaly_detection",
            output_kind="anomaly_score",
            values=scores,
            algorithm_name=name,
            threshold=threshold,
        )

    def alarm_mask(self, batch: SignalBatch) -> np.ndarray:
        """Return threshold comparisons without changing reference parameters."""
        result = self.score(batch)
        return result.values > result.threshold

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        raise NotImplementedError


class ZScoreDetector(_GaussianBaseline):
    """Absolute per-channel standardized residual with sigma threshold."""

    def __init__(self, *, sigma_threshold: float = 3.0, min_std: float = 1e-8) -> None:
        super().__init__(min_std=min_std)
        if not np.isfinite(sigma_threshold) or sigma_threshold <= 0:
            raise ValueError("sigma_threshold must be finite and positive")
        self.sigma_threshold = float(sigma_threshold)

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        return self._result("ZScoreDetector", np.abs(self._standardized(batch)), self.sigma_threshold)


class CUSUMDetector(_GaussianBaseline):
    """Two-sided tabular CUSUM per independent input window.

    k is the normalized allowance; threshold h is in accumulated standardized
    residual units. Positive/negative accumulators reset at every window boundary.
    No hidden state flows from one evaluation sample to the next.
    """

    def __init__(self, *, k: float = 0.5, h: float = 5.0, min_std: float = 1e-8) -> None:
        super().__init__(min_std=min_std)
        if not np.isfinite(k) or k < 0 or not np.isfinite(h) or h <= 0:
            raise ValueError("CUSUM requires finite k >= 0 and h > 0")
        self.k = float(k)
        self.h = float(h)

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        z = self._standardized(batch)
        n, length, channels = z.shape
        positive = np.zeros((n, channels), dtype=np.float64)
        negative = np.zeros((n, channels), dtype=np.float64)
        output = np.empty_like(z)
        for index in range(length):
            positive = np.maximum(0.0, positive + z[:, index, :] - self.k)
            negative = np.maximum(0.0, negative - z[:, index, :] - self.k)
            output[:, index, :] = np.maximum(positive, negative)
        return self._result("CUSUMDetector", output, self.h)


class EWMADetector(_GaussianBaseline):
    """Per-window standardized EWMA with an exact finite-horizon control limit.

    The initial standardized EWMA is zero. Each sample resets to baseline at its
    window start. Reported score is absolute EWMA divided by its finite-horizon
    reference standard deviation, with threshold in control-sigma units.
    """

    def __init__(self, *, alpha: float = 0.2, control_limit: float = 3.0, min_std: float = 1e-8) -> None:
        super().__init__(min_std=min_std)
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be finite and in (0, 1]")
        if not np.isfinite(control_limit) or control_limit <= 0:
            raise ValueError("control_limit must be finite and positive")
        self.alpha = float(alpha)
        self.control_limit = float(control_limit)

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        z = self._standardized(batch)
        n, length, channels = z.shape
        state = np.zeros((n, channels), dtype=np.float64)
        output = np.empty_like(z)
        for index in range(length):
            state = self.alpha * z[:, index, :] + (1.0 - self.alpha) * state
            variance = self.alpha / (2.0 - self.alpha) * (1.0 - (1.0 - self.alpha) ** (2 * (index + 1)))
            output[:, index, :] = np.abs(state) / np.sqrt(variance)
        return self._result("EWMADetector", output, self.control_limit)


def register_statistical_detectors(registry: Any = None) -> None:
    """Explicitly install the built-in detectors; no import-time global mutation."""
    from torch_like.diagnostics.registry import diagnostic_registry
    from torch_like.diagnostics.types import DiagnosticAlgorithmSpec

    target = diagnostic_registry if registry is None else registry
    for name, factory in (
        ("ZScoreDetector", ZScoreDetector),
        ("CUSUMDetector", CUSUMDetector),
        ("EWMADetector", EWMADetector),
    ):
        target.register(
            DiagnosticAlgorithmSpec(
                name=name,
                backend="statistical",
                task="anomaly_detection",
                output_kind="anomaly_score",
                operations=("fit", "score"),
            ),
            factory,
        )
