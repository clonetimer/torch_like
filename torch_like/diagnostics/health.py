"""Healthy-reference health indices without dataset-provenance governance.

One score per input window. A high value means closer to the user's fitted
reference, not an absolute physical health percentage or failure probability.
"""

from __future__ import annotations

import numpy as np

from torch_like.diagnostics.types import DiagnosticAlgorithmSpec, DiagnosticResult, SignalBatch


class _ReferenceHealthIndex:
    """Reference-only fit, standardized deviation, and bounded index mapping."""

    def __init__(self, *, health_scale: float = 1.0, min_std: float = 1e-8) -> None:
        if not np.isfinite(health_scale) or health_scale <= 0:
            raise ValueError("health_scale must be finite and positive")
        if not np.isfinite(min_std) or min_std <= 0:
            raise ValueError("min_std must be finite and positive")
        self.health_scale = float(health_scale)
        self.min_std = float(min_std)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._names: tuple[str, ...] | None = None
        self._units: tuple[str, ...] | None = None
        self._sample_rate: float | None = None

    def fit(self, reference: SignalBatch) -> "_ReferenceHealthIndex":
        """Fit to a caller-selected healthy reference; no labels or split needed."""
        if not isinstance(reference, SignalBatch):
            raise TypeError("reference must be SignalBatch")
        if not np.isfinite(reference.values).all():
            raise ValueError("healthy reference requires finite numeric signals")
        n, t, channels = reference.shape
        if n * t < 2:
            raise ValueError("healthy reference requires at least two observations")
        flattened = reference.values.reshape(-1, channels)
        mean = flattened.mean(axis=0)
        std = np.maximum(flattened.std(axis=0, ddof=1), self.min_std)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("reference moments overflowed; rescale input signals")
        self._fit_metric(flattened, mean, std)
        mean.setflags(write=False)
        std.setflags(write=False)
        self._mean = mean
        self._std = std
        self._names = reference.channel_names
        self._units = reference.units
        self._sample_rate = reference.sample_rate_hz
        return self

    def _fit_metric(self, flattened: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        """Subclass extension point for correlation-aware healthy reference."""

    def _standardized(self, batch: SignalBatch) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("fit a user-supplied healthy reference before scoring")
        if not isinstance(batch, SignalBatch):
            raise TypeError("batch must be SignalBatch")
        if not np.isfinite(batch.values).all():
            raise ValueError("health scoring requires finite numeric signals")
        if batch.channel_names != self._names:
            raise ValueError("channel_names and channel ordering differ from the reference")
        if batch.units != self._units:
            raise ValueError("units differ from the reference")
        if batch.sample_rate_hz != self._sample_rate:
            raise ValueError("sample_rate_hz differs from the reference")
        standardized = (batch.values - self._mean) / self._std
        if not np.isfinite(standardized).all():
            raise ValueError("health standardization produced nonfinite values")
        return standardized

    def _deviation(self, standardized: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(standardized**2, axis=(1, 2)))

    def deviation(self, batch: SignalBatch) -> np.ndarray:
        """Return dimensionless RMS deviation from the fitted healthy reference."""
        result = np.asarray(self._deviation(self._standardized(batch)), dtype=np.float64)
        if result.ndim != 1 or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError("health deviation must be a finite nonnegative value per window")
        result.setflags(write=False)
        return result

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        """Exponential index exp(-deviation / health_scale), bounded in [0, 1]."""
        values = np.exp(-self.deviation(batch) / self.health_scale)
        return DiagnosticResult(
            task="health_assessment",
            output_kind="health_index",
            values=values,
            algorithm_name=self.__class__.__name__,
        )


class GaussianHealthIndex(_ReferenceHealthIndex):
    """RMS channel-wise z-distance; ignores cross-channel covariance."""


class MahalanobisHealthIndex(_ReferenceHealthIndex):
    """Regularized channel-covariance RMS Mahalanobis reference distance."""

    def __init__(self, *, health_scale: float = 1.0, min_std: float = 1e-8, shrinkage: float = 0.1) -> None:
        super().__init__(health_scale=health_scale, min_std=min_std)
        if not np.isfinite(shrinkage) or not 0 < shrinkage <= 1:
            raise ValueError("shrinkage must be finite and in (0, 1]")
        self.shrinkage = float(shrinkage)
        self._precision: np.ndarray | None = None

    def _fit_metric(self, flattened: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        standardized = (flattened - mean) / std
        if not np.isfinite(standardized).all():
            raise ValueError("reference standardization produced nonfinite values")
        channels = flattened.shape[1]
        covariance = (standardized.T @ standardized) / (len(flattened) - 1)
        regularized = (1 - self.shrinkage) * covariance + self.shrinkage * np.eye(channels)
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
            raise ValueError("regularized covariance is not positive definite")
        precision = (eigenvectors / eigenvalues) @ eigenvectors.T
        if not np.isfinite(precision).all():
            raise ValueError("health reference precision matrix is nonfinite")
        precision.setflags(write=False)
        self._precision = precision

    def _deviation(self, standardized: np.ndarray) -> np.ndarray:
        if self._precision is None:
            raise RuntimeError("fit the covariance reference before scoring")
        squared = np.einsum("ntc,cd,ntd->nt", standardized, self._precision, standardized)
        return np.sqrt(np.maximum(np.mean(squared, axis=1) / standardized.shape[2], 0.0))


def register_health_index_estimators(registry=None) -> None:
    """Opt-in native algorithm registration; import has no global side effects."""
    from torch_like.diagnostics.registry import diagnostic_registry

    target = diagnostic_registry if registry is None else registry
    for name, factory in (
        ("GaussianHealthIndex", GaussianHealthIndex),
        ("MahalanobisHealthIndex", MahalanobisHealthIndex),
    ):
        target.register(
            DiagnosticAlgorithmSpec(
                name=name,
                backend="statistical",
                task="health_assessment",
                output_kind="health_index",
                operations=("fit", "score"),
                description="Unsupervised healthy-reference index, not physical remaining life",
            ),
            factory,
        )
