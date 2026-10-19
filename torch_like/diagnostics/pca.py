"""Train/calibration-separated PCA T-squared and squared prediction error monitors."""

from __future__ import annotations

import numpy as np

from torch_like.diagnostics.types import DiagnosticAlgorithmSpec, DiagnosticResult, SignalBatch


class _PCABaseline:
    """Channel PCA fitted on healthy development observations only.

    Reference whitening uses sample variance (ddof=1). The user must select
    nonoverlapping development calibration data externally; there is no
    automatic test-set-based threshold calibration.
    """

    statistic: str = ""

    def __init__(
        self, *, n_components: int = 1, min_std: float = 1e-8, min_eigenvalue: float = 1e-10
    ) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if not np.isfinite(min_std) or min_std <= 0:
            raise ValueError("min_std must be finite and positive")
        if not np.isfinite(min_eigenvalue) or min_eigenvalue <= 0:
            raise ValueError("min_eigenvalue must be finite and positive")
        self.n_components = n_components
        self.min_std = float(min_std)
        self.min_eigenvalue = float(min_eigenvalue)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._names: tuple[str, ...] | None = None
        self._units: tuple[str, ...] | None = None
        self._rate: float | None = None
        self._reference: SignalBatch | None = None
        self._threshold: float | None = None

    def fit(self, reference: SignalBatch) -> "_PCABaseline":
        if not isinstance(reference, SignalBatch):
            raise TypeError("reference must be SignalBatch")
        if not np.isfinite(reference.values).all():
            raise ValueError("PCA reference must contain only finite observations")
        n, length, channels = reference.shape
        if channels < 2 or self.n_components >= channels or n * length <= channels:
            raise ValueError("PCA requires >=2 channels, components < channels and sufficient reference points")
        flattened = reference.values.reshape(-1, channels)
        mean = np.mean(flattened, axis=0)
        scale = np.maximum(np.std(flattened, axis=0, ddof=1), self.min_std)
        standardized = (flattened - mean) / scale
        if not np.isfinite(standardized).all():
            raise ValueError("PCA reference normalization overflowed")
        covariance = standardized.T @ standardized / (flattened.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        order = np.argsort(eigvals)[::-1]
        chosen = eigvals[order[: self.n_components]]
        if not np.isfinite(chosen).all() or np.any(chosen <= self.min_eigenvalue):
            raise ValueError("retained principal components have insufficient nonzero variance")
        components = eigvecs[:, order[: self.n_components]]
        for value in (mean, scale, chosen, components):
            value.setflags(write=False)
        self._mean = mean
        self._std = scale
        self._eigenvalues = chosen
        self._components = components
        self._names = reference.channel_names
        self._units = reference.units
        self._rate = reference.sample_rate_hz
        self._reference = reference
        self._threshold = None
        return self

    @property
    def threshold_(self) -> float:
        if self._threshold is None:
            raise RuntimeError("calibrate on independent development data before evaluating alarms")
        return self._threshold

    @property
    def eigenvalues_(self) -> np.ndarray:
        if self._eigenvalues is None:
            raise RuntimeError("PCA must be fitted before inspecting principal components")
        return self._eigenvalues.copy()

    def _scores(self, batch: SignalBatch) -> tuple[np.ndarray, np.ndarray]:
        if self._mean is None or self._std is None or self._components is None or self._eigenvalues is None:
            raise RuntimeError("PCA must be fitted on a healthy baseline before scoring")
        if not isinstance(batch, SignalBatch):
            raise TypeError("batch must be SignalBatch")
        if not np.isfinite(batch.values).all():
            raise ValueError("PCA inference requires finite observations")
        if batch.channel_names != self._names:
            raise ValueError("channel_names or order differs from PCA reference")
        if batch.units != self._units:
            raise ValueError("units differ from PCA reference")
        if batch.sample_rate_hz != self._rate:
            raise ValueError("sample_rate_hz differs from PCA reference")
        normalized = (batch.values - self._mean) / self._std
        coordinates = normalized @ self._components
        t2 = np.sum(coordinates**2 / self._eigenvalues, axis=-1)
        residual = normalized - coordinates @ self._components.T
        spe = np.sum(residual**2, axis=-1)
        return t2, spe

    def _statistic_scores(self, batch: SignalBatch) -> np.ndarray:
        t2, spe = self._scores(batch)
        return t2 if self.statistic == "t2" else spe

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        return DiagnosticResult(
            task="anomaly_detection",
            output_kind="anomaly_score",
            values=self._statistic_scores(batch),
            algorithm_name=self.__class__.__name__,
            threshold=self._threshold,
        )

    def calibrate(self, development: SignalBatch, *, quantile: float = 0.99) -> "_PCABaseline":
        """Determine empirical alarm limit from separate development data only."""
        if not np.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be finite and strictly between zero and one")
        if development is self._reference:
            raise ValueError("calibration must use independent development data, not the fitted object")
        scores = self._statistic_scores(development)
        if scores.size < 2:
            raise ValueError("calibration requires at least two development observations")
        limit = float(np.quantile(scores, quantile))
        if not np.isfinite(limit):
            raise ValueError("calibrated limit is nonfinite")
        self._threshold = limit
        return self

    def alarm_mask(self, batch: SignalBatch) -> np.ndarray:
        limit = self.threshold_
        return self._statistic_scores(batch) > limit


class PCAT2Detector(_PCABaseline):
    """Hotelling T-squared statistic on retained principal subspace."""

    statistic = "t2"


class PCASPEDetector(_PCABaseline):
    """Squared prediction error (Q/SPE) in residual subspace."""

    statistic = "spe"


def register_pca_detectors(registry=None) -> None:
    """Explicit factory admission; keep the global registry empty by default."""
    from torch_like.diagnostics.registry import diagnostic_registry

    target = diagnostic_registry if registry is None else registry
    for name, factory in (("PCAT2Detector", PCAT2Detector), ("PCASPEDetector", PCASPEDetector)):
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
