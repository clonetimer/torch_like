"""Explicit feature-to-sklearn anomaly detection, with train/test separation.

The adapter preserves the native scikit-learn estimator's fit/decision_function
API and uses the shared SignalBatch and DiagnosticResult domain contracts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from torch_like.diagnostics.features import time_domain_features
from torch_like.diagnostics.types import DiagnosticAlgorithmSpec, DiagnosticResult, SignalBatch

DEFAULT_WINDOW_FEATURES = ("mean", "std", "rms", "peak", "crest_factor", "skewness", "kurtosis")


class WindowFeatureTransformer:
    """Stateless channel-wise window statistics flattened into a feature matrix."""

    def __init__(self, *, features: tuple[str, ...] = DEFAULT_WINDOW_FEATURES) -> None:
        selected = tuple(features)
        if (
            not selected
            or len(selected) != len(set(selected))
            or any(name not in DEFAULT_WINDOW_FEATURES for name in selected)
        ):
            raise ValueError("features must be nonempty, unique recognized names")
        self.features = selected

    def transform(self, batch: SignalBatch) -> np.ndarray:
        """Return (samples, channels * features), channel-major in declared order."""
        statistics = time_domain_features(batch)
        feature_cube = np.stack([statistics[name] for name in self.features], axis=-1)
        feature_matrix = feature_cube.reshape(batch.shape[0], -1)
        if not np.isfinite(feature_matrix).all():
            raise ValueError("window feature values must be finite; scale input data upstream")
        return np.array(feature_matrix, copy=True)

    def column_names(self, channel_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(f"{channel}__{feature}" for channel in channel_names for feature in self.features)


class SklearnWindowAnomalyDetector:
    """Wrap a native sklearn decision-function estimator without modifying it.

    A positive output is more anomalous: -estimator.decision_function(X).
    For sklearn IsolationForest, threshold zero corresponds to its native
    outlier decision boundary. The caller provides train-only baseline windows.
    """

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        features: tuple[str, ...] = DEFAULT_WINDOW_FEATURES,
        **isolation_forest_kwargs: Any,
    ) -> None:
        if estimator is not None and isolation_forest_kwargs:
            raise ValueError("cannot pass both an estimator and IsolationForest keyword arguments")
        if estimator is None:
            from torch_like.classical import create_estimator

            options = {"n_estimators": 100, "contamination": "auto", "random_state": 0}
            options.update(isolation_forest_kwargs)
            estimator = create_estimator("IsolationForest", **options)
        if not callable(getattr(estimator, "fit", None)) or not callable(getattr(estimator, "decision_function", None)):
            raise TypeError("estimator must expose native fit and decision_function")
        self.estimator = estimator
        self.transformer = WindowFeatureTransformer(features=features)
        self._channel_names: tuple[str, ...] | None = None
        self._units: tuple[str, ...] | None = None
        self._sample_rate_hz: float | None = None
        self._window_length: int | None = None
        self._fitted = False

    def fit(self, reference: SignalBatch) -> "SklearnWindowAnomalyDetector":
        """Fit the native estimator on caller-supplied healthy development windows."""
        if not isinstance(reference, SignalBatch):
            raise TypeError("reference must be a SignalBatch")
        if reference.shape[0] < 2:
            raise ValueError("fit requires at least two development windows")
        features = self.transformer.transform(reference)
        self.estimator.fit(features)
        self._channel_names = reference.channel_names
        self._units = reference.units
        self._sample_rate_hz = reference.sample_rate_hz
        self._window_length = reference.shape[1]
        self._fitted = True
        return self

    @property
    def feature_names_(self) -> tuple[str, ...]:
        if not self._fitted or self._channel_names is None:
            raise RuntimeError("fit on development windows before requesting feature names")
        return self.transformer.column_names(self._channel_names)

    def transform(self, batch: SignalBatch) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit a reference before transforming inference windows")
        if not isinstance(batch, SignalBatch):
            raise TypeError("batch must be a SignalBatch")
        if batch.channel_names != self._channel_names:
            raise ValueError("channel_names and their order must match development reference")
        if batch.units != self._units:
            raise ValueError("units must match development reference")
        if batch.sample_rate_hz != self._sample_rate_hz:
            raise ValueError("sample_rate_hz must match development reference")
        if batch.shape[1] != self._window_length:
            raise ValueError("window length must match the development reference")
        return self.transformer.transform(batch)

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        """Produce label-free per-window scores without refitting or calibration."""
        matrix = self.transform(batch)
        decisions = np.asarray(self.estimator.decision_function(matrix), dtype=np.float64)
        if decisions.shape != (batch.shape[0],) or not np.isfinite(decisions).all():
            raise ValueError("estimator decision_function must return finite (samples,) values")
        return DiagnosticResult(
            task="anomaly_detection",
            output_kind="anomaly_score",
            values=-decisions,
            algorithm_name="SklearnWindowAnomalyDetector",
            threshold=0.0,
        )

    def alarm_mask(self, batch: SignalBatch) -> np.ndarray:
        return self.score(batch).values > 0.0


def register_sklearn_window_detectors(registry=None) -> None:
    """Explicitly admit the native sklearn IsolationForest-backed detector."""
    from torch_like.diagnostics.registry import diagnostic_registry

    target = diagnostic_registry if registry is None else registry
    target.register(
        DiagnosticAlgorithmSpec(
            name="SklearnWindowAnomalyDetector",
            backend="sklearn",
            task="anomaly_detection",
            output_kind="anomaly_score",
            operations=("fit", "score"),
            input_kind="signal_batch",
        ),
        SklearnWindowAnomalyDetector,
    )
