"""Sparse Identification of Nonlinear Dynamics (SINDy) for telemetry dynamics.

The detector fits sparse state-derivative equations on caller-selected healthy
trajectories. Its anomaly score is the per-trajectory RMS derivative residual.
It is a classical numerical estimator, not a neural network.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb
from typing import Any

import numpy as np

from torch_like.diagnostics.types import DiagnosticAlgorithmSpec, DiagnosticResult, SignalBatch


class SINDyDetector:
    """Fit sparse nonlinear dynamics and score departures from the fitted model.

    Inputs use SignalBatch shape (samples, time, state_channels). Each sample is
    treated as an independent trajectory; derivatives never cross sample
    boundaries. Optional controls have shape (samples, time, control_channels).

    The sparse threshold is applied after column normalization, following the
    sequentially thresholded least-squares (STLSQ) procedure.
    """

    _MAX_LIBRARY_TERMS = 4096

    def __init__(
        self,
        *,
        polynomial_degree: int = 2,
        sparsity_threshold: float = 0.05,
        ridge_alpha: float = 1e-8,
        max_iter: int = 20,
        include_bias: bool = True,
        include_sine: bool = False,
    ) -> None:
        if isinstance(polynomial_degree, bool) or not isinstance(polynomial_degree, int):
            raise ValueError("polynomial_degree must be an integer in [1, 4]")
        if not 1 <= polynomial_degree <= 4:
            raise ValueError("polynomial_degree must be an integer in [1, 4]")
        if not np.isfinite(sparsity_threshold) or sparsity_threshold < 0:
            raise ValueError("sparsity_threshold must be finite and nonnegative")
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
            raise ValueError("ridge_alpha must be finite and nonnegative")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(include_bias, bool) or not isinstance(include_sine, bool):
            raise ValueError("include_bias and include_sine must be bool values")

        self.polynomial_degree = polynomial_degree
        self.sparsity_threshold = float(sparsity_threshold)
        self.ridge_alpha = float(ridge_alpha)
        self.max_iter = max_iter
        self.include_bias = include_bias
        self.include_sine = include_sine

        self._coefficients: np.ndarray | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._channel_names: tuple[str, ...] | None = None
        self._units: tuple[str, ...] | None = None
        self._sample_rate_hz: float | None = None
        self._control_dim = 0
        self._reference: SignalBatch | None = None
        self._threshold: float | None = None

    @staticmethod
    def _prepare_controls(
        controls: Any,
        leading_shape: tuple[int, ...],
        *,
        expected_dim: int | None = None,
    ) -> np.ndarray | None:
        if controls is None:
            if expected_dim not in (None, 0):
                raise ValueError("controls are required because the fitted model uses control inputs")
            return None
        raw = np.asarray(controls)
        if np.iscomplexobj(raw):
            raise ValueError("complex-valued controls require an explicit real-valued representation")
        values = np.asarray(raw, dtype=np.float64)
        if (
            values.ndim != len(leading_shape) + 1
            or values.shape[:-1] != leading_shape
            or values.shape[-1] <= 0
        ):
            raise ValueError(f"controls must have shape {leading_shape} + (control_channels,)")
        if expected_dim is not None and values.shape[-1] != expected_dim:
            raise ValueError(f"expected {expected_dim} control channels, got {values.shape[-1]}")
        if not np.isfinite(values).all():
            raise ValueError("controls must contain only finite values")
        return values

    def _library(
        self,
        states: np.ndarray,
        controls: np.ndarray | None,
        state_names: tuple[str, ...],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        variables = states if controls is None else np.concatenate((states, controls), axis=1)
        names = state_names + (
            () if controls is None else tuple(f"u{index}" for index in range(controls.shape[1]))
        )
        variable_count = variables.shape[1]
        term_count = int(self.include_bias)
        term_count += sum(comb(variable_count + degree - 1, degree) for degree in range(1, self.polynomial_degree + 1))
        term_count += 2 * variable_count if self.include_sine else 0
        if term_count > self._MAX_LIBRARY_TERMS:
            raise ValueError(
                f"candidate library has {term_count} terms; reduce polynomial_degree or input channels "
                f"(limit {self._MAX_LIBRARY_TERMS})"
            )

        columns: list[np.ndarray] = []
        feature_names: list[str] = []
        if self.include_bias:
            columns.append(np.ones(states.shape[0], dtype=np.float64))
            feature_names.append("1")
        for degree in range(1, self.polynomial_degree + 1):
            for indices in combinations_with_replacement(range(variable_count), degree):
                columns.append(np.prod(variables[:, indices], axis=1))
                feature_names.append("*".join(names[index] for index in indices))
        if self.include_sine:
            for index, name in enumerate(names):
                columns.extend((np.sin(variables[:, index]), np.cos(variables[:, index])))
                feature_names.extend((f"sin({name})", f"cos({name})"))

        library = np.column_stack(columns)
        if not np.isfinite(library).all():
            raise ValueError("candidate function library contains nonfinite values; rescale input signals")
        return library, tuple(feature_names)

    def _solve(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        if self.ridge_alpha == 0:
            return np.linalg.lstsq(design, target, rcond=None)[0]
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += self.ridge_alpha
        return np.linalg.solve(gram, design.T @ target)

    def _stlsq(self, library: np.ndarray, derivatives: np.ndarray) -> np.ndarray:
        scales = np.linalg.norm(library, axis=0)
        scales = np.where(scales > np.finfo(np.float64).eps, scales, 1.0)
        normalized = library / scales
        coefficients = self._solve(normalized, derivatives)

        for _ in range(self.max_iter):
            small = np.abs(coefficients) < self.sparsity_threshold
            if not small.any():
                break
            updated = coefficients.copy()
            updated[small] = 0.0
            for state_index in range(derivatives.shape[1]):
                active = ~small[:, state_index]
                if active.any():
                    updated[active, state_index] = self._solve(
                        normalized[:, active], derivatives[:, state_index]
                    )
            if np.array_equal(updated, coefficients):
                break
            coefficients = updated

        result = coefficients / scales[:, None]
        if not np.isfinite(result).all():
            raise ValueError("sparse regression produced nonfinite coefficients; rescale data or adjust ridge_alpha")
        return result

    def fit(self, reference: SignalBatch, controls: Any = None) -> "SINDyDetector":
        """Identify dynamics from caller-selected healthy reference trajectories."""
        if not isinstance(reference, SignalBatch):
            raise TypeError("reference must be SignalBatch")
        values = reference.values
        if not np.isfinite(values).all():
            raise ValueError("SINDy fitting requires finite reference signals")
        sample_count, time_steps, state_count = values.shape
        if time_steps < 3:
            raise ValueError("SINDy requires at least three time steps per trajectory")
        dt = 1.0 / reference.sample_rate_hz
        if reference.timestamps_s is not None:
            intervals = np.diff(reference.timestamps_s, axis=1)
            tolerance = max(1e-9, dt * 1e-6)
            if not np.allclose(intervals, dt, rtol=1e-4, atol=tolerance):
                raise ValueError("SINDy requires uniformly sampled timestamps consistent with sample_rate_hz")

        control_values = self._prepare_controls(controls, (sample_count, time_steps))
        control_dim = 0 if control_values is None else control_values.shape[-1]
        flat_states = values.reshape(-1, state_count)
        flat_controls = None if control_values is None else control_values.reshape(-1, control_dim)
        library, feature_names = self._library(flat_states, flat_controls, reference.channel_names)
        derivatives = np.gradient(values, dt, axis=1, edge_order=2).reshape(-1, state_count)
        if not np.isfinite(derivatives).all():
            raise ValueError("finite-difference derivatives are nonfinite; rescale or inspect the reference signals")
        coefficients = self._stlsq(library, derivatives)
        coefficients.setflags(write=False)

        self._coefficients = coefficients
        self._feature_names = feature_names
        self._channel_names = reference.channel_names
        self._units = reference.units
        self._sample_rate_hz = reference.sample_rate_hz
        self._control_dim = control_dim
        self._reference = reference
        self._threshold = None
        return self

    @property
    def coefficients_(self) -> np.ndarray:
        if self._coefficients is None:
            raise RuntimeError("fit SINDyDetector before inspecting coefficients")
        result = self._coefficients.copy()
        result.setflags(write=False)
        return result

    @property
    def feature_names_(self) -> tuple[str, ...]:
        if self._feature_names is None:
            raise RuntimeError("fit SINDyDetector before inspecting feature names")
        return self._feature_names

    @property
    def equations_(self) -> tuple[str, ...]:
        coefficients = self.coefficients_
        feature_names = self.feature_names_
        if self._channel_names is None:
            raise RuntimeError("fit SINDyDetector before inspecting equations")
        equations = []
        for state_index, state_name in enumerate(self._channel_names):
            terms = [
                f"{coefficient:+.6g}*{feature_name}"
                for feature_name, coefficient in zip(
                    feature_names, coefficients[:, state_index], strict=True
                )
                if abs(float(coefficient)) > 1e-10
            ]
            equations.append(f"d({state_name})/dt = {' '.join(terms) if terms else '0'}")
        return tuple(equations)

    @property
    def threshold_(self) -> float:
        if self._threshold is None:
            raise RuntimeError("calibrate on independent healthy development data before generating alarms")
        return self._threshold

    def _check_batch(self, batch: SignalBatch) -> None:
        if self._coefficients is None or self._channel_names is None:
            raise RuntimeError("fit SINDyDetector on healthy reference trajectories before scoring")
        if not isinstance(batch, SignalBatch):
            raise TypeError("batch must be SignalBatch")
        if not np.isfinite(batch.values).all():
            raise ValueError("SINDy scoring requires finite signals")
        if batch.channel_names != self._channel_names:
            raise ValueError("channel_names and channel ordering differ from the fitted reference")
        if batch.units != self._units:
            raise ValueError("units differ from the fitted reference")
        if batch.sample_rate_hz != self._sample_rate_hz:
            raise ValueError("sample_rate_hz differs from the fitted reference")
        if batch.shape[1] < 3:
            raise ValueError("SINDy scoring requires at least three time steps per trajectory")
        if batch.timestamps_s is not None:
            dt = 1.0 / batch.sample_rate_hz
            intervals = np.diff(batch.timestamps_s, axis=1)
            if not np.allclose(intervals, dt, rtol=1e-4, atol=max(1e-9, dt * 1e-6)):
                raise ValueError("SINDy requires uniformly sampled timestamps consistent with sample_rate_hz")

    def predict_derivative(self, states: Any, controls: Any = None) -> np.ndarray:
        """Evaluate the identified right-hand side at state/control observations."""
        if self._coefficients is None or self._channel_names is None:
            raise RuntimeError("fit SINDyDetector before predicting derivatives")
        raw = np.asarray(states)
        if np.iscomplexobj(raw):
            raise ValueError("complex-valued states require an explicit real-valued representation")
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim < 2 or values.shape[-1] != len(self._channel_names) or not np.isfinite(values).all():
            raise ValueError("states must be finite with shape (..., time_or_batch, fitted_state_channels)")
        control_values = self._prepare_controls(
            controls, values.shape[:-1], expected_dim=self._control_dim
        )
        flat_states = values.reshape(-1, values.shape[-1])
        flat_controls = None if control_values is None else control_values.reshape(-1, self._control_dim)
        library, feature_names = self._library(flat_states, flat_controls, self._channel_names)
        if feature_names != self._feature_names:
            raise RuntimeError("candidate library changed after fitting")
        prediction = library @ self._coefficients
        return prediction.reshape(values.shape)

    def residuals(self, batch: SignalBatch, controls: Any = None) -> np.ndarray:
        """Return derivative residuals with shape (samples, time, state_channels)."""
        self._check_batch(batch)
        control_values = self._prepare_controls(controls, batch.shape[:2], expected_dim=self._control_dim)
        dt = 1.0 / batch.sample_rate_hz
        observed = np.gradient(batch.values, dt, axis=1, edge_order=2)
        predicted = self.predict_derivative(batch.values, control_values)
        residuals = observed - predicted
        if not np.isfinite(residuals).all():
            raise ValueError("SINDy residuals are nonfinite")
        residuals.setflags(write=False)
        return residuals

    def calibrate(
        self,
        development: SignalBatch,
        controls: Any = None,
        *,
        quantile: float = 0.99,
    ) -> "SINDyDetector":
        """Set an empirical alarm threshold from independent healthy development data."""
        if not np.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be finite and strictly between zero and one")
        if development is self._reference:
            raise ValueError("calibration must use independent development data, not the fitted reference object")
        if (
            self._reference is not None
            and self._reference.sample_ids is not None
            and development.sample_ids is not None
            and set(self._reference.sample_ids).intersection(development.sample_ids)
        ):
            raise ValueError("calibration sample_ids overlap the fitted reference")
        residuals = self.residuals(development, controls)
        scores = np.sqrt(np.mean(residuals**2, axis=(1, 2)))
        if scores.size < 2:
            raise ValueError("calibration requires at least two independent development trajectories")
        threshold = float(np.quantile(scores, quantile))
        if not np.isfinite(threshold):
            raise ValueError("calibrated SINDy threshold is nonfinite")
        self._threshold = threshold
        return self

    def score(self, batch: SignalBatch, controls: Any = None) -> DiagnosticResult:
        """Return one RMS dynamics-residual anomaly score per trajectory."""
        residuals = self.residuals(batch, controls)
        scores = np.sqrt(np.mean(residuals**2, axis=(1, 2)))
        return DiagnosticResult(
            task="anomaly_detection",
            output_kind="anomaly_score",
            values=scores,
            algorithm_name="SINDyDetector",
            threshold=self._threshold,
        )

    def alarm_mask(self, batch: SignalBatch, controls: Any = None) -> np.ndarray:
        """Compare scores with a threshold fitted on independent healthy data."""
        return self.score(batch, controls).values > self.threshold_


def register_sindy_detector(registry: Any = None) -> None:
    """Explicitly register SINDy; importing this module has no global side effects."""
    from torch_like.diagnostics.registry import diagnostic_registry
    from torch_like.diagnostics.types import DiagnosticParameterSpec

    target = diagnostic_registry if registry is None else registry
    parameters = (
        DiagnosticParameterSpec(
            "polynomial_degree",
            "integer",
            default=2,
            minimum=1,
            maximum=4,
            description="Maximum polynomial degree in the candidate dynamics library.",
        ),
        DiagnosticParameterSpec(
            "sparsity_threshold",
            "number",
            default=0.05,
            minimum=0.0,
            description="STLSQ coefficient threshold after candidate-column normalization.",
        ),
        DiagnosticParameterSpec(
            "ridge_alpha",
            "number",
            default=1e-8,
            minimum=0.0,
            description="Ridge stabilization used during sparse least-squares refits.",
        ),
        DiagnosticParameterSpec(
            "max_iter",
            "integer",
            default=20,
            minimum=1,
            maximum=1000,
            description="Maximum sequential threshold/refit iterations.",
        ),
        DiagnosticParameterSpec(
            "include_bias",
            "boolean",
            default=True,
            description="Include a constant candidate term.",
        ),
        DiagnosticParameterSpec(
            "include_sine",
            "boolean",
            default=False,
            description="Include sine and cosine candidates for state and control variables.",
        ),
    )
    target.register(
        DiagnosticAlgorithmSpec(
            name="SINDyDetector",
            backend="statistical",
            task="anomaly_detection",
            output_kind="anomaly_score",
            operations=("fit", "score"),
            description=(
                "Sparse nonlinear dynamics identification with STLSQ; scores healthy-model "
                "derivative residuals. Requires independent threshold calibration for alarms."
            ),
            parameters=parameters,
        ),
        SINDyDetector,
    )


__all__ = ["SINDyDetector", "register_sindy_detector"]
