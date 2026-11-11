"""SINDy sparse dynamics identification and residual-monitor tests."""

import unittest

import numpy as np

from torch_like.diagnostics import DiagnosticRegistry, SignalBatch, register_sindy_detector
from torch_like.diagnostics.sindy import SINDyDetector


def exponential_batch(initial_values, *, rate=100.0, decay=0.5, names=("omega",), ids=None):
    time = np.arange(601, dtype=np.float64) / rate
    values = np.stack([initial * np.exp(-decay * time) for initial in initial_values], axis=0)[..., None]
    return SignalBatch(
        values,
        sample_rate_hz=rate,
        channel_names=names,
        units=("rad/s",),
        sample_ids=None if ids is None else tuple(ids),
    )


class SINDyDetectorTests(unittest.TestCase):
    def test_recovers_sparse_linear_dynamics_across_independent_trajectories(self):
        reference = exponential_batch([0.5, 1.0, 2.0], ids=("r0", "r1", "r2"))
        model = SINDyDetector(
            polynomial_degree=1,
            sparsity_threshold=0.01,
            ridge_alpha=1e-10,
            include_bias=False,
        ).fit(reference)

        self.assertEqual(model.feature_names_, ("omega",))
        self.assertEqual(model.coefficients_.shape, (1, 1))
        self.assertAlmostEqual(float(model.coefficients_[0, 0]), -0.5, places=4)
        self.assertIn("d(omega)/dt", model.equations_[0])
        self.assertIn("omega", model.equations_[0])

        predicted = model.predict_derivative(reference.values)
        expected = -0.5 * reference.values
        np.testing.assert_allclose(predicted, expected, atol=2e-4, rtol=2e-4)

    def test_controlled_dynamics_include_control_terms(self):
        rate = 100.0
        time = np.arange(801, dtype=np.float64) / rate
        initial = (1.0, 1.5, 2.0)
        states = np.stack(
            [2.5 + (value - 2.5) * np.exp(-0.2 * time) for value in initial],
            axis=0,
        )[..., None]
        controls = np.ones((len(initial), len(time), 1), dtype=np.float64)
        reference = SignalBatch(
            states,
            sample_rate_hz=rate,
            channel_names=("current",),
            units=("A",),
        )
        model = SINDyDetector(
            polynomial_degree=1,
            sparsity_threshold=0.01,
            ridge_alpha=1e-10,
            include_bias=False,
        ).fit(reference, controls=controls)

        self.assertEqual(model.feature_names_, ("current", "u0"))
        coefficients = dict(zip(model.feature_names_, model.coefficients_[:, 0], strict=True))
        self.assertAlmostEqual(float(coefficients["current"]), -0.2, places=3)
        self.assertAlmostEqual(float(coefficients["u0"]), 0.5, places=3)
        with self.assertRaisesRegex(ValueError, "controls are required"):
            model.predict_derivative(states)

    def test_independent_healthy_calibration_and_fault_residual(self):
        reference = exponential_batch([0.5, 1.0, 2.0], ids=("fit-0", "fit-1", "fit-2"))
        development = exponential_batch([0.7, 1.4, 2.5], ids=("dev-0", "dev-1", "dev-2"))
        anomalous = exponential_batch([0.7, 1.4, 2.5], decay=1.0, ids=("test-0", "test-1", "test-2"))
        model = SINDyDetector(
            polynomial_degree=1,
            sparsity_threshold=0.01,
            ridge_alpha=1e-10,
            include_bias=False,
        ).fit(reference)

        self.assertIsNone(model.score(development).threshold)
        model.calibrate(development, quantile=0.95)
        self.assertGreaterEqual(model.threshold_, 0.0)
        healthy_scores = model.score(development).values
        fault_scores = model.score(anomalous).values
        self.assertEqual(healthy_scores.shape, (3,))
        self.assertTrue(np.all(fault_scores > model.threshold_))
        np.testing.assert_array_equal(model.alarm_mask(anomalous), [True, True, True])

    def test_fit_and_score_keep_trajectory_boundaries_and_metadata(self):
        reference = exponential_batch([0.5, 1.0], ids=("a", "b"))
        model = SINDyDetector(polynomial_degree=1, include_bias=False).fit(reference)
        repeated = SignalBatch(
            np.concatenate((reference.values[:1], reference.values[:1]), axis=0),
            sample_rate_hz=100,
            channel_names=("omega",),
            units=("rad/s",),
        )
        scores = model.score(repeated).values
        self.assertEqual(scores.shape, (2,))
        self.assertAlmostEqual(float(scores[0]), float(scores[1]), places=12)

        wrong_rate = SignalBatch(
            reference.values,
            sample_rate_hz=50,
            channel_names=("omega",),
            units=("rad/s",),
        )
        with self.assertRaisesRegex(ValueError, "sample_rate_hz"):
            model.score(wrong_rate)
        wrong_units = SignalBatch(
            reference.values,
            sample_rate_hz=100,
            channel_names=("omega",),
            units=("deg/s",),
        )
        with self.assertRaisesRegex(ValueError, "units"):
            model.score(wrong_units)

    def test_rejects_short_missing_irregular_and_mismatched_inputs(self):
        short = SignalBatch.from_sequence(
            np.array([[1.0], [0.9]]),
            sample_rate_hz=100,
            channel_names=("omega",),
        )
        with self.assertRaisesRegex(ValueError, "three time steps"):
            SINDyDetector().fit(short)

        missing = SignalBatch(
            np.array([[[1.0], [0.9], [0.8], [0.7]]]),
            sample_rate_hz=100,
            channel_names=("omega",),
            allow_missing=True,
        )
        values = np.array(missing.values, copy=True)
        values[0, 1, 0] = np.nan
        missing = SignalBatch(values, sample_rate_hz=100, channel_names=("omega",), allow_missing=True)
        with self.assertRaisesRegex(ValueError, "finite"):
            SINDyDetector().fit(missing)

        irregular = SignalBatch(
            np.array([[[1.0], [0.9], [0.8], [0.7]]]),
            sample_rate_hz=100,
            channel_names=("omega",),
            timestamps_s=np.array([[0.0, 0.01, 0.025, 0.03]]),
        )
        with self.assertRaisesRegex(ValueError, "uniformly sampled"):
            SINDyDetector().fit(irregular)

        model = SINDyDetector(polynomial_degree=1).fit(exponential_batch([1.0]))
        with self.assertRaisesRegex(ValueError, "controls must have shape"):
            model.score(exponential_batch([1.0]), controls=np.ones((1, 600, 1)))

    def test_calibration_rejects_reference_reuse_and_overlapping_sample_ids(self):
        reference = exponential_batch([1.0, 2.0], ids=("shared", "fit-only"))
        model = SINDyDetector(polynomial_degree=1, include_bias=False).fit(reference)
        with self.assertRaisesRegex(ValueError, "independent"):
            model.calibrate(reference)
        overlap = exponential_batch([0.7, 1.4], ids=("shared", "dev-only"))
        with self.assertRaisesRegex(ValueError, "sample_ids overlap"):
            model.calibrate(overlap)

    def test_registry_registration_and_parameter_validation(self):
        registry = DiagnosticRegistry()
        register_sindy_detector(registry)
        spec = registry.get_spec("SINDyDetector")
        self.assertEqual(spec.operations, ("fit", "score"))
        self.assertEqual(spec.task, "anomaly_detection")
        created = registry.create(
            "SINDyDetector",
            polynomial_degree=1,
            include_bias=False,
        )
        self.assertIsInstance(created, SINDyDetector)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_sindy_detector(registry)
        with self.assertRaisesRegex(ValueError, "unknown diagnostic parameters"):
            registry.create("SINDyDetector", unknown_option=True)


if __name__ == "__main__":
    unittest.main()
