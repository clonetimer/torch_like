"""Numerical references and leakage checks for statistical diagnostic baselines."""

import unittest

import numpy as np

from torch_like.diagnostics import (
    CUSUMDetector,
    DiagnosticRegistry,
    EWMADetector,
    SignalBatch,
    ZScoreDetector,
    register_statistical_detectors,
)


def healthy_reference():
    return SignalBatch.from_sequence(
        np.array([[-1.0], [1.0], [-1.0], [1.0]]),
        sample_rate_hz=100,
        channel_names=("x",),
    )


def signal(sequence, *, rate=100, names=("x",)):
    return SignalBatch.from_sequence(
        np.asarray(sequence, dtype=np.float64).reshape(-1, 1),
        sample_rate_hz=rate,
        channel_names=names,
    )


class StatisticalDetectorsTests(unittest.TestCase):
    def test_zscore_matches_hand_calculation(self):
        model = ZScoreDetector(sigma_threshold=2.5).fit(healthy_reference())
        scores = model.score(signal([0, 2, -3, 0]))
        np.testing.assert_allclose(scores.values.reshape(-1), [0, 2, 3, 0])
        self.assertEqual(scores.threshold, 2.5)
        self.assertEqual(scores.algorithm_name, "ZScoreDetector")
        np.testing.assert_array_equal(model.alarm_mask(signal([0, 2, -3, 0])).reshape(-1), [0, 0, 1, 0])

    def test_cusum_two_sided_reference(self):
        model = CUSUMDetector(k=0.5, h=2.5).fit(healthy_reference())
        scores = model.score(signal([0, 2, 2, -2]))
        np.testing.assert_allclose(scores.values.reshape(-1), [0, 1.5, 3, 1.5])
        np.testing.assert_array_equal(model.alarm_mask(signal([0, 2, 2, -2])).reshape(-1), [0, 0, 1, 0])

    def test_ewma_uses_finite_horizon_variance(self):
        model = EWMADetector(alpha=0.5, control_limit=3.0).fit(healthy_reference())
        result = model.score(signal([2, 2, 2]))
        np.testing.assert_allclose(
            result.values.reshape(-1),
            [2.0, 2.6832815729997477, 3.0550504633038935],
            rtol=1e-12,
        )
        np.testing.assert_array_equal(model.alarm_mask(signal([2, 2, 2])).reshape(-1), [0, 0, 1])

    def test_window_state_isolated_between_samples(self):
        x = signal([0, 2, 2]).values
        windows = SignalBatch(np.concatenate([x, x], axis=0), sample_rate_hz=100, channel_names=("x",))
        for detector in (CUSUMDetector(), EWMADetector()):
            with self.subTest(name=detector.__class__.__name__):
                detector.fit(healthy_reference())
                scores = detector.score(windows).values
                np.testing.assert_array_equal(scores[0], scores[1])

    def test_scoring_does_not_update_reference(self):
        detector = ZScoreDetector().fit(healthy_reference())
        original_mean = detector.mean_
        original_std = detector.std_
        detector.score(signal([1000, 2000]))
        np.testing.assert_array_equal(detector.mean_, original_mean)
        np.testing.assert_array_equal(detector.std_, original_std)
        original_mean[:] = 100
        self.assertEqual(float(detector.mean_[0]), 0.0)

    def test_mismatched_sampling_and_channel_order_are_rejected(self):
        detector = ZScoreDetector().fit(healthy_reference())
        with self.assertRaisesRegex(ValueError, "sample_rate_hz"):
            detector.score(signal([1, 2], rate=50))
        with self.assertRaisesRegex(ValueError, "channel_names"):
            detector.score(signal([1, 2], names=("other",)))
        with self.assertRaisesRegex(ValueError, "units"):
            detector.score(
                SignalBatch.from_sequence(
                    np.array([[1.0], [2.0]]),
                    sample_rate_hz=100,
                    channel_names=("x",),
                    units=("g",),
                )
            )

    def test_fitted_state_and_development_only_guardrails(self):
        detector = ZScoreDetector()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            detector.score(signal([1]))
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            _ = detector.mean_
        with self.assertRaisesRegex(ValueError, "at least two"):
            detector.fit(signal([1]))
        with self.assertRaisesRegex(ValueError, "finite"):
            detector.fit(signal([np.nan]))
        with self.assertRaisesRegex(ValueError, "finite"):
            detector.fit(signal([np.nan, 1]))

    def test_nan_inference_is_rejected_even_with_explicit_allow_missing(self):
        detector = ZScoreDetector().fit(healthy_reference())
        x = SignalBatch.from_sequence(np.array([[np.nan], [1]]), sample_rate_hz=100, allow_missing=True)
        with self.assertRaisesRegex(ValueError, "finite"):
            detector.score(x)

    def test_multichannel_baseline_keeps_channel_specific_statistics(self):
        base = SignalBatch.from_sequence(
            np.array([[-1, 20], [1, 40], [-1, 20], [1, 40]], dtype=np.float64),
            sample_rate_hz=10,
            channel_names=("vibration", "temperature"),
        )
        test = SignalBatch.from_sequence(
            np.array([[2, 60]], dtype=np.float64),
            sample_rate_hz=10,
            channel_names=("vibration", "temperature"),
        )
        result = ZScoreDetector().fit(base).score(test)
        np.testing.assert_allclose(result.values[0, 0], [2, 3])

    def test_constant_baseline_scale_is_floored(self):
        fixed = signal([4, 4, 4, 4])
        detector = ZScoreDetector(min_std=0.5).fit(fixed)
        self.assertEqual(float(detector.std_[0]), 0.5)
        np.testing.assert_allclose(detector.score(signal([5])).values.reshape(-1), [2])

    def test_invalid_detector_hyperparameters(self):
        for value in (0, -1, np.inf, np.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ZScoreDetector(sigma_threshold=value)
                with self.assertRaises(ValueError):
                    ZScoreDetector(min_std=value)
                with self.assertRaises(ValueError):
                    EWMADetector(control_limit=value)
                with self.assertRaises(ValueError):
                    CUSUMDetector(h=value)
        for alpha in (0.0, -1.0, 1.01, np.nan):
            with self.assertRaises(ValueError):
                EWMADetector(alpha=alpha)
        with self.assertRaises(ValueError):
            CUSUMDetector(k=-0.1)

    def test_explicit_registration_returns_real_detector_objects(self):
        registry = DiagnosticRegistry()
        self.assertFalse(registry.list_specs())
        register_statistical_detectors(registry)
        self.assertEqual(len(registry.list_specs(task="anomaly_detection")), 3)
        model = registry.create("CUSUMDetector", k=1.0, h=3.0)
        self.assertIsInstance(model, CUSUMDetector)
        self.assertEqual(registry.get_spec("CUSUMDetector").operations, ("fit", "score"))
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_statistical_detectors(registry)


if __name__ == "__main__":
    unittest.main()
