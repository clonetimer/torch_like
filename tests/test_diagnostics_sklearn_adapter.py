"""Window feature matrix and native sklearn decision_function integration checks."""

import unittest

import numpy as np

from torch_like.diagnostics import DiagnosticRegistry, SignalBatch
from torch_like.diagnostics.sklearn_adapter import (
    SklearnWindowAnomalyDetector,
    WindowFeatureTransformer,
    register_sklearn_window_detectors,
)


def batch(values, *, rate=100.0, channels=("motor",), units=("g",)):
    return SignalBatch(np.asarray(values, dtype=float), sample_rate_hz=rate, channel_names=channels, units=units)


class RecordingEstimator:
    def __init__(self):
        self.fit_calls = 0
        self.decision_calls = 0
        self.fit_matrix = None

    def fit(self, matrix):
        self.fit_calls += 1
        self.fit_matrix = np.array(matrix, copy=True)
        return self

    def decision_function(self, matrix):
        self.decision_calls += 1
        return 0.5 - np.asarray(matrix)[:, 0]


class WindowAdapterTests(unittest.TestCase):
    def test_channel_major_feature_vector_values(self):
        signals = batch(np.array([[[-1, 2], [1, 2], [-1, 2], [1, 2]]]), channels=("a", "b"), units=("g", "A"))
        transformer = WindowFeatureTransformer(features=("mean", "rms", "kurtosis"))
        matrix = transformer.transform(signals)
        self.assertEqual(matrix.shape, (1, 6))
        np.testing.assert_allclose(matrix[0], [0, 1, 1, 2, 2, 0])
        self.assertEqual(transformer.column_names(signals.channel_names), (
            "a__mean", "a__rms", "a__kurtosis", "b__mean", "b__rms", "b__kurtosis"
        ))

    def test_injected_native_estimator_no_refit_on_evaluation(self):
        estimator = RecordingEstimator()
        detector = SklearnWindowAnomalyDetector(estimator, features=("mean",))
        development = batch(np.array([[[0], [1]], [[-1], [0]]]))
        test = batch(np.array([[[3], [5]], [[0], [0]]]))
        detector.fit(development)
        training_features = estimator.fit_matrix.copy()
        output = detector.score(test)
        self.assertEqual(estimator.fit_calls, 1)
        self.assertEqual(estimator.decision_calls, 1)
        np.testing.assert_allclose(output.values, [3.5, -0.5])
        self.assertEqual(output.threshold, 0.0)
        self.assertEqual(output.sample_count, 2)
        np.testing.assert_array_equal(detector.alarm_mask(test), [True, False])
        np.testing.assert_array_equal(estimator.fit_matrix, training_features)
        self.assertEqual(estimator.fit_calls, 1)

    def test_real_isolation_forest_native_score_orientation(self):
        rng = np.random.default_rng(123)
        development = batch(rng.normal(size=(12, 16, 1)))
        test = batch(np.concatenate([rng.normal(size=(3, 16, 1)), rng.normal(8, 0.3, (2, 16, 1))]))
        detector = SklearnWindowAnomalyDetector(n_estimators=12, random_state=123).fit(development)
        self.assertEqual(detector.estimator.__class__.__name__, "IsolationForest")
        result = detector.score(test)
        self.assertEqual(result.values.shape, (5,))
        self.assertTrue(np.isfinite(result.values).all())
        native = detector.estimator.decision_function(detector.transform(test))
        np.testing.assert_allclose(result.values, -native)
        np.testing.assert_array_equal(detector.alarm_mask(test), detector.estimator.predict(detector.transform(test)) == -1)
        self.assertEqual(len(detector.feature_names_), 7)

    def test_reject_channel_units_rate_and_window_length_drift(self):
        detector = SklearnWindowAnomalyDetector(RecordingEstimator(), features=("rms",))
        development = batch(np.ones((3, 4, 1)))
        detector.fit(development)
        cases = [
            (batch(np.ones((2, 4, 1)), channels=("other",)), "channel_names"),
            (batch(np.ones((2, 4, 1)), units=("m/s^2",)), "units"),
            (batch(np.ones((2, 4, 1)), rate=50), "sample_rate_hz"),
            (batch(np.ones((2, 5, 1))), "window length"),
        ]
        for example, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                detector.score(example)

    def test_requires_fit_and_multiple_development_windows(self):
        detector = SklearnWindowAnomalyDetector(RecordingEstimator())
        with self.assertRaisesRegex(RuntimeError, "fit"):
            detector.score(batch(np.ones((2, 4, 1))))
        with self.assertRaisesRegex(RuntimeError, "fit"):
            _ = detector.feature_names_
        with self.assertRaisesRegex(ValueError, "at least two"):
            detector.fit(batch(np.ones((1, 4, 1))))

    def test_validate_feature_names_and_estimator_contracts(self):
        for features in ((), ("unknown",), ("rms", "rms")):
            with self.subTest(features=features), self.assertRaisesRegex(ValueError, "features"):
                WindowFeatureTransformer(features=features)
        with self.assertRaisesRegex(ValueError, "both"):
            SklearnWindowAnomalyDetector(RecordingEstimator(), n_estimators=5)
        with self.assertRaisesRegex(TypeError, "decision_function"):
            SklearnWindowAnomalyDetector(object())

    def test_malformed_estimator_decision_output_rejected(self):
        class WrongOutput(RecordingEstimator):
            def decision_function(self, matrix):
                return np.array([[1, 2]], dtype=float)

        detector = SklearnWindowAnomalyDetector(WrongOutput(), features=("rms",))
        detector.fit(batch(np.ones((2, 4, 1))))
        with self.assertRaisesRegex(ValueError, "decision_function"):
            detector.score(batch(np.ones((2, 4, 1))))

    def test_explicit_registry_preserves_estimator_type(self):
        registry = DiagnosticRegistry()
        register_sklearn_window_detectors(registry)
        metadata = registry.get_spec("SklearnWindowAnomalyDetector")
        self.assertEqual(metadata.backend, "sklearn")
        self.assertEqual(metadata.operations, ("fit", "score"))
        created = registry.create("SklearnWindowAnomalyDetector", n_estimators=3)
        self.assertEqual(created.estimator.__class__.__name__, "IsolationForest")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_sklearn_window_detectors(registry)


if __name__ == "__main__":
    unittest.main()
