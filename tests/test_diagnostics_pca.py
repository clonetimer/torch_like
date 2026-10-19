"""Reference geometry, independent calibration and metadata safety for PCA monitors."""

import unittest

import numpy as np

from torch_like.diagnostics import DiagnosticRegistry, SignalBatch
from torch_like.diagnostics.pca import PCASPEDetector, PCAT2Detector, register_pca_detectors


def reference():
    return SignalBatch.from_sequence(
        np.array([[-2, -2], [-1, -1], [1, 1], [2, 2]], dtype=float),
        sample_rate_hz=100,
        channel_names=("motor_current", "motor_torque"),
        units=("A", "N.m"),
    )


def incoming(pairs, *, rate=100, names=("motor_current", "motor_torque"), units=("A", "N.m")):
    return SignalBatch.from_sequence(
        np.asarray(pairs, dtype=float),
        sample_rate_hz=rate,
        channel_names=names,
        units=units,
    )


class PCAStatisticalTests(unittest.TestCase):
    def test_known_correlated_two_sensor_t2_and_spe_geometry(self):
        fit = reference()
        t2 = PCAT2Detector(n_components=1).fit(fit)
        spe = PCASPEDetector(n_components=1).fit(fit)
        test = incoming([[3, 3], [3, -3]])
        np.testing.assert_allclose(t2.score(test).values[0], [2.7, 0.0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(spe.score(test).values[0], [0.0, 5.4], rtol=1e-12, atol=1e-12)
        self.assertIsNone(t2.score(test).threshold)
        self.assertEqual(t2.eigenvalues_.shape, (1,))

    def test_independent_development_calibration_and_frozen_threshold(self):
        fitted = reference()
        validation = incoming([[1, 1], [2, 2], [0, 0], [1.5, 1.5]])
        test = incoming([[3, 3], [3, -3]])
        detector = PCAT2Detector().fit(fitted)
        with self.assertRaisesRegex(RuntimeError, "calibrate"):
            detector.alarm_mask(test)
        with self.assertRaisesRegex(ValueError, "independent"):
            detector.calibrate(fitted)
        detector.calibrate(validation, quantile=0.75)
        saved = detector.threshold_
        self.assertGreaterEqual(saved, 0)
        self.assertEqual(detector.score(test).threshold, saved)
        mask = detector.alarm_mask(test)
        self.assertEqual(mask.shape, (1, 2))
        detector.score(incoming([[100, 100]]))
        self.assertEqual(detector.threshold_, saved)

    def test_spe_detects_transverse_deviation(self):
        detector = PCASPEDetector().fit(reference())
        detector.calibrate(incoming([[1, 1], [1.5, 1.5], [2, 2]]))
        self.assertAlmostEqual(detector.threshold_, 0.0, places=12)
        np.testing.assert_array_equal(detector.alarm_mask(incoming([[3, -3], [0, 0]])), [[True, False]])

    def test_reject_mismatched_sensor_order_units_and_frequency(self):
        detector = PCAT2Detector().fit(reference())
        for kwargs, message in (
            ({"names": ("motor_torque", "motor_current")}, "channel_names"),
            ({"units": ("mA", "N.m")}, "units"),
            ({"rate": 50}, "sample_rate_hz"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                detector.score(incoming([[1, 1]], **kwargs))

    def test_parameter_rank_and_input_quality_guards(self):
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            PCAT2Detector().score(incoming([[1, 1]]))
        with self.assertRaisesRegex(ValueError, "components"):
            PCAT2Detector(n_components=2).fit(reference())
        with self.assertRaisesRegex(ValueError, "components"):
            PCAT2Detector(n_components=0)
        with self.assertRaisesRegex(ValueError, "nonzero variance"):
            PCAT2Detector().fit(incoming([[1, 1], [1, 1], [1, 1]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            PCAT2Detector().fit(
                SignalBatch.from_sequence(
                    np.array([[np.nan, 1], [1, 1], [2, 2]]),
                    sample_rate_hz=100,
                    channel_names=("motor_current", "motor_torque"),
                    units=("A", "N.m"),
                    allow_missing=True,
                )
            )
        detector = PCAT2Detector().fit(reference())
        for value in (0.0, 1.0, np.nan, np.inf):
            with self.subTest(quantile=value), self.assertRaisesRegex(ValueError, "quantile"):
                detector.calibrate(incoming([[1, 1], [2, 2]]), quantile=value)
        with self.assertRaisesRegex(ValueError, "at least two"):
            detector.calibrate(incoming([[1, 1]]))

    def test_registry_factory_admission_and_duplicate_guard(self):
        registry = DiagnosticRegistry()
        register_pca_detectors(registry)
        self.assertEqual(set(registry.list_specs()), {"PCAT2Detector", "PCASPEDetector"})
        self.assertIsInstance(registry.create("PCASPEDetector", n_components=1), PCASPEDetector)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_pca_detectors(registry)

    def test_refitting_clears_calibration(self):
        detector = PCAT2Detector().fit(reference())
        detector.calibrate(incoming([[1, 1], [2, 2]]))
        detector.fit(reference())
        self.assertIsNone(detector.score(incoming([[3, 3]])).threshold)
        with self.assertRaises(RuntimeError):
            _ = detector.threshold_


if __name__ == "__main__":
    unittest.main()
