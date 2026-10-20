"""R5a health assessment: numeric semantics without caller-data governance."""

import unittest

import numpy as np

from torch_like.diagnostics.health import (
    GaussianHealthIndex,
    MahalanobisHealthIndex,
    register_health_index_estimators,
)
from torch_like.diagnostics.registry import DiagnosticRegistry
from torch_like.diagnostics.types import DiagnosticOutputKind, DiagnosticTask, SignalBatch


def batch(data, *, channels=("a", "b"), units=("A", "V"), rate=50.0):
    return SignalBatch(
        np.asarray(data, dtype=float),
        sample_rate_hz=rate,
        channel_names=channels,
        units=units,
    )


def baseline():
    x = np.array(
        [
            [[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]],
            [[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]],
        ]
    )
    return batch(x)


class HealthIndexTests(unittest.TestCase):
    def test_finite_bounded_health_index_high_means_reference_closeness(self):
        reference = baseline()
        algorithm = GaussianHealthIndex(health_scale=2.0).fit(reference)
        current = batch([[[0.0, 0.0]], [[1.0, 2.0]], [[4.0, 8.0]]])
        result = algorithm.score(current)
        self.assertEqual(result.task, DiagnosticTask.HEALTH_ASSESSMENT)
        self.assertEqual(result.output_kind, DiagnosticOutputKind.HEALTH_INDEX)
        self.assertEqual(result.values.shape, (3,))
        self.assertEqual(result.algorithm_name, "GaussianHealthIndex")
        self.assertIsNone(result.threshold)
        self.assertEqual(result.values[0], 1.0)
        self.assertTrue(result.values[0] > result.values[1] > result.values[2] > 0)
        deviations = algorithm.deviation(current)
        np.testing.assert_allclose(result.values, np.exp(-deviations / 2))
        self.assertFalse(deviations.flags.writeable)
        self.assertFalse(result.values.flags.writeable)

    def test_dimensionless_std_rms_matches_manual_reference_calculation(self):
        reference = baseline()
        algorithm = GaussianHealthIndex(health_scale=1).fit(reference)
        current = batch([[[1, 2], [-1, -2]]])
        std_a = np.std(reference.values[:, :, 0], ddof=1)
        std_b = np.std(reference.values[:, :, 1], ddof=1)
        expected = np.sqrt((1 / std_a**2 + 4 / std_b**2) / 2)
        self.assertAlmostEqual(algorithm.deviation(current)[0], expected)
        self.assertAlmostEqual(algorithm.score(current).values[0], np.exp(-expected))

    def test_mahalanobis_full_shrinkage_equivalent_to_gaussian_index(self):
        reference = baseline()
        current = batch([[[1, -2], [0, 0]], [[2, 3], [-1, 2]]])
        gaussian = GaussianHealthIndex(health_scale=2).fit(reference)
        mahal = MahalanobisHealthIndex(health_scale=2, shrinkage=1).fit(reference)
        np.testing.assert_allclose(mahal.deviation(current), gaussian.deviation(current))
        np.testing.assert_allclose(mahal.score(current).values, gaussian.score(current).values)

    def test_mahalanobis_detects_off_correlation_direction(self):
        ref = batch([[[x, x] for x in (-3, -2, -1, 1, 2, 3)]])
        model = MahalanobisHealthIndex(shrinkage=0.1).fit(ref)
        same = batch([[[1.5, 1.5]]])
        broken = batch([[[1.5, -1.5]]])
        self.assertLess(model.deviation(same)[0], model.deviation(broken)[0])
        self.assertGreater(model.score(same).values[0], model.score(broken).values[0])

    def test_single_channel_and_flat_reference_numerically_regularized(self):
        ref = batch([[[3], [3], [3]]], channels=("x",), units=("m",))
        probe = batch([[[3]], [[3.0001]]], channels=("x",), units=("m",))
        for class_ in (GaussianHealthIndex, MahalanobisHealthIndex):
            with self.subTest(class_name=class_.__name__):
                result = class_().fit(ref).score(probe)
                self.assertTrue(np.isfinite(result.values).all())
                self.assertEqual(result.values[0], 1.0)
                self.assertGreater(result.values[0], result.values[1])
                self.assertTrue(np.all((result.values >= 0) & (result.values <= 1)))

    def test_scoring_is_pure_and_does_not_require_split_labels_or_recordings(self):
        ref = baseline()
        query = batch([[[1, 1]], [[4, 4]]])
        for cls in (GaussianHealthIndex, MahalanobisHealthIndex):
            model = cls().fit(ref)
            a = model.score(query)
            b = model.score(query)
            np.testing.assert_array_equal(a.values, b.values)
            self.assertEqual(a.sample_count, 2)
            self.assertFalse(hasattr(model, "recording_ids"))
            self.assertFalse(hasattr(model, "test_labels"))

    def test_fit_must_precede_score(self):
        for cls in (GaussianHealthIndex, MahalanobisHealthIndex):
            model = cls()
            with self.assertRaisesRegex(RuntimeError, "fit a user-supplied"):
                model.score(baseline())
            with self.assertRaisesRegex(RuntimeError, "fit a user-supplied"):
                model.deviation(baseline())

    def test_metadata_safety_is_not_dataset_source_audit(self):
        ref = baseline()
        for cls in (GaussianHealthIndex, MahalanobisHealthIndex):
            model = cls().fit(ref)
            cases = [
                (batch([[[0, 0]]], channels=("b", "a")), "channel_names"),
                (batch([[[0, 0]]], units=("V", "A")), "units"),
                (batch([[[0, 0]]], rate=100), "sample_rate_hz"),
            ]
            for candidate, expected in cases:
                with self.subTest(algorithm=cls.__name__, error=expected):
                    with self.assertRaisesRegex(ValueError, expected):
                        model.score(candidate)

    def test_reject_nonfinite_and_invalid_reference_or_parameters(self):
        for kwargs in (
            {"health_scale": 0},
            {"health_scale": -1},
            {"health_scale": float("nan")},
            {"min_std": 0},
            {"min_std": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GaussianHealthIndex(**kwargs)
        for shrinkage in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(shrinkage=shrinkage), self.assertRaisesRegex(ValueError, "shrinkage"):
                MahalanobisHealthIndex(shrinkage=shrinkage)
        with self.assertRaisesRegex(TypeError, "SignalBatch"):
            GaussianHealthIndex().fit(object())
        one = batch([[[1, 2]]])
        with self.assertRaisesRegex(ValueError, "two observations"):
            GaussianHealthIndex().fit(one)
        nan_ref = SignalBatch(
            np.array([[[1.0, np.nan], [1.0, 2.0]]]),
            sample_rate_hz=50,
            channel_names=("a", "b"),
            units=("A", "V"),
            allow_missing=True,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            GaussianHealthIndex().fit(nan_ref)
        with self.assertRaisesRegex(ValueError, "finite"):
            GaussianHealthIndex().fit(baseline()).score(nan_ref)

    def test_baseline_refit_is_explicit_and_updates_health_reference(self):
        model = GaussianHealthIndex().fit(baseline())
        original = model.score(batch([[[0, 0]]])).values[0]
        shifted = batch([[[10, 10], [11, 11], [12, 12]]])
        model.fit(shifted)
        now = model.score(batch([[[0, 0]]])).values[0]
        self.assertEqual(original, 1.0)
        self.assertLess(now, original)

    def test_opt_in_registry_returns_native_health_estimators(self):
        registry = DiagnosticRegistry()
        register_health_index_estimators(registry)
        specs = registry.list_specs(task="health_assessment")
        self.assertEqual(set(specs), {"GaussianHealthIndex", "MahalanobisHealthIndex"})
        for name, cls in (
            ("GaussianHealthIndex", GaussianHealthIndex),
            ("MahalanobisHealthIndex", MahalanobisHealthIndex),
        ):
            obj = registry.create(name, health_scale=1.5)
            self.assertIsInstance(obj, cls)
            self.assertEqual(specs[name].output_kind, DiagnosticOutputKind.HEALTH_INDEX)
            self.assertEqual(specs[name].operations, ("fit", "score"))
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_health_index_estimators(registry)


if __name__ == "__main__":
    unittest.main()
