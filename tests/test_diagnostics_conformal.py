"""Mask-conditional split conformal calibration tests."""

import unittest

import numpy as np

from torch_like.diagnostics.conformal import (
    ConformalPredictionSet,
    ModalityConditionedConformalClassifier,
)


def calibration_data():
    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    masks = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=bool)
    return probabilities, labels, masks


class ModalityConditionedConformalTests(unittest.TestCase):
    def test_calibrates_each_exact_availability_mask_independently(self):
        probabilities, labels, masks = calibration_data()
        calibrator = ModalityConditionedConformalClassifier(alpha=0.5, min_calibration_per_mask=3)
        calibrator.fit(probabilities, labels, masks)

        quantiles = calibrator.group_quantiles_
        self.assertAlmostEqual(quantiles[(1, 0)], 0.3)
        self.assertAlmostEqual(quantiles[(0, 1)], 0.2)
        self.assertEqual(calibrator.group_sizes_, {(1, 0): 3, (0, 1): 3})

    def test_prediction_sets_use_the_matching_group_quantile_and_allow_empty_sets(self):
        probabilities, labels, masks = calibration_data()
        calibrator = ModalityConditionedConformalClassifier(alpha=0.5, min_calibration_per_mask=3)
        calibrator.fit(probabilities, labels, masks)

        result = calibrator.predict_set(
            np.array([[0.75, 0.25], [0.3, 0.7]], dtype=np.float64),
            np.array([[1, 0], [0, 1]], dtype=bool),
        )
        self.assertIsInstance(result, ConformalPredictionSet)
        np.testing.assert_array_equal(result.included, [[True, False], [False, False]])
        np.testing.assert_array_equal(result.set_sizes, [1, 0])
        self.assertEqual(result.class_indices, ((0,), ()))
        self.assertEqual(result.to_dict()["method"], "mask_conditional_split_conformal_lac")

    def test_unseen_mask_is_rejected_without_global_fallback(self):
        probabilities, labels, masks = calibration_data()
        calibrator = ModalityConditionedConformalClassifier(alpha=0.5).fit(probabilities, labels, masks)
        with self.assertRaisesRegex(ValueError, "no calibration group"):
            calibrator.predict_set(
                np.array([[0.8, 0.2]]),
                np.array([[1, 1]], dtype=bool),
            )

    def test_small_groups_use_the_finite_sample_conservative_quantile(self):
        calibrator = ModalityConditionedConformalClassifier(alpha=0.1)
        calibrator.fit(
            np.array([[0.6, 0.4]]),
            np.array([0], dtype=np.int64),
            np.array([[1, 0]], dtype=bool),
        )
        self.assertEqual(calibrator.group_quantiles_[(1, 0)], 1.0)
        result = calibrator.predict_set(np.array([[0.6, 0.4]]), np.array([[1, 0]]))
        np.testing.assert_array_equal(result.included, [[True, True]])

    def test_minimum_group_size_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "requires at least 2"):
            ModalityConditionedConformalClassifier(
                alpha=0.1,
                min_calibration_per_mask=2,
            ).fit(
                np.array([[0.6, 0.4]]),
                np.array([0], dtype=np.int64),
                np.array([[1, 0]], dtype=bool),
            )

    def test_validates_probabilities_labels_and_availability_masks(self):
        calibrator = ModalityConditionedConformalClassifier(alpha=0.1)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            calibrator.fit(
                np.array([[0.6, 0.6]]),
                np.array([0], dtype=np.int64),
                np.array([[1, 0]]),
            )
        with self.assertRaisesRegex(ValueError, "integer class index"):
            calibrator.fit(
                np.array([[0.6, 0.4]]),
                np.array([0.0]),
                np.array([[1, 0]]),
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            calibrator.fit(
                np.array([[0.6, 0.4]]),
                np.array([2], dtype=np.int64),
                np.array([[1, 0]]),
            )
        with self.assertRaisesRegex(ValueError, "at least one available"):
            calibrator.fit(
                np.array([[0.6, 0.4]]),
                np.array([0], dtype=np.int64),
                np.array([[0, 0]]),
            )
        with self.assertRaisesRegex(ValueError, "0/1"):
            calibrator.fit(
                np.array([[0.6, 0.4]]),
                np.array([0], dtype=np.int64),
                np.array([[1, 2]]),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            calibrator.fit(
                np.array([[np.nan, np.nan]]),
                np.array([0], dtype=np.int64),
                np.array([[1, 0]]),
            )

    def test_prediction_requires_matching_class_and_modality_dimensions(self):
        probabilities, labels, masks = calibration_data()
        calibrator = ModalityConditionedConformalClassifier(alpha=0.5).fit(probabilities, labels, masks)
        with self.assertRaisesRegex(ValueError, "at least two classes"):
            calibrator.predict_set(np.array([[1.0]]), np.array([[1, 0]]))
        with self.assertRaisesRegex(ValueError, "modality columns"):
            calibrator.predict_set(np.array([[0.5, 0.5]]), np.array([[1, 0, 1]]))
        with self.assertRaisesRegex(ValueError, "sample counts differ"):
            calibrator.predict_set(np.array([[0.5, 0.5], [0.4, 0.6]]), np.array([[1, 0]]))

    def test_prediction_result_owns_immutable_copies(self):
        included = np.array([[True, False]])
        masks = np.array([[True, False]])
        quantiles = np.array([0.2])
        result = ConformalPredictionSet(included, masks, quantiles, 0.1)
        included[0, 0] = False
        masks[0, 0] = False
        quantiles[0] = 1.0
        self.assertTrue(result.included[0, 0])
        self.assertTrue(result.modality_mask[0, 0])
        self.assertEqual(float(result.quantiles[0]), 0.2)
        self.assertFalse(result.included.flags.writeable)
        self.assertFalse(result.modality_mask.flags.writeable)
        self.assertFalse(result.quantiles.flags.writeable)

    def test_rejects_invalid_alpha_and_minimum_group_size(self):
        for alpha in (0.0, 1.0, np.nan, np.inf):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                ModalityConditionedConformalClassifier(alpha=alpha)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ModalityConditionedConformalClassifier(min_calibration_per_mask=size)


if __name__ == "__main__":
    unittest.main()
