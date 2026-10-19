"""Regression tests for the task-neutral diagnostics protocol."""

import unittest

import numpy as np
import torch

from torch_like.classical import create_estimator
from torch_like.diagnostics import (
    DiagnosticAlgorithmSpec,
    DiagnosticOutputKind,
    DiagnosticRegistry,
    DiagnosticResult,
    DiagnosticTask,
    SignalBatch,
)


class DiagnosticContractsTests(unittest.TestCase):
    def test_signal_axes_rate_and_defensive_copy(self):
        original = np.arange(24.0).reshape(2, 4, 3)
        batch = SignalBatch(original, 200.0, ("ax", "ay", "az"), units=("g", "g", "g"))
        self.assertEqual(batch.shape, (2, 4, 3))
        self.assertAlmostEqual(batch.duration_s, 0.02)
        original[0, 0, 0] = 999
        self.assertEqual(batch.values[0, 0, 0], 0.0)
        self.assertFalse(batch.values.flags.writeable)

    def test_2d_input_requires_explicit_single_sequence_conversion(self):
        batch = SignalBatch.from_sequence(np.ones((10, 2)), sample_rate_hz=100)
        self.assertEqual(batch.shape, (1, 10, 2))
        self.assertEqual(batch.channel_names, ("channel_0", "channel_1"))
        with self.assertRaises(ValueError):
            SignalBatch(np.ones((10, 2)), sample_rate_hz=100)
        with self.assertRaises(ValueError):
            SignalBatch.from_sequence(np.ones((1, 10, 2)), sample_rate_hz=100)

    def test_reject_missing_or_invalid_sampling_metadata(self):
        for rate in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "sample_rate_hz"):
                SignalBatch(np.ones((1, 4, 1)), rate)
        with self.assertRaisesRegex(ValueError, "channel_names"):
            SignalBatch(np.ones((1, 4, 2)), 100, ("same", "same"))
        with self.assertRaisesRegex(ValueError, "units"):
            SignalBatch(np.ones((1, 4, 2)), 100, units=("g",))
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            SignalBatch(np.ones((2, 4, 2)), 100, sample_ids=("duplicate", "duplicate"))

    def test_nonfinite_missing_value_policy(self):
        with self.assertRaises(ValueError):
            SignalBatch(np.array([[[float("nan")]]]), 1)
        batch = SignalBatch(np.array([[[float("nan")]]]), 1, allow_missing=True)
        self.assertTrue(np.isnan(batch.values[0, 0, 0]))
        with self.assertRaises(ValueError):
            SignalBatch(np.array([[[float("inf")]]]), 1, allow_missing=True)

    def test_complex_input_must_not_silently_discard_imaginary_part(self):
        with self.assertRaisesRegex(ValueError, "complex-valued"):
            SignalBatch(np.array([[[1.0 + 2.0j]]]), sample_rate_hz=100)
        with self.assertRaisesRegex(ValueError, "complex-valued"):
            DiagnosticResult("anomaly_detection", "anomaly_score", np.array([1.0 + 2.0j]), "score")

    def test_timestamps_are_strictly_increasing_per_sample(self):
        batch = SignalBatch(np.ones((2, 4, 1)), 100, timestamps_s=np.arange(4))
        self.assertEqual(batch.timestamps_s.shape, (2, 4))
        self.assertFalse(batch.timestamps_s.flags.writeable)
        with self.assertRaisesRegex(ValueError, "increasing"):
            SignalBatch(np.ones((2, 4, 1)), 100, timestamps_s=[0, 1, 1, 2])
        with self.assertRaisesRegex(ValueError, "timestamps_s"):
            SignalBatch(np.ones((2, 4, 1)), 100, timestamps_s=np.ones((3, 4)))

    def test_anomaly_output_and_threshold(self):
        source = np.array([[0.2, 0.7], [0.1, 0.8]])
        result = DiagnosticResult("anomaly_detection", "anomaly_score", source, "GlobalSTD", threshold=0.5)
        self.assertEqual(result.task, DiagnosticTask.ANOMALY_DETECTION)
        self.assertEqual(result.output_kind, DiagnosticOutputKind.ANOMALY_SCORE)
        self.assertEqual(result.sample_count, 2)
        source[0, 0] = 5
        self.assertEqual(result.values[0, 0], 0.2)
        with self.assertRaisesRegex(ValueError, "anomaly_score"):
            DiagnosticResult("anomaly_detection", "rul_value", np.array([0.1]), "invalid")

    def test_classification_results_and_invalid_probabilities(self):
        result = DiagnosticResult(
            "fault_classification",
            "class_probability",
            np.array([[0.2, 0.8]]),
            "classifier",
            label_names=("normal", "fault"),
        )
        self.assertEqual(result.label_names, ("normal", "fault"))
        for invalid in (np.array([0.2, 0.8]), np.array([[0.2, 1.2]])):
            with self.subTest(values=invalid), self.assertRaises(ValueError):
                DiagnosticResult("fault_classification", "class_probability", invalid, "classifier")
        with self.assertRaisesRegex(ValueError, "label_names"):
            DiagnosticResult(
                "fault_classification",
                "class_probability",
                np.ones((2, 2)),
                "classifier",
                label_names=("only",),
            )
        labels = DiagnosticResult("fault_classification", "fault_label", np.array(["normal", "fault"]), "rules")
        self.assertEqual(labels.sample_count, 2)

    def test_rul_nonnegative_and_threshold_contract(self):
        result = DiagnosticResult("rul_regression", "rul_value", np.array([12.0, 10.0]), "rul_model")
        self.assertEqual(result.sample_count, 2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            DiagnosticResult("rul_regression", "rul_value", np.array([-1.0]), "rul_model")
        with self.assertRaisesRegex(ValueError, "threshold"):
            DiagnosticResult("rul_regression", "rul_value", np.array([1.0]), "rul_model", threshold=0.3)

    def test_isolation_and_health_output_semantics(self):
        isolated = DiagnosticResult("fault_isolation", "fault_source_score", np.array([[0.8, 0.2]]), "isolate")
        health = DiagnosticResult("health_assessment", "health_index", np.array([0.9]), "health")
        self.assertEqual(isolated.sample_count, 1)
        self.assertEqual(health.sample_count, 1)
        for task, wrong, message in (
            ("fault_isolation", "anomaly_score", "fault_source_score"),
            ("health_assessment", "rul_value", "health_index"),
        ):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, message):
                DiagnosticResult(task, wrong, np.array([0.1]), "invalid")
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, message):
                DiagnosticAlgorithmSpec("invalid", "statistical", task, wrong)

    def test_registry_preserves_native_sklearn_and_torch_objects(self):
        registry = DiagnosticRegistry()
        registry.register(
            DiagnosticAlgorithmSpec("native_rf", "sklearn", "fault_classification", "class_probability"),
            lambda **kwargs: create_estimator("RandomForestClassifier", **kwargs),
        )
        registry.register(
            DiagnosticAlgorithmSpec("native_torch", "torch_like", "fault_classification", "class_probability"),
            lambda **kwargs: torch.nn.Linear(**kwargs),
        )
        native_rf = registry.create("native_rf", n_estimators=3)
        self.assertEqual(native_rf.__class__.__name__, "RandomForestClassifier")
        self.assertTrue(hasattr(native_rf, "fit"))
        native_torch = registry.create("native_torch", in_features=2, out_features=3)
        self.assertIsInstance(native_torch, torch.nn.Module)
        self.assertEqual(len(registry.list_specs(task="fault_classification")), 2)
        self.assertEqual(len(registry.list_specs(backend="sklearn")), 1)

    def test_registry_duplicate_unknown_and_spec_errors(self):
        registry = DiagnosticRegistry()
        spec = DiagnosticAlgorithmSpec("score", "statistical", "anomaly_detection", "anomaly_score")
        registry.register(spec, lambda **kwargs: kwargs)
        self.assertEqual(registry.create("score", threshold=1), {"threshold": 1})
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(spec, lambda: None)
        with self.assertRaises(KeyError):
            registry.get_spec("unknown")
        with self.assertRaises(KeyError):
            registry.create("unknown")
        with self.assertRaises(TypeError):
            registry.register(DiagnosticAlgorithmSpec("bad", "sklearn", "anomaly_detection", "anomaly_score"), None)
        with self.assertRaisesRegex(ValueError, "backend"):
            DiagnosticAlgorithmSpec("bad", "none", "anomaly_detection", "anomaly_score")
        with self.assertRaisesRegex(ValueError, "operations"):
            DiagnosticAlgorithmSpec("bad", "sklearn", "anomaly_detection", "anomaly_score", ("fit", "fit"))
        with self.assertRaisesRegex(ValueError, "anomaly_score"):
            DiagnosticAlgorithmSpec("bad", "statistical", "anomaly_detection", "rul_value")
        with self.assertRaisesRegex(ValueError, "rul_value"):
            DiagnosticAlgorithmSpec("bad", "torch_like", "rul_regression", "anomaly_score")


if __name__ == "__main__":
    unittest.main()
