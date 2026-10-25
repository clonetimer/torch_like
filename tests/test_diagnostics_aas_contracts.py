"""P1/P2 diagnostics: machine-readable parameter schema and explicit decisions."""

import json
import unittest

import numpy as np

from torch_like.diagnostics import (
    CUSUMDetector,
    DecisionMode,
    DiagnosticAlgorithmSpec,
    DiagnosticDecision,
    DiagnosticParameterSpec,
    DiagnosticRegistry,
    DiagnosticResult,
    EWMADetector,
    ZScoreDetector,
    register_statistical_detectors,
)


class ParameterContractTests(unittest.TestCase):
    def test_schema_is_json_serializable_and_exact(self):
        p = DiagnosticParameterSpec(
            "alpha",
            "number",
            default=0.2,
            minimum=0.0,
            exclusive_minimum=True,
            maximum=1.0,
            description="EWMA smoothing",
        )
        spec = DiagnosticAlgorithmSpec(
            "ewma",
            "statistical",
            "anomaly_detection",
            "anomaly_score",
            parameters=(p, DiagnosticParameterSpec("enabled", "boolean", default=False)),
        )
        self.assertEqual(json.loads(json.dumps(spec.to_dict()))["parameters"][0]["name"], "alpha")
        self.assertEqual(spec.validate_parameters({}), {"alpha": 0.2, "enabled": False})
        self.assertEqual(spec.validate_parameters({"alpha": 1}), {"alpha": 1, "enabled": False})
        for value in (0, -1.0, float("inf"), "0.2", True, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spec.validate_parameters({"alpha": value})
        with self.assertRaisesRegex(ValueError, "unknown"):
            spec.validate_parameters({"unsupported": 1})

    def test_invalid_metadata_defaults_required_choices_and_names(self):
        for kwargs in (
            {"name": "bad", "kind": "number", "default": float("nan")},
            {"name": "bad", "kind": "boolean", "minimum": 0},
            {"name": "bad", "kind": "number", "minimum": 2, "maximum": 1},
            {"name": "bad", "kind": "number", "minimum": 0, "exclusive_minimum": True, "default": 0},
            {"name": "bad", "kind": "integer", "choices": (1, 1)},
            {"name": "bad", "kind": "string", "choices": ("a", 2)},
            {"name": "bad", "kind": "integer", "required": True, "default": 2},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                DiagnosticParameterSpec(**kwargs)
        spec = DiagnosticAlgorithmSpec(
            "x",
            "external",
            "anomaly_detection",
            "anomaly_score",
            parameters=(DiagnosticParameterSpec("foo", "integer", required=True),),
        )
        with self.assertRaisesRegex(ValueError, "required"):
            spec.validate_parameters({})
        with self.assertRaisesRegex(ValueError, "invalid value"):
            spec.validate_parameters({"foo": 1.4})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            DiagnosticAlgorithmSpec(
                "dup",
                "external",
                "anomaly_detection",
                "anomaly_score",
                parameters=(DiagnosticParameterSpec("x", "string"), DiagnosticParameterSpec("x", "string")),
            )

    def test_registry_preserves_legacy_constructor_behavior(self):
        reg = DiagnosticRegistry()
        reg.register(
            DiagnosticAlgorithmSpec("old", "external", "anomaly_detection", "anomaly_score"),
            lambda **kwargs: kwargs,
        )
        self.assertEqual(reg.create("old", arbitrary=7), {"arbitrary": 7})

    def test_registry_validates_real_statistical_models(self):
        reg = DiagnosticRegistry()
        register_statistical_detectors(reg)
        self.assertIsInstance(reg.create("CUSUMDetector"), CUSUMDetector)
        self.assertIsInstance(reg.create("ZScoreDetector"), ZScoreDetector)
        self.assertIsInstance(reg.create("EWMADetector"), EWMADetector)
        self.assertEqual(reg.get_spec("CUSUMDetector").validate_parameters({"k": 0.5})["h"], 5.0)
        self.assertEqual(reg.get_spec("CUSUMDetector").validate_parameters({"k": 0})["k"], 0)
        self.assertEqual(reg.create("CUSUMDetector", k=0).k, 0.0)
        self.assertEqual(reg.get_spec("EWMADetector").to_dict()["parameters"][0]["name"], "alpha")
        for name, kwargs in (
            ("CUSUMDetector", {"h": 0}),
            ("EWMADetector", {"alpha": 0}),
            ("ZScoreDetector", {"sigma_threshold": "3"}),
            ("CUSUMDetector", {"extra": True}),
        ):
            with self.subTest(name=name, kwargs=kwargs), self.assertRaises(ValueError):
                reg.create(name, **kwargs)


class DiagnosticDecisionTests(unittest.TestCase):
    def test_score_threshold_keeps_resolution_and_does_not_calibrate(self):
        result = DiagnosticResult(
            "anomaly_detection",
            "anomaly_score",
            np.array([[[0.1, 3.5], [4.0, 2.1]]]),
            "CUSUM",
            threshold=99.0,
        )
        decision = DiagnosticDecision.from_threshold(result, 3.0)
        np.testing.assert_array_equal(decision.labels, [[[0, 1], [1, 0]]])
        self.assertEqual(decision.decision_mode, DecisionMode.SCORE_THRESHOLD)
        self.assertEqual(decision.labels.shape, result.values.shape)
        self.assertEqual(decision.thresholds.shape, result.values.shape)
        self.assertFalse(decision.labels.flags.writeable)
        self.assertFalse(decision.thresholds.flags.writeable)
        self.assertEqual(json.loads(json.dumps(decision.to_dict()))["decision_mode"], "score_threshold")
        self.assertEqual(result.threshold, 99.0)  # caller threshold does not mutate original result

    def test_time_varying_thresholds_and_invalid_broadcasting(self):
        result = DiagnosticResult(
            "anomaly_detection",
            "anomaly_score",
            np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
            "EWMA",
        )
        thresholds = np.array([[0.0, 2.0, 4.0], [2.0, 3.0, 0.0]])
        d = DiagnosticDecision.from_threshold(result, thresholds)
        np.testing.assert_array_equal(d.labels, [[1, 0, 0], [1, 0, 1]])
        thresholds[0, 0] = 100.0
        self.assertEqual(d.thresholds[0, 0], 0.0)
        for invalid in (np.ones((2, 1)), np.ones((3,)), np.array(True), np.array(np.nan)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DiagnosticDecision.from_threshold(result, invalid)

    def test_single_authority_rejects_fabricated_label_and_generic_threshold(self):
        score = DiagnosticResult("anomaly_detection", "anomaly_score", np.array([0.9, 0.3]), "S")
        with self.assertRaisesRegex(ValueError, "disagree"):
            DiagnosticDecision(
                "anomaly_detection",
                "score_threshold",
                [0, 0],
                "S",
                source_result=score,
                thresholds=np.array([0.5, 0.5]),
            )
        with self.assertRaisesRegex(ValueError, "generic thresholds"):
            DiagnosticDecision("anomaly_detection", "native", [1, 0], "S", thresholds=np.array([0.5, 0.5]))
        with self.assertRaisesRegex(ValueError, "competing score"):
            DiagnosticDecision.from_labels(
                task="anomaly_detection", labels=np.array([1, 0]), algorithm_name="S", source_result=score
            )
        with self.assertRaisesRegex(ValueError, "binary"):
            DiagnosticDecision.from_labels(task="anomaly_detection", labels=np.array([0, 2]), algorithm_name="S")

    def test_native_and_multiclass_label_are_distinct_from_scores(self):
        score = DiagnosticResult("anomaly_detection", "anomaly_score", np.array([2.2, 0.1]), "Native")
        native = DiagnosticDecision.from_labels(
            task="anomaly_detection",
            labels=np.array([1, 0]),
            algorithm_name="Native",
            decision_mode="native",
            source_result=score,
        )
        self.assertIs(native.source_result, score)
        multiclass = DiagnosticDecision.from_labels(
            task="fault_classification",
            labels=np.array([0, 2, 1]),
            algorithm_name="Classifier",
            label_names=("normal", "bearing", "gear"),
        )
        self.assertEqual(multiclass.sample_count, 3)
        np.testing.assert_array_equal(multiclass.labels, [0, 2, 1])
        self.assertEqual(multiclass.to_dict()["label_names"][2], "gear")
        with self.assertRaisesRegex(ValueError, "outside"):
            DiagnosticDecision.from_labels(
                task="fault_classification",
                labels=np.array([3]),
                algorithm_name="Bad",
                label_names=("normal", "fault"),
            )
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            DiagnosticDecision.from_labels(
                task="fault_classification",
                labels=np.ones((2, 2), dtype=int),
                algorithm_name="Bad",
            )


if __name__ == "__main__":
    unittest.main()
