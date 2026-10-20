"""Direct R5b algorithm composition, including typed dependent source mapping."""

import json
import unittest

import numpy as np

from torch_like.diagnostics import (
    DiagnosticPipeline,
    DiagnosticResult,
    DiagnosticStage,
    FaultLabelSchema,
    FaultSourceEvidenceMapper,
    FaultSourceRulebook,
    GaussianHealthIndex,
    SignalBatch,
    SklearnWindowFaultClassifier,
    ZScoreDetector,
)


def signals(x):
    return SignalBatch(
        np.asarray(x, dtype=float),
        sample_rate_hz=50,
        channel_names=("vibration", "temperature"),
        units=("g", "C"),
    )


class FakeAlgorithm:
    def score(self, batch):
        return DiagnosticResult("health_assessment", "health_index", np.ones(batch.shape[0]), "fake")


class WrongOutput:
    def score(self, batch):
        return np.zeros(batch.shape[0])


class WrongCount:
    def score(self, batch):
        return DiagnosticResult("health_assessment", "health_index", np.ones(batch.shape[0] + 1), "bad")


class CompositionTests(unittest.TestCase):
    def setUp(self):
        healthy = signals([[[i, i] for i in (-2, -1, 0, 1, 2)] for _ in range(3)])
        train = signals([[[float(k) + (i - 2) * 0.1] * 2 for i in range(5)] for k in (0, 0, 0, 4, 4, 4)])
        self.normal = GaussianHealthIndex().fit(healthy)
        self.z = ZScoreDetector().fit(healthy)
        self.classifier = SklearnWindowFaultClassifier(n_estimators=8, random_state=5)
        self.schema = FaultLabelSchema(("normal", "misalignment"))
        self.classifier.fit(train, ("normal",) * 3 + ("misalignment",) * 3, self.schema)
        self.rulebook = FaultSourceRulebook(
            class_labels=("normal", "misalignment"),
            source_names=("motor",),
            fault_to_sources={"normal": (), "misalignment": ("motor",)},
        )
        self.batch = signals([[[1, 1]] * 5, [[4, 4]] * 5])

    def test_mixed_parallel_diagnostics_and_serial_fault_source_mapper(self):
        mapper = FaultSourceEvidenceMapper(self.rulebook)
        pipeline = DiagnosticPipeline(
            DiagnosticStage("health", self.normal, method="score"),
            DiagnosticStage("z", self.z, method="score"),
            DiagnosticStage("fault", self.classifier, method="predict_proba"),
            DiagnosticStage("sources", mapper, method="predict", source="fault"),
        )
        result = pipeline.predict(self.batch)
        self.assertEqual(set(result.results), {"health", "z", "fault", "sources"})
        self.assertEqual(result["health"].task.value, "health_assessment")
        self.assertEqual(result["z"].task.value, "anomaly_detection")
        self.assertEqual(result["fault"].task.value, "fault_classification")
        self.assertEqual(result["sources"].task.value, "fault_isolation")
        self.assertEqual(result["sources"].values.shape, (2, 1))
        np.testing.assert_array_equal(result["sources"].values[:, 0], result["fault"].values[:, 1])
        self.assertEqual(result["health"].values.shape, (2,))
        self.assertEqual(result["z"].values.shape, (2, 5, 2))
        self.assertEqual(pipeline.score(self.batch).to_dict(), result.to_dict())
        self.assertIs(result["fault"].values.flags.writeable, False)
        self.assertEqual(json.loads(json.dumps(result.to_dict()))["fault"]["label_names"], ["normal", "misalignment"])
        self.assertNotIn("fused_score", result.to_dict())

    def test_no_unannounced_fit_evaluation_or_inference_fusion(self):
        p = DiagnosticPipeline(DiagnosticStage("health", self.normal))
        self.assertFalse(hasattr(p, "fit"))
        self.assertFalse(hasattr(p, "calibrate"))
        self.assertEqual(p.predict(self.batch)["health"].sample_count, 2)
        with self.assertRaises(TypeError):
            p.predict(object())
        with self.assertRaises(TypeError):
            p.predict(None)

    def test_invalid_stage_graph_and_duplicate_names(self):
        a = DiagnosticStage("a", FakeAlgorithm())
        for stages, error in (
            ((), "at least one"),
            ((a, a), "duplicate"),
            ((DiagnosticStage("b", FakeAlgorithm(), source="c"),), "unknown"),
        ):
            with self.subTest(stages=stages), self.assertRaisesRegex(ValueError, error):
                DiagnosticPipeline(*stages) if stages else DiagnosticPipeline()
        with self.assertRaises(TypeError):
            DiagnosticPipeline(object())
        with self.assertRaisesRegex(ValueError, "cannot be signals"):
            DiagnosticStage("signals", FakeAlgorithm())
        with self.assertRaisesRegex(ValueError, "method"):
            DiagnosticStage("a", FakeAlgorithm(), method="fit")
        with self.assertRaisesRegex(TypeError, "inference method"):
            DiagnosticStage("a", object())
        with self.assertRaisesRegex(ValueError, "source"):
            DiagnosticStage("a", FakeAlgorithm(), source="")

    def test_stage_output_and_source_semantic_errors_propagate(self):
        with self.assertRaisesRegex(TypeError, "must return DiagnosticResult"):
            DiagnosticPipeline(DiagnosticStage("bad", WrongOutput())).predict(self.batch)
        with self.assertRaisesRegex(ValueError, "sample axis"):
            DiagnosticPipeline(DiagnosticStage("bad", WrongCount())).predict(self.batch)
        bad = DiagnosticPipeline(
            DiagnosticStage("health", self.normal),
            DiagnosticStage("wrong", FaultSourceEvidenceMapper(self.rulebook), method="predict", source="health"),
        )
        with self.assertRaisesRegex(ValueError, "classification"):
            bad.predict(self.batch)

    def test_bundle_mapping_cannot_be_modified(self):
        result = DiagnosticPipeline(DiagnosticStage("health", self.normal)).predict(self.batch)
        with self.assertRaises(TypeError):
            result.results["other"] = result["health"]


if __name__ == "__main__":
    unittest.main()
