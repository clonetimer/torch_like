"""Fault source mapping is explicit, many-to-many, immutable and non-causal."""

import unittest

import numpy as np

from torch_like.diagnostics.fault_sources import (
    FaultSourceEvidenceMapper,
    FaultSourceRulebook,
    register_fault_source_mapper,
)
from torch_like.diagnostics.registry import DiagnosticRegistry
from torch_like.diagnostics.types import DiagnosticResult


def rulebook():
    return FaultSourceRulebook(
        class_labels=("normal", "overheat", "supply_fault"),
        source_names=("fan", "sensor", "power_bus"),
        fault_to_sources={
            "normal": (),
            "overheat": ("fan", "sensor"),
            "supply_fault": ("sensor", "power_bus"),
        },
        version="reviewed-v1",
    )


class FaultSourceTests(unittest.TestCase):
    def test_many_to_many_mapping_preserves_ambiguity_not_causal_probability(self):
        classes = DiagnosticResult(
            "fault_classification",
            "class_probability",
            np.array([[0.1, 0.6, 0.3], [0.9, 0.05, 0.05]]),
            "classifier",
            label_names=("normal", "overheat", "supply_fault"),
        )
        mapped = FaultSourceEvidenceMapper(rulebook()).predict(classes)
        self.assertEqual(mapped.task, "fault_isolation")
        self.assertEqual(mapped.output_kind, "fault_source_score")
        self.assertEqual(mapped.label_names, ("fan", "sensor", "power_bus"))
        np.testing.assert_allclose(mapped.values, [[0.6, 0.6, 0.3], [0.05, 0.05, 0.05]])
        self.assertGreater(mapped.values[0].sum(), 1.0)
        self.assertFalse(mapped.values.flags.writeable)

    def test_rulebook_snapshots_mapping_and_rejects_mutation(self):
        raw = {"normal": (), "overheat": ("fan", "sensor"), "supply_fault": ("sensor", "power_bus")}
        record = FaultSourceRulebook(
            class_labels=("normal", "overheat", "supply_fault"),
            source_names=("fan", "sensor", "power_bus"),
            fault_to_sources=raw,
        )
        raw["overheat"] = ("power_bus",)
        self.assertEqual(record.fault_to_sources["overheat"], ("fan", "sensor"))
        with self.assertRaises(TypeError):
            record.fault_to_sources["normal"] = ("fan",)

    def test_label_schema_order_must_match_exactly(self):
        values = np.array([[0.1, 0.6, 0.3]])
        wrong = DiagnosticResult(
            "fault_classification",
            "class_probability",
            values,
            "cls",
            label_names=("normal", "supply_fault", "overheat"),
        )
        with self.assertRaisesRegex(ValueError, "label ordering"):
            FaultSourceEvidenceMapper(rulebook()).predict(wrong)

    def test_wrong_task_nonprobability_and_non_normalized_scores_rejected(self):
        mapper = FaultSourceEvidenceMapper(rulebook())
        with self.assertRaisesRegex(ValueError, "class_probability"):
            mapper.predict(DiagnosticResult("anomaly_detection", "anomaly_score", np.array([0.9]), "cls"))
        invalid = DiagnosticResult(
            "fault_classification",
            "class_probability",
            np.array([[0.2, 0.2, 0.2]]),
            "cls",
            label_names=rulebook().class_labels,
        )
        with self.assertRaisesRegex(ValueError, "normalized"):
            mapper.predict(invalid)

    def test_unknown_missing_source_and_normal_mapping_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly one mapping"):
            FaultSourceRulebook(("normal", "fault"), ("a",), {"normal": ()})
        with self.assertRaisesRegex(ValueError, "normal_label"):
            FaultSourceRulebook(("normal", "fault"), ("a",), {"normal": ("a",), "fault": ("a",)})
        with self.assertRaisesRegex(ValueError, "unknown"):
            FaultSourceRulebook(("normal", "fault"), ("a",), {"normal": (), "fault": ("unknown",)})
        with self.assertRaisesRegex(ValueError, "candidate source"):
            FaultSourceRulebook(("normal", "fault"), ("a",), {"normal": (), "fault": ()})
        with self.assertRaisesRegex(ValueError, "associated"):
            FaultSourceRulebook(("normal", "fault"), ("a", "b"), {"normal": (), "fault": ("a",)})

    def test_native_fault_classifier_to_source_evidence(self):
        """The R4a probability output feeds R4b without causal semantics."""
        from torch_like.diagnostics.classification import FaultLabelSchema, SklearnWindowFaultClassifier
        from torch_like.diagnostics.types import SignalBatch

        schema = FaultLabelSchema(rulebook().class_labels)
        reference = SignalBatch(
            np.array([[0.0] * 8, [0.1] * 8, [2.0] * 8, [2.1] * 8, [4.0] * 8, [4.1] * 8])[:, :, None],
            sample_rate_hz=80.0,
            channel_names=("temperature",),
            units=("C",),
        )
        labels = ("normal", "normal", "overheat", "overheat", "supply_fault", "supply_fault")
        classifier = SklearnWindowFaultClassifier(n_estimators=12, random_state=11).fit(reference, labels, schema)
        queries = SignalBatch(
            np.array([[0.05] * 8, [3.9] * 8])[:, :, None],
            sample_rate_hz=80.0,
            channel_names=("temperature",),
            units=("C",),
        )
        probabilities = classifier.predict_proba(queries)
        sources = FaultSourceEvidenceMapper(rulebook()).predict(probabilities)
        self.assertEqual(sources.label_names, ("fan", "sensor", "power_bus"))
        self.assertEqual(sources.values.shape, (2, 3))
        np.testing.assert_allclose(sources.values[:, 0], probabilities.values[:, 1])
        np.testing.assert_allclose(
            sources.values[:, 1],
            np.maximum(probabilities.values[:, 1], probabilities.values[:, 2]),
        )
        np.testing.assert_allclose(sources.values[:, 2], probabilities.values[:, 2])
        self.assertEqual(sources.output_kind, "fault_source_score")

    def test_registry_creates_mapping_with_explicit_rulebook(self):
        registry = DiagnosticRegistry()
        register_fault_source_mapper(registry)
        native = registry.create("FaultSourceEvidenceMapper", rulebook=rulebook())
        self.assertIsInstance(native, FaultSourceEvidenceMapper)
        self.assertEqual(registry.get_spec("FaultSourceEvidenceMapper").task, "fault_isolation")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_fault_source_mapper(registry)


if __name__ == "__main__":
    unittest.main()
