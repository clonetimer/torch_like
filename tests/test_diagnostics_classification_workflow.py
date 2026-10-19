"""R4a: recording-disjoint fault-type classification and sealed test evaluation."""
import unittest

import numpy as np

from torch_like.diagnostics.classification import FaultLabelSchema, SklearnWindowFaultClassifier
from torch_like.diagnostics.classification_workflow import (
    FaultClassificationEvaluation,
    FaultClassificationPartition,
    FaultClassificationSplit,
    FaultClassificationWorkflow,
)
from torch_like.diagnostics.types import DiagnosticResult, SignalBatch


SCHEMA = FaultLabelSchema(("normal", "bearing", "imbalance"))


def make_partition(rows, labels, *, recording, prefix, rate=100, channel="vibration"):
    windows = np.asarray(rows, dtype=float)[:, :, None]
    data = SignalBatch(
        windows,
        sample_rate_hz=rate,
        channel_names=(channel,),
        units=("g",),
        sample_ids=tuple(f"{prefix}:{i}" for i in range(len(windows))),
    )
    return FaultClassificationPartition(data, (recording,) * len(windows), tuple(labels))


def make_split():
    train = make_partition(
        [[0] * 8, [0.1] * 8, [1] * 8, [1.1] * 8, [3] * 8, [3.1] * 8],
        ["normal", "normal", "bearing", "bearing", "imbalance", "imbalance"],
        recording="dev_recording_A",
        prefix="train",
    )
    val = make_partition(
        [[0] * 8, [1] * 8, [3] * 8],
        ["normal", "bearing", "imbalance"],
        recording="dev_recording_B",
        prefix="val",
    )
    test = make_partition(
        [[0] * 8, [1] * 8, [3] * 8, [3] * 8],
        ["normal", "bearing", "imbalance", "bearing"],
        recording="sealed_device_C",
        prefix="test",
    )
    return FaultClassificationSplit(SCHEMA, train, test, val)


class SpyClassifier:
    def __init__(self):
        self.fit_calls = 0
        self.test_calls = 0

    def fit(self, reference, labels, schema):
        self.fit_calls += 1
        self.train_batch = reference
        self.train_labels = tuple(labels)
        self.schema = schema
        return self

    def predict_proba(self, signals):
        self.test_calls += 1
        means = signals.values.mean(axis=(1, 2))
        predicted = np.where(means < 0.5, 0, np.where(means < 2, 1, 2))
        values = np.eye(3, dtype=float)[predicted]
        return DiagnosticResult(
            "fault_classification", "class_probability", values, "SpyClassifier", label_names=self.schema.labels
        )


class FaultClassificationWorkflowTests(unittest.TestCase):
    def test_confusion_macro_f1_and_one_shot_sealed_evaluation(self):
        classifier = SpyClassifier()
        workflow = FaultClassificationWorkflow(classifier)
        split = make_split()
        result = workflow.run(split)
        self.assertEqual(classifier.fit_calls, 1)
        self.assertEqual(classifier.test_calls, 1)
        self.assertIs(classifier.train_batch, split.train.signals)
        self.assertEqual(classifier.train_labels, split.train.labels)
        self.assertEqual(result.predictions, ("normal", "bearing", "imbalance", "imbalance"))
        np.testing.assert_array_equal(result.confusion, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        self.assertAlmostEqual(result.accuracy, 0.75)
        self.assertAlmostEqual(result.macro_f1, (1 + 2 / 3 + 2 / 3) / 3)
        self.assertFalse(result.confusion.flags.writeable)
        self.assertFalse(result.probabilities.flags.writeable)
        serialized = result.to_dict()
        self.assertEqual(serialized["metric_level"], "window")
        self.assertEqual(serialized["support"], [1, 2, 1])
        self.assertEqual(serialized["sample_ids"], ["test:0", "test:1", "test:2", "test:3"])
        self.assertNotIn("true_labels", serialized)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            workflow.run(split)

    def test_native_random_forest_end_to_end(self):
        split = make_split()
        model = SklearnWindowFaultClassifier(n_estimators=16, random_state=1)
        result = FaultClassificationWorkflow(model).run(split)
        self.assertEqual(result.probabilities.shape, (4, 3))
        self.assertEqual(result.class_labels, SCHEMA.labels)
        self.assertGreaterEqual(result.accuracy, 0)
        self.assertLessEqual(result.accuracy, 1)
        self.assertEqual(result.confusion.sum(), 4)
        self.assertEqual(model.estimator.__class__.__name__, "RandomForestClassifier")

    def test_recording_overlap_rejected_even_when_samples_renamed(self):
        split = make_split()
        collision = make_partition([[0] * 8], ["normal"], recording="dev_recording_A", prefix="renamed")
        with self.assertRaisesRegex(ValueError, "recording_ids"):
            FaultClassificationSplit(SCHEMA, split.train, collision)

    def test_sample_overlap_rejected_when_recording_changes(self):
        split = make_split()
        duplicate = FaultClassificationPartition(
            split.train.signals,
            ("different_group",) * split.train.signals.shape[0],
            split.train.labels,
        )
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            FaultClassificationSplit(SCHEMA, split.train, duplicate)

    def test_schema_disallows_unseen_sealed_class_and_unrepresented_train_class(self):
        split = make_split()
        unknown = make_partition([[5] * 8], ["unknown"], recording="novel", prefix="novel")
        with self.assertRaisesRegex(ValueError, "predefined"):
            FaultClassificationSplit(SCHEMA, split.train, unknown)
        train = make_partition([[0] * 8, [1] * 8], ["normal", "bearing"], recording="dev", prefix="dev")
        with self.assertRaisesRegex(ValueError, "all declared"):
            FaultClassificationSplit(SCHEMA, train, split.test)

    def test_validation_is_reserved_and_not_used_during_final_scoring(self):
        split = make_split()
        classifier = SpyClassifier()
        result = FaultClassificationWorkflow(classifier).run(split)
        self.assertEqual(classifier.test_calls, 1)
        self.assertEqual(result.recording_ids, ("sealed_device_C",) * 4)
        self.assertNotIn("dev_recording_B", result.recording_ids)

    def test_partitions_reject_missing_sample_provenance_and_label_misalignment(self):
        missing = SignalBatch(np.ones((2, 8, 1)), sample_rate_hz=100)
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            FaultClassificationPartition(missing, ("r", "r"), ("normal", "normal"))
        sample = make_split().test.signals
        with self.assertRaisesRegex(ValueError, "labels"):
            FaultClassificationPartition(sample, ("run",) * 4, ("normal",))
        with self.assertRaisesRegex(ValueError, "recording_ids"):
            FaultClassificationPartition(sample, ("run",), ("normal",) * 4)

    def test_bad_prediction_outputs_rejected_after_single_sealed_call(self):
        class WrongKind(SpyClassifier):
            def predict_proba(self, signals):
                return DiagnosticResult(
                    "anomaly_detection", "anomaly_score",
                    np.zeros(signals.shape[0]), "bad", threshold=0.5
                )

        with self.assertRaisesRegex(ValueError, "fault_classification"):
            FaultClassificationWorkflow(WrongKind()).run(make_split())

        class SwappedNames(SpyClassifier):
            def predict_proba(self, signals):
                return DiagnosticResult(
                    "fault_classification", "class_probability",
                    np.tile([0.4, 0.3, 0.3], (signals.shape[0], 1)),
                    "bad", label_names=("bearing", "normal", "imbalance")
                )

        with self.assertRaisesRegex(ValueError, "class ordering"):
            FaultClassificationWorkflow(SwappedNames()).run(make_split())

    def test_evaluation_rejects_incorrect_confusion_and_prediction_counts(self):
        with self.assertRaisesRegex(ValueError, "confusion"):
            FaultClassificationEvaluation(
                "model", ("normal", "fault"), ("id",), ("recording",),
                ("normal",), np.array([[0.5, 0.5]]), np.zeros((2, 2)),
            )

    def test_workflow_requires_fit_and_predict_proba(self):
        with self.assertRaisesRegex(TypeError, "fit and predict_proba"):
            FaultClassificationWorkflow(object())
        with self.assertRaisesRegex(TypeError, "FaultClassificationSplit"):
            FaultClassificationWorkflow(SpyClassifier()).run("invalid")


if __name__ == "__main__":
    unittest.main()
