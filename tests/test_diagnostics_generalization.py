"""R4c: leave-device/condition-out transfer evaluation without test leakage."""

import unittest

import numpy as np

from torch_like.diagnostics.classification import FaultLabelSchema, SklearnWindowFaultClassifier
from torch_like.diagnostics.classification_workflow import FaultClassificationPartition
from torch_like.diagnostics.generalization import (
    FaultDomainPartition,
    FaultGeneralizationSplit,
    FaultGeneralizationWorkflow,
)
from torch_like.diagnostics.types import DiagnosticResult, SignalBatch

SCHEMA = FaultLabelSchema(("normal", "bearing", "imbalance"))


def make_part(values, labels, *, run, device, condition, prefix):
    signals = SignalBatch(
        np.asarray(values, dtype=float)[:, :, None],
        sample_rate_hz=100,
        channel_names=("vibration",),
        units=("g",),
        sample_ids=tuple(f"{prefix}-{i}" for i in range(len(values))),
    )
    return FaultDomainPartition(
        FaultClassificationPartition(signals, (run,) * len(values), tuple(labels)),
        (device,) * len(values),
        (condition,) * len(values),
    )


def make_split(*, holdout="both", minimum=2):
    train = make_part(
        [[0] * 8, [0.1] * 8, [1] * 8, [1.1] * 8, [3] * 8, [3.1] * 8],
        ("normal", "normal", "bearing", "bearing", "imbalance", "imbalance"),
        run="train-run", device="device-A", condition="cold", prefix="train",
    )
    validation = make_part(
        [[0] * 8, [1] * 8, [3] * 8],
        ("normal", "bearing", "imbalance"),
        run="val-run", device="device-B", condition="warm", prefix="val",
    )
    test = make_part(
        [[0] * 8, [1] * 8, [3] * 8, [1] * 8],
        ("normal", "bearing", "imbalance", "bearing"),
        run="test-run", device="device-C", condition="hot", prefix="test",
    )
    return FaultGeneralizationSplit(
        SCHEMA, train, test, validation, holdout=holdout, min_domain_windows=minimum
    )


class SpyClassifier:
    def __init__(self):
        self.fit_calls = 0
        self.test_calls = 0

    def fit(self, signals, labels, schema):
        self.fit_calls += 1
        self.trained_signals = signals
        self.trained_labels = tuple(labels)
        self.schema = schema
        return self

    def predict_proba(self, signals):
        self.test_calls += 1
        means = signals.values.mean(axis=(1, 2))
        indices = np.where(means < 0.5, 0, np.where(means < 2.0, 1, 2))
        return DiagnosticResult(
            "fault_classification",
            "class_probability",
            np.eye(3)[indices],
            "SpyClassifier",
            label_names=self.schema.labels,
        )


class GeneralizationTests(unittest.TestCase):
    def test_both_axes_one_shot_holdout_and_provenance_slices(self):
        split = make_split()
        classifier = SpyClassifier()
        workflow = FaultGeneralizationWorkflow(classifier)
        evaluation = workflow.run(split)
        self.assertEqual(classifier.fit_calls, 1)
        self.assertEqual(classifier.test_calls, 1)
        self.assertIs(classifier.trained_signals, split.train.classification.signals)
        self.assertEqual(classifier.trained_labels, split.train.classification.labels)
        self.assertEqual(evaluation.overall.accuracy, 1.0)
        report = evaluation.to_dict()
        self.assertEqual(report["holdout"], "both")
        self.assertEqual(report["train_devices"], ["device-A"])
        self.assertEqual(report["test_conditions"], ["hot"])
        self.assertEqual(len(report["domain_slices"]), 4)
        self.assertNotIn("true_labels", str(report))
        for group in report["domain_slices"]:
            self.assertEqual(group["num_windows"], 4)
            self.assertEqual(group["num_recordings"], 1)
            self.assertEqual(group["support"], [1, 2, 1])
            self.assertEqual(group["macro_f1_all_classes"], 1.0)
            self.assertTrue(group["sufficient_window_support"])
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            workflow.run(split)

    def test_native_random_forest_cross_device_path(self):
        split = make_split()
        clf = SklearnWindowFaultClassifier(n_estimators=14, random_state=7)
        result = FaultGeneralizationWorkflow(clf).run(split)
        self.assertEqual(clf.estimator.__class__.__name__, "RandomForestClassifier")
        self.assertEqual(result.overall.probabilities.shape, (4, 3))
        self.assertEqual(result.overall.confusion.sum(), 4)

    def test_device_holdout_rejects_same_device_under_renamed_run(self):
        split = make_split()
        violation = make_part(
            [[0] * 8], ("normal",),
            run="renamed-run", device="device-A", condition="hot", prefix="new",
        )
        with self.assertRaisesRegex(ValueError, "leaks device"):
            FaultGeneralizationSplit(SCHEMA, split.train, violation, holdout="device")

    def test_condition_holdout_rejects_seen_operating_condition(self):
        split = make_split()
        violation = make_part(
            [[3] * 8], ("imbalance",),
            run="new-run", device="device-Q", condition="cold", prefix="new",
        )
        with self.assertRaisesRegex(ValueError, "leaks condition"):
            FaultGeneralizationSplit(SCHEMA, split.train, violation, holdout="condition")

    def test_validation_also_cannot_overlap_target_domain(self):
        split = make_split()
        val = make_part(
            [[0] * 8], ("normal",),
            run="different-run", device="device-A", condition="warm", prefix="val-new",
        )
        with self.assertRaisesRegex(ValueError, "leaks device"):
            FaultGeneralizationSplit(SCHEMA, split.train, split.test, val, holdout="device")

    def test_pair_holdout_allows_reused_individual_domain_identity(self):
        split = make_split()
        unseen_combo = make_part(
            [[0] * 8], ("normal",),
            run="combo-run", device="device-A", condition="hot", prefix="combo",
        )
        result = FaultGeneralizationSplit(SCHEMA, split.train, unseen_combo, holdout="pair")
        self.assertEqual(result.holdout, "pair")
        with self.assertRaisesRegex(ValueError, "leaks device"):
            FaultGeneralizationSplit(SCHEMA, split.train, unseen_combo, holdout="both")
        repeated_pair = make_part(
            [[0] * 8], ("normal",),
            run="another-run", device="device-A", condition="cold", prefix="repeated",
        )
        with self.assertRaisesRegex(ValueError, "leaks pair"):
            FaultGeneralizationSplit(SCHEMA, split.train, repeated_pair, holdout="pair")

    def test_context_consistency_within_recording(self):
        source = make_split().test.classification
        with self.assertRaisesRegex(ValueError, "one original recording_id"):
            FaultDomainPartition(source, ("X", "X", "Y", "X"), ("hot",) * 4)
        with self.assertRaisesRegex(ValueError, "condition_ids"):
            FaultDomainPartition(source, ("X",) * 4, ("hot",))
        with self.assertRaisesRegex(ValueError, "device_ids"):
            FaultDomainPartition(source, ("X", "", "X", "X"), ("hot",) * 4)
        with self.assertRaisesRegex(TypeError, "FaultClassificationPartition"):
            FaultDomainPartition(object(), ("X",), ("cold",))

    def test_reuse_r4a_recording_and_sample_group_disjointness(self):
        split = make_split()
        with self.assertRaisesRegex(ValueError, "recording_ids"):
            FaultGeneralizationSplit(SCHEMA, split.train, split.train, holdout="device")

    def test_invalid_mode_and_domain_support_policy(self):
        split = make_split()
        with self.assertRaisesRegex(ValueError, "holdout"):
            FaultGeneralizationSplit(SCHEMA, split.train, split.test, holdout="arbitrary")
        for minimum in (0, True, 1.3):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, "min_domain_windows"):
                FaultGeneralizationSplit(SCHEMA, split.train, split.test, min_domain_windows=minimum)

    def test_absent_true_classes_and_low_support_exposed(self):
        split = make_split()
        sparse = make_part(
            [[0] * 8, [3] * 8], ("normal", "imbalance"),
            run="sparse-sealed", device="device-C", condition="hot", prefix="sparse",
        )
        evaluated = FaultGeneralizationWorkflow(SpyClassifier()).run(
            FaultGeneralizationSplit(
                SCHEMA, split.train, sparse, split.validation,
                holdout="both", min_domain_windows=3,
            )
        )
        report = evaluated.to_dict()
        self.assertEqual(report["underpowered_domain_count"], 4)
        for item in report["domain_slices"]:
            self.assertFalse(item["sufficient_window_support"])
            self.assertEqual(item["support"], [1, 0, 1])
            self.assertEqual(item["absent_true_classes"], ["bearing"])
            self.assertAlmostEqual(item["macro_f1_all_classes"], 2 / 3)
            self.assertAlmostEqual(item["macro_f1_present_classes"], 1.0)

    def test_multi_device_test_domain_counts_do_not_pool_sources(self):
        split = make_split()
        values = [[0] * 8, [1] * 8, [3] * 8, [1] * 8]
        signal = SignalBatch(
            np.asarray(values, dtype=float)[:, :, None],
            sample_rate_hz=100,
            channel_names=("vibration",),
            units=("g",),
            sample_ids=("C0", "C1", "D0", "D1"),
        )
        test = FaultDomainPartition(
            FaultClassificationPartition(
                signal,
                ("run-C", "run-C", "run-D", "run-D"),
                ("normal", "bearing", "imbalance", "bearing"),
            ),
            ("device-C", "device-C", "device-D", "device-D"),
            ("hot",) * 4,
        )
        result = FaultGeneralizationWorkflow(SpyClassifier()).run(
            FaultGeneralizationSplit(SCHEMA, split.train, test, split.validation, holdout="both")
        )
        domains = [x for x in result.to_dict()["domain_slices"] if x["axis"] == "device"]
        self.assertEqual(
            [(x["identity"], x["num_windows"]) for x in domains],
            [(["device-C"], 2), (["device-D"], 2)],
        )
        self.assertEqual(sum(group["num_windows"] for group in domains), 4)

    def test_workflow_rejects_bad_interfaces(self):
        with self.assertRaisesRegex(TypeError, "fit and predict_proba"):
            FaultGeneralizationWorkflow(object())
        with self.assertRaisesRegex(TypeError, "FaultGeneralizationSplit"):
            FaultGeneralizationWorkflow(SpyClassifier()).run(object())


if __name__ == "__main__":
    unittest.main()
