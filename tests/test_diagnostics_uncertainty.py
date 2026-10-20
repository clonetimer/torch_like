"""R4f: whole-source cluster bootstrap from one already sealed R4c evaluation."""

import json
import unittest
from dataclasses import replace

import numpy as np

from torch_like.diagnostics.classification import FaultLabelSchema
from torch_like.diagnostics.classification_workflow import FaultClassificationPartition
from torch_like.diagnostics.generalization import (
    FaultDomainPartition,
    FaultGeneralizationSplit,
    FaultGeneralizationWorkflow,
)
from torch_like.diagnostics.types import DiagnosticResult, SignalBatch
from torch_like.diagnostics.uncertainty import bootstrap_fault_generalization

LABELS = ("normal", "bearing", "imbalance")
SCHEMA = FaultLabelSchema(LABELS)


class FixedClassifier:
    def __init__(self):
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, signals, labels, schema):
        self.fit_calls += 1
        self.schema = schema
        return self

    def predict_proba(self, signals):
        self.predict_calls += 1
        predictions = signals.values[:, 0, 0].astype(int)
        return DiagnosticResult(
            "fault_classification",
            "class_probability",
            np.eye(len(self.schema.labels))[predictions],
            "FixedClassifier",
            label_names=self.schema.labels,
        )


def make_domain(prefix, recordings):
    """(recording, device, condition, [(true_class, predicted_class), ...])."""
    xs, labels, runs, devices, conditions, sample_ids = [], [], [], [], [], []
    for run, device, condition, pairs in recordings:
        for truth, predicted in pairs:
            sample_ids.append(f"{prefix}:{run}:{len(xs)}")
            xs.append(np.full((5, 1), float(predicted)))
            labels.append(LABELS[truth])
            runs.append(run)
            devices.append(device)
            conditions.append(condition)
    data = SignalBatch(
        np.asarray(xs, dtype=float),
        sample_rate_hz=50,
        channel_names=("vibration",),
        units=("g",),
        sample_ids=tuple(sample_ids),
    )
    return FaultDomainPartition(
        FaultClassificationPartition(data, tuple(runs), tuple(labels)),
        tuple(devices),
        tuple(conditions),
    )


def make_sealed(*, test_runs=None, holdout="both"):
    train = make_domain(
        "train",
        [("train-run", "train-device", "train-condition", [(0, 0), (1, 1), (2, 2)])],
    )
    if test_runs is None:
        test_runs = [
            ("r0", "D0", "hot0", [(0, 0), (1, 1), (2, 2)]),
            ("r1", "D0", "hot0", [(0, 0), (1, 0), (2, 0)]),
            ("r2", "D1", "hot1", [(0, 0), (1, 1), (2, 2)]),
            ("r3", "D1", "hot1", [(0, 0), (1, 1), (2, 2)]),
            ("r4", "D2", "hot2", [(0, 0), (1, 0), (2, 0)]),
            ("r5", "D2", "hot2", [(0, 0), (1, 1), (2, 2)]),
        ]
    test = make_domain("test", test_runs)
    classifier = FixedClassifier()
    evaluation = FaultGeneralizationWorkflow(classifier).run(
        FaultGeneralizationSplit(SCHEMA, train, test, holdout=holdout)
    )
    return evaluation, classifier


class ClusterUncertaintyTests(unittest.TestCase):
    def test_recording_cluster_repeatable_and_no_additional_model_calls(self):
        sealed, classifier = make_sealed()
        before = (classifier.fit_calls, classifier.predict_calls)
        a = bootstrap_fault_generalization(sealed, seed=31, n_resamples=600)
        b = bootstrap_fault_generalization(sealed, seed=31, n_resamples=600)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual((classifier.fit_calls, classifier.predict_calls), before)
        report = a.to_dict()
        self.assertEqual(report["protocol"], "sealed_posthoc_cluster_percentile_bootstrap")
        self.assertEqual(report["num_clusters"], 6)
        self.assertEqual(report["num_devices"], 3)
        self.assertEqual(report["num_recordings"], 6)
        self.assertEqual(report["num_windows"], 18)
        self.assertEqual(report["class_cluster_support"], [6, 6, 6])
        self.assertAlmostEqual(report["accuracy"]["estimate"], sealed.overall.accuracy)
        self.assertAlmostEqual(report["macro_f1_all_classes"]["estimate"], sealed.overall.macro_f1)
        self.assertLess(report["accuracy"]["lower"], report["accuracy"]["upper"])
        self.assertNotIn("true_labels", json.dumps(report))
        self.assertIn("windows_with_whole_cluster_resampling", report["metric_weighting"])

    def test_device_bootstrap_with_three_genuinely_held_out_device_ids(self):
        sealed, _ = make_sealed()
        result = bootstrap_fault_generalization(sealed, cluster_by="device", seed=7, n_resamples=700)
        report = result.to_dict()
        self.assertEqual(report["cluster_by"], "device")
        self.assertEqual(report["cluster_ids"], ["D0", "D1", "D2"])
        self.assertEqual(report["class_cluster_support"], [3, 3, 3])
        self.assertEqual(report["missing_class_resamples"], {"normal": 0, "bearing": 0, "imbalance": 0})
        self.assertEqual(result.accuracy.estimate, sealed.overall.accuracy)

    def test_recording_bootstrap_instead_of_artificially_independent_windows(self):
        # One long successful original recording must remain one cluster.
        records = [
            ("long", "D0", "hot0", [(i % 3, i % 3) for i in range(30)]),
            ("short-bad", "D1", "hot1", [(0, 1), (1, 2), (2, 0)]),
            ("short-good", "D2", "hot2", [(0, 0), (1, 1), (2, 2)]),
        ]
        sealed, _ = make_sealed(test_runs=records)
        clustered = bootstrap_fault_generalization(sealed, seed=14, n_resamples=3000)
        # Naive window bootstrap misses the between-recording variation.
        correct = np.concatenate((np.ones(30), np.zeros(3), np.ones(3)))
        rng = np.random.default_rng(14)
        naive = rng.choice(correct, size=(3000, len(correct)), replace=True).mean(axis=1)
        naive_lower = float(np.quantile(naive, 0.025))
        self.assertLess(clustered.accuracy.lower, naive_lower - 0.1)
        self.assertEqual(clustered.recording_count, 3)
        self.assertEqual(clustered.window_count, 36)

    def test_no_variation_and_absent_class_resamples_are_reported(self):
        all_correct = [(f"r{i}", f"D{i}", f"hot{i}", [(0, 0), (1, 1), (2, 2)]) for i in range(3)]
        sealed, _ = make_sealed(test_runs=all_correct)
        result = bootstrap_fault_generalization(sealed, n_resamples=200, seed=7)
        self.assertEqual(result.accuracy.lower, 1)
        self.assertEqual(result.accuracy.upper, 1)
        self.assertEqual(result.macro_f1_all_classes.lower, 1)
        self.assertEqual(result.resamples_with_any_missing_class, 0)

        sparse = [
            ("a", "D0", "hot0", [(0, 0), (1, 1)]),
            ("b", "D1", "hot1", [(0, 0), (2, 2)]),
            ("c", "D2", "hot2", [(0, 0), (1, 1), (2, 2)]),
        ]
        sealed2, _ = make_sealed(test_runs=sparse)
        result2 = bootstrap_fault_generalization(sealed2, seed=0, n_resamples=1000)
        self.assertGreater(result2.resamples_with_any_missing_class, 0)
        self.assertGreater(result2.missing_class_resamples[1], 0)
        self.assertGreater(result2.missing_class_resamples[2], 0)
        self.assertEqual(result2.class_cluster_support, (3, 2, 2))
        self.assertEqual(result2.macro_f1_all_classes.estimate, 1)

    def test_sparse_true_class_and_inadequate_clusters_fail_closed(self):
        sparse = [
            ("one", "D0", "hot0", [(0, 0), (1, 1)]),
            ("two", "D1", "hot1", [(0, 0), (1, 1)]),
            ("three", "D2", "hot2", [(0, 0), (2, 2)]),
        ]
        sealed, _ = make_sealed(test_runs=sparse)
        with self.assertRaisesRegex(ValueError, "class-bearing clusters"):
            bootstrap_fault_generalization(sealed)
        with self.assertRaisesRegex(ValueError, "source clusters"):
            bootstrap_fault_generalization(make_sealed(test_runs=sparse[:2])[0])
        with self.assertRaisesRegex(ValueError, "class-bearing clusters"):
            bootstrap_fault_generalization(make_sealed()[0], min_class_clusters=7)

    def test_do_not_confuse_recording_and_device_uncertainty(self):
        same_device = [(f"r{i}", "D0", "hot0", [(0, 0), (1, 1), (2, 2)]) for i in range(4)]
        sealed, _ = make_sealed(test_runs=same_device)
        self.assertEqual(bootstrap_fault_generalization(sealed, n_resamples=100).recording_count, 4)
        with self.assertRaisesRegex(ValueError, "source clusters"):
            bootstrap_fault_generalization(sealed, cluster_by="device", n_resamples=100)
        pair_sealed, _ = make_sealed(holdout="pair")
        with self.assertRaisesRegex(ValueError, "unseen-device"):
            bootstrap_fault_generalization(pair_sealed, cluster_by="device", n_resamples=100)

    def test_reject_tampered_slice_and_missing_or_duplicate_source_identity(self):
        sealed, _ = make_sealed()
        parts = list(sealed.slices)
        record = next(x for x in parts if x.axis == "recording")
        self.assertEqual(record.recording_count, 1)
        wrong = replace(record, confusion=np.diag([3, 0, 0]))
        tampered = replace(sealed, slices=tuple(wrong if x is record else x for x in parts))
        with self.assertRaisesRegex(ValueError, "reconciliation|predicted counts"):
            bootstrap_fault_generalization(tampered)
        duplicate = replace(sealed, slices=tuple(parts) + (record,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bootstrap_fault_generalization(duplicate)
        missing = replace(sealed, slices=tuple(x for x in parts if x.axis != "recording"))
        with self.assertRaisesRegex(ValueError, "missing recording"):
            bootstrap_fault_generalization(missing)
        renamed_overall = replace(sealed.overall, recording_ids=("aliased",) + sealed.overall.recording_ids[1:])
        with self.assertRaisesRegex(ValueError, "per-recording identities"):
            bootstrap_fault_generalization(replace(sealed, overall=renamed_overall))

    def test_reject_inconsistent_marginals_or_prediction_provenance(self):
        sealed, _ = make_sealed()
        parts = list(sealed.slices)
        device = next(x for x in parts if x.axis == "device")
        others = tuple(x for x in parts if x is not device)
        with self.assertRaisesRegex(ValueError, "device domain confusion"):
            bootstrap_fault_generalization(replace(sealed, slices=others))
        altered = replace(sealed, test_devices=("not-the-physical-device",))
        with self.assertRaisesRegex(ValueError, "device identities"):
            bootstrap_fault_generalization(altered)
        changed_preds = ("bearing",) + sealed.overall.predictions[1:]
        malformed = replace(sealed.overall)
        # Mimic tampering after a valid sealed result was created.
        object.__setattr__(malformed, "predictions", changed_preds)
        with self.assertRaisesRegex(ValueError, "predictions do not match"):
            bootstrap_fault_generalization(replace(sealed, overall=malformed))

    def test_invalid_configuration_rejected_before_resampling(self):
        sealed, _ = make_sealed()
        for kwargs in (
            {"cluster_by": "window"},
            {"n_resamples": 99},
            {"n_resamples": True},
            {"confidence": 0},
            {"confidence": 1},
            {"confidence": float("nan")},
            {"seed": -1},
            {"seed": True},
            {"min_clusters": 2},
            {"min_class_clusters": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap_fault_generalization(sealed, **kwargs)
        with self.assertRaisesRegex(TypeError, "already sealed"):
            bootstrap_fault_generalization(object())


if __name__ == "__main__":
    unittest.main()
