"""R4e: candidate comparison uses validation alone, before a single sealed test."""

import json
import unittest
from dataclasses import replace

import numpy as np

from torch_like.diagnostics import (
    FaultCandidateSpec,
    FaultClassificationPartition,
    FaultClassificationSplit,
    FaultDomainPartition,
    FaultGeneralizationSplit,
    FaultLabelSchema,
    FaultSelectionWorkflow,
)
from torch_like.diagnostics.types import DiagnosticResult, SignalBatch

SCHEMA = FaultLabelSchema(("normal", "bearing", "imbalance"))


def make_part(prefix, *, values=(0, 1, 2, 0, 1, 2), device="A", condition="cold"):
    values = tuple(values)
    signals = SignalBatch(
        np.array([np.full((8, 1), float(v)) for v in values]),
        sample_rate_hz=100,
        channel_names=("vibration",),
        units=("g",),
        sample_ids=tuple(f"{prefix}:{i}" for i in range(len(values))),
    )
    labels = tuple(SCHEMA.labels[int(value) % 3] for value in values)
    return FaultDomainPartition(
        FaultClassificationPartition(signals, (prefix,) * len(values), labels),
        (device,) * len(values),
        (condition,) * len(values),
    )


def make_split(*, domain=False):
    train = make_part("train", device="A", condition="cold")
    validation = make_part("val", device="B", condition="warm")
    test = make_part("sealed", values=(0, 1, 2), device="C", condition="hot")
    if domain:
        return FaultGeneralizationSplit(SCHEMA, train, test, validation, holdout="both")
    return FaultClassificationSplit(SCHEMA, train.classification, test.classification, validation.classification)


def spec(identifier, *, seeds=(3, 5), backend="sklearn", algorithm="RandomForestClassifier", **kwargs):
    return FaultCandidateSpec(
        identifier, backend, algorithm, seeds,
        model_parameters=tuple(kwargs.items()), budget_unit="trees", budget_value=kwargs.get("n_estimators", 100)
    )


class SpyClassifier:
    def __init__(self, identifier, seed, calls, *, fail=False):
        self.identifier = identifier
        self.seed = seed
        self.calls = calls
        self.fail = fail
        self.schema = None

    def fit(self, signals, labels, schema):
        self.calls.append(("fit", self.identifier, self.seed, signals.sample_ids, tuple(labels)))
        self.schema = schema
        if self.fail:
            raise ValueError("simulated training failure")
        return self

    def predict_proba(self, signals):
        self.calls.append(("predict", self.identifier, self.seed, signals.sample_ids, None))
        pred = signals.values.mean(axis=(1, 2)).astype(int)
        if self.identifier == "bad":
            pred = np.zeros_like(pred)
        return DiagnosticResult(
            task="fault_classification",
            output_kind="class_probability",
            values=np.eye(len(self.schema.labels))[pred],
            algorithm_name="SpyClassifier",
            label_names=self.schema.labels,
        )


class FaultSelectionTests(unittest.TestCase):
    def test_r4a_validation_ranks_candidate_before_single_sealed_prediction(self):
        calls = []

        def factory(candidate, seed):
            return SpyClassifier(candidate.identifier, seed, calls)

        selection = FaultSelectionWorkflow((spec("bad"), spec("good")), factory=factory)
        result = selection.run(make_split())
        report = result.selection.to_dict()
        self.assertEqual(report["selected_candidate"], "good")
        self.assertEqual(report["selected_seed"], 3)
        self.assertEqual([trial["seed"] for trial in report["seed_results"]], [3, 5, 3, 5])
        self.assertEqual(len([c for c in calls if c[0] == "predict" and c[3][0].startswith("val:")]), 4)
        self.assertEqual(len([c for c in calls if c[0] == "predict" and c[3][0].startswith("sealed:")]), 1)
        self.assertEqual(len([c for c in calls if c[0] == "fit"]), 5)
        self.assertEqual(result.sealed.accuracy, 1.0)
        self.assertEqual(report["validation_support"], [2, 2, 2])
        self.assertEqual(report["validation_recordings"], 1)
        self.assertNotIn("sealed", json.dumps(report))
        self.assertEqual(report["rankings"][0]["mean_validation_score"], 1 / 6)
        self.assertEqual(report["rankings"][1]["mean_validation_score"], 1)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            selection.run(make_split())

    def test_validation_selection_preserves_r4c_domain_holdout(self):
        calls = []
        result = FaultSelectionWorkflow(
            (spec("good"),),
            factory=lambda candidate, seed: SpyClassifier(candidate.identifier, seed, calls),
        ).run(make_split(domain=True))
        self.assertEqual(result.sealed.holdout, "both")
        self.assertEqual(result.sealed.overall.confusion.sum(), 3)
        report = result.selection.to_dict()
        self.assertEqual(len(report["validation_domains"]), 3)
        self.assertTrue(all(part["recordings"] == 1 for part in report["validation_domains"]))
        self.assertEqual(report["selected_candidate"], "good")
        self.assertNotIn("sealed:", json.dumps(report))
        self.assertEqual(len([c for c in calls if c[0] == "predict" and c[3][0].startswith("sealed:")]), 1)

    def test_missing_validation_fails_without_test_inference(self):
        calls = []
        split = replace(make_split(), validation=None)
        selector = FaultSelectionWorkflow(
            (spec("good"),),
            factory=lambda candidate, seed: SpyClassifier(candidate.identifier, seed, calls),
        )
        with self.assertRaisesRegex(ValueError, "requires an independent validation"):
            selector.run(split)
        self.assertEqual(calls, [])

    def test_insufficient_class_recording_and_domain_support_fail_closed(self):
        cls_split = make_split()
        with self.assertRaisesRegex(ValueError, "validation class support"):
            FaultSelectionWorkflow((spec("good"),), min_class_windows=3).run(cls_split)
        with self.assertRaisesRegex(ValueError, "independent validation recordings"):
            FaultSelectionWorkflow((spec("good"),), min_validation_recordings=2).run(cls_split)
        with self.assertRaisesRegex(ValueError, "validation recordings for"):
            FaultSelectionWorkflow((spec("good"),), min_domain_recordings=2).run(make_split(domain=True))

    def test_bad_candidate_isolated_and_partially_failed_candidate_ineligible(self):
        calls = []

        def factory(candidate, seed):
            return SpyClassifier(candidate.identifier, seed, calls, fail=candidate.identifier == "bad" and seed == 5)

        evaluated = FaultSelectionWorkflow((spec("bad"), spec("good")), factory=factory).run(make_split())
        report = evaluated.selection.to_dict()
        self.assertEqual(report["selected_candidate"], "good")
        self.assertFalse(report["rankings"][0]["eligible"])
        self.assertIsNone(report["rankings"][0]["mean_validation_score"])
        self.assertEqual(report["rankings"][0]["successful_seeds"], 1)
        self.assertEqual(report["seed_results"][1]["failure_type"], "ValueError")
        self.assertEqual(report["seed_results"][1]["failure_message"], "simulated training failure")
        self.assertEqual(len([c for c in calls if c[0] == "predict" and c[3][0].startswith("sealed:")]), 1)

    def test_zero_valid_candidates_refuse_to_open_sealed_test(self):
        calls = []
        selector = FaultSelectionWorkflow(
            (spec("bad"),),
            factory=lambda candidate, seed: SpyClassifier(candidate.identifier, seed, calls, fail=True),
        )
        with self.assertRaisesRegex(RuntimeError, "no fully validated candidate"):
            selector.run(make_split())
        self.assertFalse(any(c[0] == "predict" for c in calls))
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            selector.run(make_split())

    def test_explicit_tie_break_is_deterministic_and_not_best_seed(self):
        calls = []
        selection = FaultSelectionWorkflow(
            (spec("zulu", seeds=(8, 2)), spec("alpha", seeds=(7, 1))),
            metric="accuracy",
            factory=lambda candidate, seed: SpyClassifier(candidate.identifier, seed, calls),
        )
        report = selection.run(make_split()).selection.to_dict()
        self.assertEqual(report["selected_candidate"], "alpha")
        self.assertEqual(report["selected_seed"], 7)
        self.assertEqual(report["metric"], "accuracy")
        self.assertIn("lower std", report["selection_rule"])

    def test_partition_and_domain_leakage_checked_before_ranking(self):
        cls = make_split()
        with self.assertRaisesRegex(ValueError, "recording_ids"):
            FaultClassificationSplit(SCHEMA, cls.train, cls.train, cls.validation)
        duplicate = FaultClassificationPartition(
            cls.test.signals, ("renamed",) * 3, cls.test.labels
        )
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            FaultClassificationSplit(SCHEMA, cls.train, duplicate, cls.validation)
        dom = make_split(domain=True)
        with self.assertRaisesRegex(ValueError, "leaks device"):
            FaultGeneralizationSplit(SCHEMA, dom.train, dom.test, dom.train, holdout="device")
        with self.assertRaisesRegex(ValueError, "leaks pair"):
            FaultGeneralizationSplit(SCHEMA, dom.train, dom.test, dom.train, holdout="pair")

    def test_candidate_specs_immutable_and_budget_audited(self):
        original = [("n_estimators", 5)]
        candidate = FaultCandidateSpec(
            "rf", "sklearn", "RandomForestClassifier", (0, 2),
            model_parameters=tuple(original), budget_unit="trees", budget_value=5
        )
        original.append(("max_depth", 4))
        self.assertEqual(dict(candidate.model_parameters), {"n_estimators": 5})
        with self.assertRaisesRegex(ValueError, "budget_value"):
            replace(candidate, budget_value=10)
        with self.assertRaisesRegex(ValueError, "seed"):
            replace(candidate, seeds=(0, 0))
        with self.assertRaisesRegex(ValueError, "must be unique"):
            replace(candidate, model_parameters=(("n_estimators", 5), ("n_estimators", 5)))
        with self.assertRaisesRegex(ValueError, "immutable"):
            replace(candidate, model_parameters=(("foo", {"unsafe": 1}),))
        with self.assertRaisesRegex(ValueError, "model identity"):
            replace(candidate, model_parameters=(("random_state", 3),))
        with self.assertRaisesRegex(ValueError, "probability=True"):
            FaultCandidateSpec("svc", "sklearn", "SVC", (1, 2),
                               model_parameters=(("probability", False),), budget_unit="fits", budget_value=1)
        with self.assertRaisesRegex(ValueError, "candidate identifiers"):
            FaultSelectionWorkflow((candidate, candidate))
        with self.assertRaisesRegex(ValueError, "minimum support"):
            FaultSelectionWorkflow((candidate,), min_class_windows=0)

    def test_native_sklearn_rf_two_seeds_then_fresh_final_model(self):
        candidate = FaultCandidateSpec(
            "rf", "sklearn", "RandomForestClassifier", (3, 11),
            model_parameters=(("n_estimators", 8),), budget_unit="trees", budget_value=8
        )
        workflow = FaultSelectionWorkflow((candidate,))
        result = workflow.run(make_split())
        self.assertEqual(len(result.selection.seed_results), 2)
        self.assertEqual(result.sealed.probabilities.shape, (3, 3))
        self.assertEqual(result.selection.selected_seed, 3)

    def test_native_torch_inception_small_end_to_end(self):
        candidate = FaultCandidateSpec(
            "inception", "torch_like", "InceptionTime", (0, 1),
            model_parameters=(("d_model", 8), ("num_blocks", 1), ("bottleneck_dim", 4),
                              ("kernel_sizes", (3,)), ("dropout", 0)),
            training_parameters=(("epochs", 1), ("batch_size", 3)),
            budget_unit="epochs", budget_value=1,
        )
        result = FaultSelectionWorkflow((candidate,)).run(make_split(domain=True))
        self.assertEqual(result.sealed.overall.probabilities.shape, (3, 3))
        self.assertEqual(result.selection.selected_seed, 0)
        self.assertEqual(len(result.selection.seed_results), 2)

    def test_invalid_interface_and_metric_rejected(self):
        with self.assertRaisesRegex(ValueError, "metric"):
            FaultSelectionWorkflow((spec("a"),), metric="test_f1")
        with self.assertRaisesRegex(TypeError, "validated classification"):
            FaultSelectionWorkflow((spec("a"),)).run("invalid")
        with self.assertRaisesRegex(TypeError, "factory"):
            FaultSelectionWorkflow((spec("a"),), factory=object())


if __name__ == "__main__":
    unittest.main()
