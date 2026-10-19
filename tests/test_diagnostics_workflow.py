"""Leakage-safe whole-window diagnostic workflow contract and integration checks."""

import unittest

import numpy as np

from torch_like.diagnostics import (
    DiagnosticResult,
    SignalBatch,
    SklearnWindowAnomalyDetector,
    ZScoreDetector,
)
from torch_like.diagnostics.workflow import (
    AnomalyDiagnosticWorkflow,
    AnomalyPartition,
    AnomalySplit,
)


def partition(samples, *, group, prefix, labels=None, rate=100):
    x = np.asarray(samples, dtype=float)
    if x.ndim == 2:
        x = x[:, :, None]
    signals = SignalBatch(
        x,
        sample_rate_hz=rate,
        channel_names=("motor",) if x.shape[-1] == 1 else ("motor", "torque"),
        units=("A",) if x.shape[-1] == 1 else ("A", "N.m"),
        sample_ids=tuple(f"{prefix}-{i}" for i in range(x.shape[0])),
    )
    return AnomalyPartition(
        signals=signals,
        recording_ids=(group,) * x.shape[0],
        labels=None if labels is None else np.asarray(labels),
    )


class CountingDetector:
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.fit_calls = 0
        self.score_calls = 0

    def fit(self, batch):
        self.fit_calls += 1
        self.reference_mean = float(np.mean(batch.values))
        return self

    def score(self, batch):
        self.score_calls += 1
        score = np.mean(batch.values, axis=(1, 2)) - self.reference_mean
        return DiagnosticResult("anomaly_detection", "anomaly_score", score, "CountingDetector", self.threshold)


class BadDetector(CountingDetector):
    def score(self, batch):
        return DiagnosticResult("rul_regression", "rul_value", np.ones(batch.shape[0]), "wrong")


class AnomalyWorkflowTests(unittest.TestCase):
    def get_split(self):
        reference = partition([[-1, 1, -1, 1], [1, -1, 1, -1], [-1, -1, 1, 1]], group="healthy_run", prefix="ref")
        validation = partition([[0, 0, 0, 0], [1, 1, 1, 1], [3, 3, 3, 3], [4, 4, 4, 4]], group="val_run", prefix="val", labels=[0, 0, 1, 1])
        test = partition([[0, 0, 0, 0], [5, 5, 5, 5]], group="sealed_run", prefix="test", labels=[0, 1])
        return AnomalySplit(reference=reference, validation=validation, test=test)

    def test_native_threshold_end_to_end_zscore_and_once_only(self):
        detector = ZScoreDetector(sigma_threshold=2.0)
        workflow = AnomalyDiagnosticWorkflow(detector)
        report = workflow.run(self.get_split())
        self.assertEqual((report.tp, report.tn, report.fp, report.fn), (1, 1, 0, 0))
        self.assertEqual(report.threshold, 2.0)
        self.assertEqual(report.threshold_source, "native")
        self.assertEqual(report.f1, 1.0)
        self.assertEqual(report.false_alarm_rate, 0.0)
        self.assertEqual(report.sample_ids, ("test-0", "test-1"))
        self.assertFalse(report.scores.flags.writeable)
        self.assertEqual(report.to_dict()["predictions"], [False, True])
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            workflow.run(self.get_split())

    def test_validation_f1_uses_development_only_for_threshold(self):
        detector = CountingDetector(threshold=1000)
        workflow = AnomalyDiagnosticWorkflow(detector, threshold_policy="validation_f1")
        report = workflow.run(self.get_split())
        self.assertEqual(detector.fit_calls, 1)
        self.assertEqual(detector.score_calls, 2)
        self.assertEqual((report.tp, report.tn, report.fp, report.fn), (1, 1, 0, 0))
        self.assertEqual(report.threshold_source, "validation_f1")
        self.assertGreater(report.threshold, 1)
        self.assertLess(report.threshold, 3)
        self.assertEqual(detector.threshold, 1000)

    def test_fitted_detector_with_native_sklearn_adapter(self):
        rng = np.random.default_rng(7)
        reference = partition(rng.normal(size=(16, 16, 1)), group="machine_A", prefix="a")
        test = partition(
            np.concatenate([rng.normal(size=(3, 16, 1)), rng.normal(5.0, 0.1, (3, 16, 1))]),
            group="machine_B",
            prefix="b",
            labels=[0, 0, 0, 1, 1, 1],
        )
        workflow = AnomalyDiagnosticWorkflow(SklearnWindowAnomalyDetector(n_estimators=16, random_state=7))
        evaluation = workflow.run(AnomalySplit(reference, test))
        self.assertEqual(evaluation.threshold, 0.0)
        self.assertEqual(evaluation.algorithm_name, "SklearnWindowAnomalyDetector")
        self.assertEqual(len(evaluation.scores), 6)
        self.assertEqual(evaluation.tp + evaluation.tn + evaluation.fp + evaluation.fn, 6)

    def test_reject_duplicate_recording_even_with_unique_window_ids(self):
        split = self.get_split()
        reused = partition([[0, 0, 0, 0]], group="healthy_run", prefix="unseen", labels=[0])
        with self.assertRaisesRegex(ValueError, "recording_ids"):
            AnomalySplit(reference=split.reference, validation=split.validation, test=reused)

    def test_reject_shared_sample_ids_even_if_recordings_are_renamed(self):
        split = self.get_split()
        duplicate = AnomalyPartition(
            signals=split.reference.signals,
            recording_ids=("different_rec",) * split.reference.signals.shape[0],
            labels=np.zeros(split.reference.signals.shape[0]),
        )
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            AnomalySplit(reference=split.reference, validation=split.validation, test=duplicate)

    def test_require_provenance_and_binary_window_labels(self):
        signals = SignalBatch(np.ones((2, 4, 1)), sample_rate_hz=1)
        with self.assertRaisesRegex(ValueError, "sample_id"):
            AnomalyPartition(signals, ("rec", "rec"))
        normal = self.get_split().reference.signals
        for labels in ([0, 2, 0], [0, np.nan, 0], [0, 1]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "labels"):
                AnomalyPartition(normal, ("r",) * 3, np.asarray(labels))
        with self.assertRaisesRegex(ValueError, "recording_ids"):
            AnomalyPartition(normal, ("", "x", "z"))
        with self.assertRaisesRegex(ValueError, "healthy-only"):
            AnomalySplit(
                reference=AnomalyPartition(normal, ("train",) * 3, labels=[0, 1, 0]),
                test=self.get_split().test,
            )

    def test_calibration_requires_labeled_two_class_validation(self):
        base = self.get_split()
        model = AnomalyDiagnosticWorkflow(CountingDetector(), threshold_policy="validation_f1")
        with self.assertRaisesRegex(ValueError, "distinct validation"):
            model.run(AnomalySplit(reference=base.reference, test=base.test))
        missing = partition([[0, 0, 0, 0], [3, 3, 3, 3]], group="val_run", prefix="vl")
        model = AnomalyDiagnosticWorkflow(CountingDetector(), threshold_policy="validation_f1")
        with self.assertRaisesRegex(ValueError, "validation labels"):
            model.run(AnomalySplit(base.reference, base.test, missing))
        all_healthy = partition([[0, 0, 0, 0], [1, 1, 1, 1]], group="val_run", prefix="vl", labels=[0, 0])
        model = AnomalyDiagnosticWorkflow(CountingDetector(), threshold_policy="validation_f1")
        with self.assertRaisesRegex(ValueError, "both healthy and anomalous"):
            model.run(AnomalySplit(base.reference, base.test, all_healthy))

    def test_sealed_labels_required_and_model_contract_validated(self):
        base = self.get_split()
        test = partition([[0, 0, 0, 0]], group="sealed", prefix="sealed")
        with self.assertRaisesRegex(ValueError, "sealed test labels"):
            AnomalyDiagnosticWorkflow(CountingDetector()).run(AnomalySplit(base.reference, test))
        with self.assertRaisesRegex(ValueError, "anomaly_detection"):
            AnomalyDiagnosticWorkflow(BadDetector()).run(base)

    def test_wrong_threshold_and_invalid_constructor(self):
        split = self.get_split()
        detector = CountingDetector(threshold=None)
        with self.assertRaisesRegex(ValueError, "native threshold"):
            AnomalyDiagnosticWorkflow(detector).run(split)
        for policy in ("", "magic", "test_optimize"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "threshold_policy"):
                AnomalyDiagnosticWorkflow(CountingDetector(), threshold_policy=policy)
        with self.assertRaisesRegex(ValueError, "threshold_candidates"):
            AnomalyDiagnosticWorkflow(CountingDetector(), threshold_candidates=1)
        with self.assertRaisesRegex(TypeError, "fit and score"):
            AnomalyDiagnosticWorkflow(object())

    def test_window_aggregation_is_explicit_max_over_time_and_channels(self):
        class ThreeDimDetector(CountingDetector):
            def score(self, batch):
                return DiagnosticResult(
                    "anomaly_detection", "anomaly_score", np.abs(batch.values),
                    "ThreeDimDetector", 2.0
                )
        result = AnomalyDiagnosticWorkflow(ThreeDimDetector()).run(self.get_split())
        np.testing.assert_array_equal(result.scores, [0, 5])
        self.assertEqual((result.tp, result.tn), (1, 1))

    def test_undefined_false_alarm_rate_when_no_healthy_test_cases(self):
        base = self.get_split()
        all_fault = partition([[4, 4, 4, 4]], group="sealed_all_fault", prefix="all_fault", labels=[1])
        report = AnomalyDiagnosticWorkflow(CountingDetector()).run(AnomalySplit(base.reference, all_fault))
        self.assertIsNone(report.false_alarm_rate)
        self.assertIsNone(report.to_dict()["false_alarm_rate"])


if __name__ == "__main__":
    unittest.main()
