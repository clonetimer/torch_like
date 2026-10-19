"""R3a -> R3b integration: provenance-preserving event reporting without retraining."""

import unittest

import numpy as np

from torch_like.diagnostics import (
    AnomalyDiagnosticWorkflow,
    AnomalyPartition,
    AnomalySplit,
    SignalBatch,
    ZScoreDetector,
    extract_anomaly_events,
)


def recording(rows, *, origin, prefix, labels=None):
    arr = np.asarray(rows, dtype=float)[:, :, None]
    signals = SignalBatch(
        arr,
        sample_rate_hz=100.0,
        channel_names=("vibration",),
        units=("g",),
        sample_ids=tuple(f"{prefix}:{index}" for index in range(arr.shape[0])),
    )
    return AnomalyPartition(
        signals,
        recording_ids=(origin,) * arr.shape[0],
        labels=None if labels is None else np.asarray(labels),
    )


class DiagnosticEventIntegrationTests(unittest.TestCase):
    def make_split(self):
        healthy = recording(
            [[-1, 1, -1, 1], [1, -1, 1, -1], [-1, -1, 1, 1]],
            origin="train-device-A",
            prefix="train",
        )
        sealed = recording(
            [[0, 0, 0, 0], [4, 4, 4, 4]],
            origin="test-device-B",
            prefix="test",
            labels=[0, 1],
        )
        return AnomalySplit(healthy, sealed)

    def test_workflow_result_uses_fixed_threshold_to_report_window_only_alarm(self):
        split = self.make_split()
        result = AnomalyDiagnosticWorkflow(ZScoreDetector(sigma_threshold=2)).run(split)
        event_input = result.as_diagnostic_result()
        self.assertEqual(event_input.algorithm_name, "ZScoreDetector")
        self.assertEqual(event_input.threshold, 2.0)
        self.assertEqual(event_input.values.shape, (2,))
        events = extract_anomaly_events(event_input, split.test.signals)
        self.assertEqual(len(events), 1)
        first = events[0]
        self.assertEqual(first.sample_id, "test:1")
        self.assertEqual(first.sample_index, 1)
        self.assertEqual(first.resolution, "window")
        self.assertEqual(first.peak_score, 4.0)
        self.assertIsNone(first.first_time_s)
        self.assertIsNone(first.last_time_s)
        self.assertIsNone(first.start_index)
        self.assertIsNone(first.end_index_exclusive)

    def test_point_level_event_requires_original_pointwise_detector_output(self):
        split = self.make_split()
        detector = ZScoreDetector(sigma_threshold=2).fit(split.reference.signals)
        time_resolved = detector.score(split.test.signals)
        events = extract_anomaly_events(time_resolved, split.test.signals, min_active_points=3)
        self.assertEqual(len(events), 1)
        point_event = events[0]
        self.assertEqual(point_event.sample_id, "test:1")
        self.assertEqual(point_event.channel_name, "vibration")
        self.assertEqual((point_event.start_index, point_event.end_index_exclusive), (0, 4))
        self.assertEqual((point_event.first_time_s, point_event.last_time_s), (0.0, 0.03))
        self.assertEqual(point_event.active_points, 4)

    def test_event_reporting_does_not_run_model_again(self):
        class CountingZScore(ZScoreDetector):
            def __init__(self):
                super().__init__(sigma_threshold=2)
                self.calls = 0

            def score(self, batch):
                self.calls += 1
                return super().score(batch)

        split = self.make_split()
        detector = CountingZScore()
        report = AnomalyDiagnosticWorkflow(detector).run(split)
        self.assertEqual(detector.calls, 1)
        _ = extract_anomaly_events(report.as_diagnostic_result(), split.test.signals)
        _ = report.to_dict()
        self.assertEqual(detector.calls, 1)


if __name__ == "__main__":
    unittest.main()
