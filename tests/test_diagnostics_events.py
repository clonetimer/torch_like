"""Event extraction must never fabricate temporal evidence from window scores."""

import unittest

import numpy as np

from torch_like.diagnostics import DiagnosticResult, SignalBatch
from torch_like.diagnostics.events import extract_anomaly_events


class DiagnosticEventTests(unittest.TestCase):
    def test_channel_specific_debounce_and_gap_episodes(self):
        input_batch = SignalBatch(
            np.zeros((1, 8, 2)),
            sample_rate_hz=4,
            channel_names=("current", "vibration"),
            sample_ids=("run-1",),
        )
        scores = np.array([[[0, 3], [2, 0], [0, 0], [4, 0], [2, 5], [0, 0], [0, 0], [3, 0]]])
        output = DiagnosticResult("anomaly_detection", "anomaly_score", scores, "ZScoreDetector", threshold=1.5)
        events = extract_anomaly_events(output, input_batch, min_active_points=2, gap_tolerance=1)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event.sample_index, event.sample_id, event.channel_name), (0, "run-1", "current"))
        self.assertEqual((event.start_index, event.end_index_exclusive), (1, 5))
        self.assertEqual(event.active_points, 3)
        self.assertEqual((event.first_time_s, event.last_time_s), (0.25, 1.0))
        self.assertEqual(event.peak_score, 4.0)
        self.assertEqual(event.resolution, "time")

    def test_time_only_scores_require_explicit_two_axis_layout(self):
        batch = SignalBatch(np.zeros((1, 5, 2)), sample_rate_hz=10)
        result = DiagnosticResult(
            "anomaly_detection", "anomaly_score", np.array([[0, 2, 2, 0, 3]]), "PCAT2Detector", threshold=1.0
        )
        with self.assertRaisesRegex(ValueError, "explicit"):
            extract_anomaly_events(result, batch)
        events = extract_anomaly_events(result, batch, score_layout="time", min_active_points=3, gap_tolerance=1)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].start_index, events[0].end_index_exclusive), (1, 5))
        self.assertEqual(events[0].active_points, 3)
        self.assertIsNone(events[0].channel_name)

    def test_window_score_has_no_invented_onset_or_duration(self):
        batch = SignalBatch(np.zeros((2, 8, 1)), sample_rate_hz=100, sample_ids=("a", "b"))
        result = DiagnosticResult(
            "anomaly_detection", "anomaly_score", np.array([1.5, 0.5]), "SklearnWindowAnomalyDetector", threshold=1.0
        )
        events = extract_anomaly_events(result, batch)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.sample_id, "a")
        self.assertEqual(event.resolution, "window")
        self.assertIsNone(event.start_index)
        self.assertIsNone(event.end_index_exclusive)
        self.assertIsNone(event.first_time_s)
        self.assertIsNone(event.last_time_s)
        self.assertIsNone(event.active_points)
        with self.assertRaisesRegex(ValueError, "window-level"):
            extract_anomaly_events(result, batch, min_active_points=2)

    def test_window_channel_layout_preserves_source_channel(self):
        batch = SignalBatch(np.zeros((1, 2, 2)), sample_rate_hz=10, channel_names=("a", "b"))
        scores = DiagnosticResult(
            "anomaly_detection", "anomaly_score", np.array([[2.0, 1.0]]), "window_model", threshold=1.0
        )
        events = extract_anomaly_events(scores, batch, score_layout="window_channel")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].channel_name, "a")
        self.assertIsNone(events[0].first_time_s)

    def test_event_times_use_explicit_timestamps_without_assuming_alignment(self):
        batch = SignalBatch.from_sequence(
            np.zeros((4, 1)), sample_rate_hz=10, timestamps_s=np.array([100, 100.2, 100.5, 100.7])
        )
        score = DiagnosticResult("anomaly_detection", "anomaly_score", np.array([[0, 2, 2, 0]]), "model", threshold=1)
        events = extract_anomaly_events(score, batch, score_layout="time", min_active_points=2)
        self.assertEqual((events[0].first_time_s, events[0].last_time_s), (100.2, 100.5))

    def test_no_merge_across_independent_windows(self):
        batch = SignalBatch(np.zeros((2, 3, 1)), sample_rate_hz=10)
        scores = np.array([[[0], [0], [3]], [[3], [0], [0]]])
        result = DiagnosticResult("anomaly_detection", "anomaly_score", scores, "model", threshold=1)
        events = extract_anomaly_events(result, batch, gap_tolerance=2)
        self.assertEqual(len(events), 2)
        self.assertEqual([e.sample_index for e in events], [0, 1])

    def test_unthresholded_and_wrong_task_outputs_fail_closed(self):
        batch = SignalBatch(np.zeros((1, 3, 1)), sample_rate_hz=10)
        uncalibrated = DiagnosticResult("anomaly_detection", "anomaly_score", np.array([1.0]), "PCA")
        with self.assertRaisesRegex(ValueError, "threshold"):
            extract_anomaly_events(uncalibrated, batch)
        wrong_task = DiagnosticResult("fault_classification", "fault_label", np.array(["fault"]), "classifier")
        with self.assertRaisesRegex(ValueError, "only anomaly"):
            extract_anomaly_events(wrong_task, batch)

    def test_shape_and_debounce_arguments_are_checked(self):
        batch = SignalBatch(np.zeros((1, 3, 2)), sample_rate_hz=10)
        mismatched = DiagnosticResult("anomaly_detection", "anomaly_score", np.array([1, 2]), "algo", threshold=1)
        with self.assertRaisesRegex(ValueError, "sample count"):
            extract_anomaly_events(mismatched, batch)
        wrong_time = DiagnosticResult("anomaly_detection", "anomaly_score", np.array([[0, 1]]), "algo", threshold=0)
        with self.assertRaisesRegex(ValueError, "time axis"):
            extract_anomaly_events(wrong_time, batch, score_layout="time")
        bad_channels = DiagnosticResult("anomaly_detection", "anomaly_score", np.zeros((1, 3, 1)), "algo", threshold=0)
        with self.assertRaisesRegex(ValueError, "channel axis"):
            extract_anomaly_events(bad_channels, batch)
        valid = DiagnosticResult("anomaly_detection", "anomaly_score", np.array([1.5]), "algo", threshold=1)
        for option in (0, True, -1):
            with self.subTest(min_active_points=option), self.assertRaisesRegex(ValueError, "min_active_points"):
                extract_anomaly_events(valid, batch, min_active_points=option)
        for option in (-1, True):
            with self.subTest(gap=option), self.assertRaisesRegex(ValueError, "gap_tolerance"):
                extract_anomaly_events(valid, batch, gap_tolerance=option)


if __name__ == "__main__":
    unittest.main()
