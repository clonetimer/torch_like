"""ALDM detector, adaptive metric, conditional flow and event-metric tests."""

import unittest

import numpy as np
import torch

from torch_like.diagnostics import (
    AnomalyDiagnosticWorkflow,
    AnomalyPartition,
    AnomalySplit,
    EventWiseMetrics,
    SignalBatch,
    compute_event_wise_metrics,
)
from torch_like.models import ALDMDetector, ALDMDetectorConfig, create_model, list_models


def make_windows(batch_size=3, *, seq_len=12, channels=4):
    generator = torch.Generator().manual_seed(41)
    return torch.randn(batch_size, seq_len, channels, generator=generator)


def small_model(**kwargs):
    settings = {
        "input_dim": 4,
        "seq_len": 12,
        "d_model": 8,
        "dropout": 0.0,
        "metric_dim": 4,
        "latent_dim": 8,
        "flow_layers": 2,
        "contrastive_weight": 0.05,
        "augmentation_noise": 0.01,
    }
    settings.update(kwargs)
    return ALDMDetector(**settings)


def make_signal_batch(prefix, *, batch_size=4, seq_len=12, channels=4):
    values = make_windows(batch_size=batch_size, seq_len=seq_len, channels=channels).numpy()
    return SignalBatch(
        values,
        sample_rate_hz=10.0,
        channel_names=tuple(f"sensor_{index}" for index in range(channels)),
        units=tuple("normalized" for _ in range(channels)),
        sample_ids=tuple(f"{prefix}-{index}" for index in range(batch_size)),
    )


class ALDMDetectorTests(unittest.TestCase):
    def test_public_registry_and_config_construction(self):
        self.assertIn("ALDMDetector", list_models("detection"))
        model = create_model(
            "ALDMDetector",
            domain="detection",
            input_dim=4,
            seq_len=12,
            d_model=8,
            metric_dim=4,
            latent_dim=8,
            flow_layers=1,
        )
        self.assertIsInstance(model, ALDMDetector)
        config = ALDMDetectorConfig(
            input_dim=4,
            seq_len=12,
            d_model=8,
            metric_dim=4,
            latent_dim=8,
            flow_layers=1,
        )
        rebuilt = ALDMDetector.from_config(config)
        self.assertIsInstance(rebuilt, ALDMDetector)

    def test_train_step_backpropagates_into_spd_metric_graph_and_flow(self):
        model = small_model()
        model.train()
        batch = model.make_detection_batch(make_windows())
        loss = model.fit_step(batch)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for parameter in (
            model.metric_factor,
            model.sensor_query.weight,
            model.flow_conditioners[0][0].weight,
            model.latent_encoder[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_adaptive_metric_is_positive_definite_symmetric_and_differentiable(self):
        model = small_model()
        context = torch.randn(2, 7, 4, requires_grad=True)
        distances = model.adaptive_distances(context)
        self.assertEqual(distances.shape, (2, 7, 7))
        torch.testing.assert_close(distances, distances.transpose(1, 2), atol=1e-6, rtol=1e-6)

        metric = model._metric_matrix()
        torch.testing.assert_close(metric, metric.transpose(0, 1), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(metric) > 0))
        distances.sum().backward()
        self.assertIsNotNone(model.metric_factor.grad)
        self.assertTrue(torch.isfinite(model.metric_factor.grad).all())

    def test_score_shapes_and_eval_determinism(self):
        model = small_model()
        model.eval()
        batch = model.make_detection_batch(make_windows(batch_size=2))
        first = model.score_batch(batch, reduction="none")
        second = model.score_batch(batch, reduction="none")
        self.assertEqual(first.shape, (2, 12, 4))
        torch.testing.assert_close(first, second)
        self.assertEqual(model.score_batch(batch, reduction="sum").shape, (2, 12))
        self.assertEqual(model.score_batch(batch, reduction="mean").shape, (2,))
        self.assertEqual(model.predict(batch).shape, (2,))

    def test_threshold_calibration_uses_healthy_windows_and_restores_mode(self):
        model = small_model()
        model.train()
        batch = model.make_detection_batch(make_windows(batch_size=5))
        threshold = model.calibrate_threshold(batch, quantile=0.8)
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual(model._config.threshold, threshold)
        self.assertTrue(model.training)

    def test_single_sample_batch_skips_instance_loss_without_nan(self):
        model = small_model()
        batch = model.make_detection_batch(make_windows(batch_size=1))
        loss = model.fit_step(batch)
        self.assertTrue(torch.isfinite(loss))

    def test_anomaly_diagnostic_workflow_uses_validation_not_test_labels(self):
        reference = AnomalyPartition(
            make_signal_batch("reference"),
            recording_ids=("ref-run-0", "ref-run-0", "ref-run-1", "ref-run-1"),
            labels=np.zeros(4, dtype=np.uint8),
        )
        validation = AnomalyPartition(
            make_signal_batch("validation"),
            recording_ids=("val-run-0", "val-run-0", "val-run-1", "val-run-1"),
            labels=np.array([0, 0, 1, 1], dtype=np.uint8),
        )
        test = AnomalyPartition(
            make_signal_batch("test"),
            recording_ids=("test-run-0", "test-run-0", "test-run-1", "test-run-1"),
            labels=np.array([0, 1, 0, 1], dtype=np.uint8),
        )
        workflow = AnomalyDiagnosticWorkflow(
            small_model(),
            threshold_policy="validation_f1",
            threshold_candidates=8,
        )
        result = workflow.run(AnomalySplit(reference=reference, validation=validation, test=test))
        self.assertEqual(result.algorithm_name, "ALDMDetector")
        self.assertEqual(result.threshold_source, "validation_f1")
        self.assertEqual(result.scores.shape, (4,))
        self.assertTrue(np.isfinite(result.threshold))
        self.assertTrue(np.isfinite(result.f1))

    def test_invalid_configuration_and_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "seq_len"):
            small_model(seq_len=1)
        with self.assertRaisesRegex(ValueError, "metric_dim"):
            small_model(metric_dim=0)
        with self.assertRaisesRegex(ValueError, "flow_layers"):
            small_model(flow_layers=0)
        with self.assertRaisesRegex(ValueError, "contrastive_temperature"):
            small_model(contrastive_temperature=0)
        with self.assertRaisesRegex(ValueError, "augmentation_mask_probability"):
            small_model(augmentation_mask_probability=1.0)

        model = small_model()
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, 10, 4))
        with self.assertRaisesRegex(ValueError, "finite"):
            model(torch.full((2, 12, 4), float("nan")))


class ALDMEventWiseMetricTests(unittest.TestCase):
    def test_perfect_event_detection_has_unit_metrics(self):
        truth = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        result = compute_event_wise_metrics(truth, truth)
        self.assertIsInstance(result, EventWiseMetrics)
        self.assertEqual((result.precision, result.recall, result.f1), (1.0, 1.0, 1.0))
        self.assertEqual(result.true_positive_events, 2)
        self.assertEqual(result.false_positive_events, 0)
        self.assertEqual(result.false_negative_events, 0)

    def test_event_recall_counts_partial_overlap_and_false_alarm_segments(self):
        truth = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        prediction = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        result = compute_event_wise_metrics(truth, prediction)
        self.assertEqual(result.true_positive_events, 1)
        self.assertEqual(result.false_positive_events, 2)
        self.assertEqual(result.false_negative_events, 1)
        self.assertEqual(result.recall, 0.5)
        self.assertLess(result.f1, 0.5)

    def test_explicit_intervals_preserve_adjacent_but_distinct_events(self):
        truth = np.array([0, 1, 1, 1, 0, 0])
        prediction = np.array([0, 1, 0, 1, 0, 0])
        result = compute_event_wise_metrics(truth, prediction, gt_intervals=((1, 2), (2, 4)))
        self.assertEqual(result.true_positive_events, 2)
        self.assertEqual(result.false_negative_events, 0)

    def test_empty_predictions_and_normal_only_series_are_safe(self):
        no_predictions = compute_event_wise_metrics([0, 1, 1, 0], [0, 0, 0, 0])
        self.assertEqual(no_predictions.precision, 0.0)
        self.assertEqual(no_predictions.recall, 0.0)
        self.assertEqual(no_predictions.f1, 0.0)

        normal_only = compute_event_wise_metrics([0, 0, 0], [0, 1, 0])
        self.assertEqual(normal_only.true_positive_events, 0)
        self.assertEqual(normal_only.false_positive_events, 1)
        self.assertEqual(normal_only.recall, 0.0)

    def test_invalid_labels_lengths_and_intervals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical lengths"):
            compute_event_wise_metrics([0, 1], [0])
        with self.assertRaisesRegex(ValueError, "binary"):
            compute_event_wise_metrics([0, 2], [0, 1])
        with self.assertRaisesRegex(ValueError, "sorted, disjoint"):
            compute_event_wise_metrics([0, 1, 1, 0], [0, 1, 0, 0], gt_intervals=((1, 3), (2, 4)))


if __name__ == "__main__":
    unittest.main()
