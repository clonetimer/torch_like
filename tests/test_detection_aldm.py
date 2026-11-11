"""ALDM detector API, optimization, and numerical-contract tests."""

import unittest

import torch

import torch_like as tl
from torch_like.models.detection.aldm import AdaptiveSPDMetric


class ALDMDetectorTests(unittest.TestCase):
    def test_registry_and_public_api_construction(self):
        model = tl.create_model(
            "ALDMDetector",
            domain="detection",
            input_dim=4,
            seq_len=9,
            d_model=16,
            metric_dim=12,
            metric_rank=4,
            flow_hidden_dim=16,
            flow_layers=2,
        )
        self.assertIsInstance(model, tl.ALDMDetector)
        self.assertEqual(model.config()["input_dim"], 4)
        self.assertEqual(model.config()["seq_len"], 9)

    def test_forward_and_window_level_score_contract(self):
        model = tl.ALDMDetector(
            input_dim=4,
            seq_len=9,
            d_model=16,
            metric_dim=12,
            metric_rank=4,
            flow_hidden_dim=16,
        )
        x = torch.randn(3, 9, 4)
        batch = model.make_detection_batch(x)
        prediction = model.predict(batch)
        scores = model.score_batch(batch, reduction="none")
        mean_score = model.score_batch(batch, reduction="mean")
        sum_score = model.score_batch(batch, reduction="sum")

        self.assertEqual(prediction.shape, (3,))
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(mean_score.ndim, 0)
        self.assertEqual(sum_score.ndim, 0)
        self.assertTrue(torch.isfinite(scores).all())

    def test_training_objective_backpropagates_through_flow_and_adaptive_metric(self):
        model = tl.ALDMDetector(
            input_dim=3,
            seq_len=9,
            d_model=12,
            metric_dim=8,
            metric_rank=3,
            flow_hidden_dim=16,
            flow_layers=2,
        )
        model.train()
        batch = model.make_detection_batch(torch.randn(4, 9, 3))
        loss = model.fit_step(batch)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        metric_gradient = model.adaptive_metric.factor.grad
        self.assertIsNotNone(metric_gradient)
        self.assertTrue(torch.isfinite(metric_gradient).all())
        self.assertGreater(float(metric_gradient.abs().sum()), 0.0)
        flow_gradients = [parameter.grad for parameter in model.flow.parameters()]
        self.assertTrue(any(gradient is not None for gradient in flow_gradients))
        contrastive_gradients = [parameter.grad for parameter in model.contrastive_encoder.parameters()]
        self.assertTrue(any(gradient is not None for gradient in contrastive_gradients))

    def test_evaluation_is_deterministic_and_sensor_graph_is_normalized(self):
        model = tl.ALDMDetector(
            input_dim=4,
            seq_len=7,
            d_model=12,
            metric_dim=8,
            metric_rank=3,
            flow_hidden_dim=16,
        )
        model.eval()
        x = torch.randn(2, 7, 4)
        with torch.no_grad():
            first = model.score_batch(model.make_detection_batch(x), reduction="none")
            second = model.score_batch(model.make_detection_batch(x), reduction="none")
            adjacency = model.sensor_adjacency(x)
        torch.testing.assert_close(first, second)
        self.assertEqual(adjacency.shape, (2, 4, 4))
        torch.testing.assert_close(
            adjacency.sum(dim=-1),
            torch.ones((2, 4)),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_adaptive_metric_is_symmetric_nonnegative_and_differentiable(self):
        metric = AdaptiveSPDMetric(feature_dim=6, rank=3)
        values = torch.randn(2, 5, 6, requires_grad=True)
        distances = metric.pairwise_distance(values)
        self.assertEqual(distances.shape, (2, 5, 5))
        self.assertTrue(torch.isfinite(distances).all())
        self.assertTrue(torch.all(distances >= 0))
        torch.testing.assert_close(distances, distances.transpose(1, 2), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.all(torch.diagonal(distances, dim1=1, dim2=2) <= 1e-5))
        distances.sum().backward()
        self.assertIsNotNone(metric.factor.grad)
        self.assertGreater(float(metric.factor.grad.abs().sum()), 0.0)

    def test_single_sample_odd_length_and_no_contrastive_training(self):
        model = tl.ALDMDetector(
            input_dim=2,
            seq_len=7,
            d_model=8,
            metric_dim=6,
            metric_rank=2,
            flow_hidden_dim=8,
        )
        batch = model.make_detection_batch(torch.randn(1, 7, 2))
        loss = model.fit_step(batch)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

        nll_only = tl.ALDMDetector(
            input_dim=2,
            seq_len=7,
            d_model=8,
            metric_dim=6,
            metric_rank=2,
            flow_hidden_dim=8,
            contrastive_weight=0.0,
        )
        self.assertTrue(torch.isfinite(nll_only.fit_step(batch)))

    def test_invalid_shapes_and_configuration_are_rejected(self):
        model = tl.ALDMDetector(
            input_dim=3,
            seq_len=8,
            d_model=8,
            metric_dim=6,
            metric_rank=2,
            flow_hidden_dim=8,
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, 7, 3))
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, 8, 4))
        with self.assertRaisesRegex(ValueError, "finite floating-point"):
            model(torch.full((2, 8, 3), float("nan")))

        invalid_configs = (
            {"input_dim": 0, "seq_len": 8},
            {"input_dim": 3, "seq_len": 1},
            {"input_dim": 3, "seq_len": 8, "metric_rank": 33, "metric_dim": 32},
            {"input_dim": 3, "seq_len": 8, "temperature": 0.0},
            {"input_dim": 3, "seq_len": 8, "instance_weight": 1.1},
            {"input_dim": 3, "seq_len": 8, "augmentation_mask_probability": 1.0},
        )
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                tl.ALDMDetector(**config)


if __name__ == "__main__":
    unittest.main()
