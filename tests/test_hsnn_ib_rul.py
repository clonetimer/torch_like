"""HSNN-IB RUL model and diagnostic-adapter contract tests."""

import unittest

import numpy as np
import torch

from torch_like.diagnostics import SignalBatch, create_rul_regressor, list_rul_models
from torch_like.models.hsnn_ib import HSNNIBRUL


def make_signals(batch=4, *, steps=12, channels=4):
    rng = np.random.default_rng(17)
    values = rng.normal(size=(batch, steps, channels)).astype(np.float32)
    # Add shared slow variation so the initial cosine graph has nontrivial edges.
    trend = np.linspace(0.0, 1.0, steps, dtype=np.float32)[None, :, None]
    values += trend
    return SignalBatch(
        values,
        sample_rate_hz=10.0,
        channel_names=tuple(f"sensor_{index}" for index in range(channels)),
        units=tuple("normalized" for _ in range(channels)),
    )


def small_model(**kwargs):
    settings = {
        "enc_in": 4,
        "seq_len": 12,
        "hidden_dim": 8,
        "num_layers": 1,
        "num_subgraphs": 2,
        "max_nodes_per_subgraph": 2,
        "ib_weight": 0.01,
    }
    settings.update(kwargs)
    return HSNNIBRUL(**settings)


class HSNNIBRULTests(unittest.TestCase):
    def test_partition_graph_returns_disjoint_bounded_subgraphs(self):
        model = small_model()
        groups = model.partition_graph(torch.ones(4, 4))
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(1 <= len(group) <= 2 for group in groups))
        flattened = [node for group in groups for node in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {0, 1, 2, 3})

    def test_partition_graph_handles_isolated_sensors(self):
        model = small_model()
        groups = model.partition_graph(torch.eye(4))
        self.assertEqual(groups, ((0,), (1,)))
        self.assertEqual(len(set(node for group in groups for node in group)), 2)

    def test_eval_forward_is_finite_deterministic_and_nonnegative(self):
        model = small_model()
        model.eval()
        x = torch.as_tensor(make_signals().values)
        first = model(x)
        second = model(x)
        self.assertEqual(first.shape, (4,))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.all(first >= 0))
        torch.testing.assert_close(first, second)

    def test_information_bottleneck_loss_backpropagates(self):
        model = small_model()
        model.train()
        x = torch.as_tensor(make_signals().values)
        targets = torch.tensor([20.0, 18.0, 12.0, 8.0])
        objective = model.training_loss(x, targets, loss="mse")
        self.assertEqual(objective.ndim, 0)
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        self.assertIsNotNone(model.ib_encoder.weight.grad)
        self.assertTrue(torch.isfinite(model.ib_encoder.weight.grad).all())
        self.assertIsNotNone(model.edge_similarity[0].weight.grad)
        self.assertTrue(torch.isfinite(model.edge_similarity[0].weight.grad).all())

    def test_rul_regressor_uses_model_specific_ib_training_loss(self):
        signals = make_signals(batch=6)
        model = small_model()
        regressor = create_rul_regressor(
            "HSNNIBRUL",
            unit="cycles",
            enc_in=4,
            seq_len=12,
            hidden_dim=8,
            num_layers=1,
            num_subgraphs=2,
            max_nodes_per_subgraph=2,
            ib_weight=0.01,
        )
        self.assertIsInstance(regressor.model, HSNNIBRUL)
        self.assertFalse(regressor.is_ready)
        regressor.fit(
            signals,
            np.array([30.0, 26.0, 22.0, 18.0, 14.0, 10.0]),
            epochs=2,
            batch_size=3,
            learning_rate=1e-3,
            seed=23,
        )
        self.assertTrue(regressor.is_ready)
        result = regressor.predict(signals)
        self.assertEqual(result.values.shape, (6,))
        self.assertTrue(np.isfinite(result.values).all())
        self.assertTrue(np.all(result.values >= 0))

    def test_rul_factory_lists_hsnn_ib(self):
        self.assertIn("HSNNIBRUL", list_rul_models())

    def test_invalid_dimensions_threshold_and_input_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_subgraphs"):
            small_model(num_subgraphs=5)
        with self.assertRaisesRegex(ValueError, "initial_edge_threshold"):
            small_model(initial_edge_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "ib_weight"):
            small_model(ib_weight=-1.0)

        model = small_model()
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.ones(2, 10, 4))
        with self.assertRaisesRegex(ValueError, "finite"):
            model(torch.full((2, 12, 4), float("nan")))

    def test_partition_rejects_invalid_edge_matrices(self):
        model = small_model()
        with self.assertRaisesRegex(ValueError, "shape"):
            model.partition_graph(torch.ones(3, 4))
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            model.partition_graph(torch.tensor([[1.0, float("nan"), 0.0, 0.0]] * 4))
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            model.partition_graph(-torch.ones(4, 4))


if __name__ == "__main__":
    unittest.main()
