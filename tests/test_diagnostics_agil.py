"""AGIL feature-graph, EdgePool and joint-task contract tests."""

import unittest

import numpy as np
import torch

from torch_like.diagnostics import (
    AGILFeatureGraphBuilder,
    AGILGraphBatch,
    AGILMultiTask,
    SignalBatch,
    extract_agil_features,
)


def make_signals(samples=8, *, length=128, rate=25600.0):
    time = np.arange(length, dtype=np.float64) / rate
    values = []
    for index in range(samples):
        frequency = 80.0 + 7.0 * index
        signal = (
            (1.0 + 0.05 * index) * np.sin(2.0 * np.pi * frequency * time)
            + 0.2 * np.sin(2.0 * np.pi * 3.0 * frequency * time)
            + 0.01 * index
        )
        values.append(signal[:, None])
    return SignalBatch(
        np.asarray(values),
        sample_rate_hz=rate,
        channel_names=("vibration",),
        units=("g",),
    )


class AGILFeatureGraphTests(unittest.TestCase):
    def test_extracts_sixteen_finite_features_in_paper_order(self):
        signals = make_signals(samples=5)
        features = extract_agil_features(signals)
        self.assertEqual(features.shape, (5, 16))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:, 0], signals.values[:, :, 0].mean(axis=1))
        np.testing.assert_allclose(
            features[:, 1],
            np.sqrt(np.mean(signals.values[:, :, 0] ** 2, axis=1)),
        )

    def test_zero_signal_features_are_finite(self):
        zeros = SignalBatch.from_sequence(
            np.zeros((4, 32, 1)),
            sample_rate_hz=100.0,
            channel_names=("vibration",),
            units=("g",),
        )
        features = extract_agil_features(zeros)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[:, 1:], 0.0)

    def test_feature_graph_is_fitted_on_training_partition_only(self):
        training = make_signals(samples=8)
        builder = AGILFeatureGraphBuilder(correlation_threshold=0.8).fit(training)
        adjacency_before = builder.adjacency_
        held_out = make_signals(samples=2)
        transformed = builder.transform(held_out)

        self.assertIsInstance(transformed, AGILGraphBatch)
        self.assertEqual(transformed.node_features.shape, (2, 16, 1))
        self.assertEqual(transformed.adjacency.shape, (16, 16))
        np.testing.assert_array_equal(transformed.adjacency, adjacency_before)
        np.testing.assert_array_equal(transformed.adjacency, transformed.adjacency.T)
        np.testing.assert_array_equal(np.diag(transformed.adjacency), np.ones(16))
        self.assertTrue(np.isfinite(transformed.node_features).all())

        x, adjacency = transformed.to_torch()
        self.assertEqual(tuple(x.shape), (2, 16, 1))
        self.assertEqual(tuple(adjacency.shape), (16, 16))
        self.assertEqual(x.dtype, torch.float32)

    def test_builder_rejects_unfitted_insufficient_and_mismatched_inputs(self):
        builder = AGILFeatureGraphBuilder()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            builder.transform(make_signals(samples=2))
        with self.assertRaisesRegex(ValueError, "at least three"):
            builder.fit(make_signals(samples=2))

        builder.fit(make_signals(samples=5))
        wrong_metadata = SignalBatch.from_sequence(
            np.ones((2, 128, 1)),
            sample_rate_hz=12800.0,
            channel_names=("vibration",),
            units=("g",),
        )
        with self.assertRaisesRegex(ValueError, "sampling rate"):
            builder.transform(wrong_metadata)

    def test_feature_extraction_requires_single_channel(self):
        multi = SignalBatch(
            np.ones((3, 32, 2)),
            sample_rate_hz=100.0,
            channel_names=("horizontal", "vertical"),
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            extract_agil_features(multi)


class AGILMultiTaskTests(unittest.TestCase):
    def test_edge_pool_contracts_dense_graph(self):
        model = AGILMultiTask(num_classes=4, hidden_dim=16, dropout=0.0)
        x = torch.randn(2, 16, 16)
        adjacency = torch.ones(16, 16)
        mask = torch.ones(2, 16, dtype=torch.bool)
        pooled_x, pooled_adjacency, pooled_mask = model.edge_pools[0](x, adjacency.unsqueeze(0).expand(2, -1, -1), mask)
        self.assertEqual(tuple(pooled_x.shape), (2, 16, 16))
        self.assertEqual(tuple(pooled_adjacency.shape), (2, 16, 16))
        self.assertEqual(tuple(pooled_mask.shape), (2, 16))
        self.assertEqual(pooled_mask.sum(dim=1).tolist(), [8, 8])
        self.assertTrue(torch.all(pooled_adjacency.diagonal(dim1=1, dim2=2)[pooled_mask] > 0))

    def test_forward_outputs_fault_logits_and_nonnegative_rul(self):
        model = AGILMultiTask(num_classes=4, hidden_dim=16, dropout=0.0)
        model.eval()
        x = torch.randn(3, 16, 1)
        adjacency = torch.ones(16, 16)
        first = model(x, adjacency)
        second = model(x, adjacency)
        self.assertEqual(first.fault_logits.shape, (3, 4))
        self.assertEqual(first.rul_values.shape, (3,))
        self.assertTrue(torch.isfinite(first.fault_logits).all())
        self.assertTrue(torch.isfinite(first.rul_values).all())
        self.assertTrue(torch.all(first.rul_values >= 0))
        torch.testing.assert_close(first.fault_logits, second.fault_logits)
        torch.testing.assert_close(first.rul_values, second.rul_values)

    def test_joint_loss_backpropagates_through_shared_and_task_specific_paths(self):
        model = AGILMultiTask(num_classes=4, hidden_dim=16, dropout=0.0)
        model.train()
        x = torch.randn(4, 16, 1)
        adjacency = torch.ones(16, 16)
        fault_labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        rul_targets = torch.tensor([40.0, 30.0, 20.0, 10.0])
        loss = model.training_loss(x, adjacency, fault_labels, rul_targets)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for parameter in (
            model.gin_layers[0].mlp[0].weight,
            model.edge_pools[0].edge_scorer.weight,
            model.fault_attention.query.weight,
            model.rul_attention.query.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_rejects_invalid_graph_labels_and_rul_targets(self):
        model = AGILMultiTask(num_classes=3, hidden_dim=16)
        x = torch.randn(2, 16, 1)
        adjacency = torch.ones(16, 16)

        with self.assertRaisesRegex(ValueError, "node_features"):
            model(torch.randn(2, 15, 1), adjacency)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            model(x, torch.triu(torch.ones(16, 16)))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            model(x, -torch.ones(16, 16))

        with self.assertRaisesRegex(ValueError, "class index"):
            model.training_loss(
                x,
                adjacency,
                torch.tensor([0, 3]),
                torch.tensor([10.0, 8.0]),
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            model.training_loss(
                x,
                adjacency,
                torch.tensor([0, 1]),
                torch.tensor([10.0, -1.0]),
            )

    def test_invalid_constructor_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_classes"):
            AGILMultiTask(num_classes=1)
        with self.assertRaisesRegex(ValueError, "lambda_fault"):
            AGILMultiTask(num_classes=3, lambda_fault=0.0)
        with self.assertRaisesRegex(ValueError, "lambda_rul"):
            AGILMultiTask(num_classes=3, lambda_rul=-0.1)


if __name__ == "__main__":
    unittest.main()
