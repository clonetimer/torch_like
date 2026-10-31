import json
import unittest

import torch

from torch_like.models.forecasting.stgcn import (
    STGCN,
    STGCNConfig,
    _ChebyshevGraphConv,
    _scaled_normalized_laplacian,
    _TemporalConv,
)


class STGCNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.adjacency = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
        self.config = {
            "enc_in": 3,
            "dec_in": 3,
            "c_out": 3,
            "seq_len": 9,
            "label_len": 0,
            "pred_len": 2,
            "d_model": 8,
            "e_layers": 2,
            "bottleneck_channels": 4,
            "temporal_kernel_size": 3,
            "cheb_order": 3,
            "dropout": 0.0,
            "adjacency": self.adjacency,
        }

    def test_forward_predict_gradient_and_optimizer(self):
        model = STGCN(**self.config)
        x = torch.randn(2, 9, 3, requires_grad=True)
        prediction = model.predict(model.make_forecast_batch(x))
        self.assertEqual(prediction.shape, (2, 2, 3))
        self.assertTrue(torch.isfinite(prediction).all())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = model.blocks[0].graph.weight.detach().clone()
        prediction.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.blocks[0].graph.weight))

    def test_chebyshev_recurrence_matches_explicit_polynomials(self):
        graph = self.adjacency.double()
        laplacian = _scaled_normalized_laplacian(graph)
        x = torch.randn(2, 2, 3, 5, dtype=torch.float64, requires_grad=True)
        for order in (1, 2, 4):
            with self.subTest(order=order):
                layer = _ChebyshevGraphConv(2, 3, order).double()
                polynomials = [torch.eye(3, dtype=torch.float64)]
                if order > 1:
                    polynomials.append(laplacian)
                for _ in range(2, order):
                    polynomials.append(2.0 * laplacian @ polynomials[-1] - polynomials[-2])
                expected = torch.zeros(2, 3, 3, 5, dtype=torch.float64)
                for polynomial, weight in zip(polynomials, layer.weight, strict=True):
                    filtered = torch.einsum("nm,bcmt->bcnt", polynomial, x)
                    expected = expected + torch.einsum("bcnt,co->bont", filtered, weight)
                expected = expected + layer.bias[None, :, None, None]
                actual = layer(x, laplacian)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
                torch.testing.assert_close(actual_grad, expected_grad, atol=1e-11, rtol=1e-11)

    def test_scaled_laplacian_oracle_self_loops_and_isolates(self):
        graph = torch.tensor([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        inv = torch.diag(torch.tensor([3.0**-0.5, 5.0**-0.5, 0.0], dtype=torch.float64))
        normalized = inv @ (torch.diag(graph.sum(1)) - graph) @ inv
        expected = 2.0 * normalized / torch.linalg.eigvalsh(normalized).max() - torch.eye(3, dtype=torch.float64)
        actual = _scaled_normalized_laplacian(graph)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual[2], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))
        for graph in (torch.zeros(3, 3), torch.eye(3), torch.diag(torch.tensor([0.0, 2.0, 3.0]))):
            torch.testing.assert_close(_scaled_normalized_laplacian(graph.double()), -torch.eye(3).double())

    def test_temporal_residual_is_right_aligned_and_gated(self):
        x = torch.arange(1.0, 13.0).reshape(1, 2, 1, 6)
        for gated in (False, True):
            layer = _TemporalConv(2, 3, 3, gated=gated)
            with torch.no_grad():
                layer.conv.weight.zero_()
                layer.conv.bias.zero_()
            expected = torch.cat((x, torch.zeros(1, 1, 1, 6)), dim=1)[..., 2:]
            if gated:
                expected = expected * 0.5
            torch.testing.assert_close(layer(x), expected)
        reducing = _TemporalConv(3, 2, 1, gated=False)
        self.assertEqual(reducing(torch.ones(1, 3, 2, 4)).shape, (1, 2, 2, 4))

    def test_graph_is_used_and_order_one_is_graph_independent(self):
        x = torch.randn(2, 9, 3)
        for order in (1, 3):
            first = STGCN(**{**self.config, "cheb_order": order}).eval()
            second = STGCN(**{**self.config, "adjacency": torch.eye(3), "cheb_order": order}).eval()
            self._copy_parameters(first, second)
            if order == 1:
                torch.testing.assert_close(first(x), second(x), atol=0, rtol=0)
            else:
                self.assertFalse(torch.allclose(first(x), second(x), atol=1e-7, rtol=1e-7))

    def test_node_permutation_equivariance(self):
        permutation = torch.tensor([2, 0, 1])
        first = STGCN(**self.config).double().eval()
        second = STGCN(**{**self.config, "adjacency": self.adjacency[permutation][:, permutation]}).double().eval()
        self._copy_parameters(first, second)
        x = torch.randn(2, 9, 3, dtype=torch.float64)
        torch.testing.assert_close(second(x[..., permutation]), first(x)[..., permutation], atol=1e-10, rtol=1e-10)

    def test_graph_buffers_do_not_alias_input_or_config(self):
        graph = self.adjacency.clone().requires_grad_()
        model = STGCN(**{**self.config, "adjacency": graph})
        initial = model.adjacency.clone()
        with torch.no_grad():
            graph.zero_()
        torch.testing.assert_close(model.adjacency, initial)
        self.assertFalse(model.adjacency.requires_grad)
        self.assertFalse(model.scaled_laplacian.requires_grad)
        self.assertIn("adjacency", dict(model.named_buffers()))
        self.assertIn("scaled_laplacian", model.state_dict())
        snapshot = model.config()
        snapshot["adjacency"][0][1] = 999
        self.assertEqual(model.adjacency[0, 1].item(), 1)
        self.assertEqual(model.config()["adjacency"][0][1], 1)
        json.dumps(model.config())

    def test_list_config_and_state_dict_roundtrip(self):
        graph = self.adjacency.tolist()
        model = STGCN(**{**self.config, "adjacency": graph}).eval()
        graph[0][1] = 77
        restored = STGCN.from_config(json.loads(json.dumps(model.config()))).eval()
        restored.load_state_dict(model.state_dict())
        x = torch.randn(2, 9, 3)
        torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)

    def test_dataclass_config_and_nonminimal_temporal_window(self):
        config = STGCNConfig(
            enc_in=3,
            dec_in=2,
            c_out=3,
            seq_len=12,
            label_len=4,
            pred_len=5,
            d_model=8,
            e_layers=1,
            bottleneck_channels=4,
            dropout=0.0,
            adjacency=self.adjacency.tolist(),
        )
        model = STGCN.from_config(config).eval()
        batch = model.make_forecast_batch(torch.randn(2, 12, 3))
        self.assertEqual(batch.x_dec.shape, (2, 9, 2))
        self.assertEqual(model.predict(batch).shape, (2, 5, 3))
        self.assertEqual(model.output_temporal.kernel_size, 8)
        self.assertEqual(model.config()["adjacency"], config.adjacency)

    def test_positive_edges_cannot_underflow_during_graph_storage(self):
        graph = torch.tensor([[0.0, 1e-80], [1e-80, 0.0]], dtype=torch.float64)
        for adjacency in (graph, graph.tolist()):
            with self.subTest(type=type(adjacency).__name__), self.assertRaisesRegex(ValueError, "underflow to zero"):
                STGCN(**{**self.config, "enc_in": 2, "dec_in": 2, "c_out": 2, "adjacency": adjacency})

    def test_rounded_stored_graph_and_config_share_the_same_operator(self):
        graph = torch.tensor(
            [
                [0.0, 0.1234567890123, 0.8765432109876],
                [0.1234567890123, 0.0, 1.2345678901234],
                [0.8765432109876, 1.2345678901234, 0.0],
            ],
            dtype=torch.float64,
        )
        model = STGCN(**{**self.config, "adjacency": graph}).eval()
        restored = STGCN.from_config(json.loads(json.dumps(model.config()))).eval()
        self.assertFalse(torch.equal(graph, model.adjacency.double()))
        torch.testing.assert_close(model.adjacency, restored.adjacency, atol=0, rtol=0)
        torch.testing.assert_close(model.scaled_laplacian, restored.scaled_laplacian, atol=0, rtol=0)
        stored_oracle = _scaled_normalized_laplacian(model.adjacency.double()).to(model.scaled_laplacian.dtype)
        torch.testing.assert_close(model.scaled_laplacian, stored_oracle, atol=0, rtol=0)
        original_operator = _scaled_normalized_laplacian(graph)
        torch.testing.assert_close(model.scaled_laplacian.double(), original_operator, atol=1e-7, rtol=1e-7)
        self._copy_parameters(model, restored)
        x = torch.randn(2, 9, 3)
        torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)

    def test_future_inputs_and_markers_are_not_used(self):
        model = STGCN(**self.config).eval()
        x = torch.randn(2, 9, 3)
        poisoned = torch.full((2, 4, 3), float("nan"))
        torch.testing.assert_close(model(x), model(x, poisoned, poisoned, poisoned), atol=0, rtol=0)

    def test_dropout_train_eval_and_float64(self):
        model = STGCN(**{**self.config, "dropout": 0.5}).double()
        x = torch.randn(2, 9, 3, dtype=torch.float64)
        self.assertFalse(torch.equal(model(x), model(x)))
        model.eval()
        actual = model(x)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertTrue(all(buffer.dtype == torch.float64 for buffer in model.buffers()))
        torch.testing.assert_close(actual, model(x), atol=0, rtol=0)

    def test_smallest_temporal_window_single_node_and_zero_graph(self):
        for graph in ([[0.0]], [[1.0]]):
            model = STGCN(
                enc_in=1,
                dec_in=1,
                c_out=1,
                seq_len=1,
                label_len=0,
                pred_len=1,
                d_model=4,
                e_layers=1,
                bottleneck_channels=2,
                temporal_kernel_size=1,
                cheb_order=1,
                adjacency=graph,
            )
            self.assertEqual(model(torch.ones(2, 1, 1)).shape, (2, 1, 1))
        model = STGCN(**{**self.config, "adjacency": torch.zeros(3, 3)})
        self.assertTrue(torch.isfinite(model(torch.randn(2, 9, 3))).all())

    def test_invalid_adjacency(self):
        cases = [
            (None, "explicit adjacency"),
            (torch.eye(2), "shape"),
            ([[0, 1, 0], [0, 0, 0], [0, 0, 0]], "symmetric"),
            (-torch.eye(3), "nonnegative"),
            (torch.full((3, 3), float("nan")), "finite"),
            (torch.full((3, 3), float("inf")), "finite"),
            (torch.eye(3, dtype=torch.complex64), "real numeric"),
            (torch.eye(3).to_sparse(), "dense real"),
            ("not a graph", "real numeric"),
            (torch.full((3, 3), 1e308, dtype=torch.float64), "row sums"),
            (torch.eye(3, dtype=torch.float64) * 1e100, "representable"),
        ]
        for graph, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                STGCN(**{**self.config, "adjacency": graph})

    def test_invalid_dimensions_and_receptive_field(self):
        for name in (
            "enc_in",
            "dec_in",
            "c_out",
            "seq_len",
            "pred_len",
            "d_model",
            "e_layers",
            "bottleneck_channels",
            "temporal_kernel_size",
            "cheb_order",
        ):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    STGCN(**{**self.config, name: value})
        for value in (-1, 10, 0.5, True):
            with self.subTest(label_len=value), self.assertRaisesRegex(ValueError, "label_len"):
                STGCN(**{**self.config, "label_len": value})
        for value in (-0.1, 1, float("nan"), float("inf"), "0.1", True):
            with self.subTest(dropout=value), self.assertRaisesRegex(ValueError, "dropout"):
                STGCN(**{**self.config, "dropout": value})
        with self.assertRaisesRegex(ValueError, "c_out must equal enc_in"):
            STGCN(**{**self.config, "c_out": 2})
        with self.assertRaisesRegex(ValueError, "seq_len must exceed"):
            STGCN(**{**self.config, "seq_len": 8})
        with self.assertRaisesRegex(ValueError, "single-channel LayerNorm"):
            STGCN(**{**self.config, "d_model": 1})

    def test_invalid_encoder_inputs(self):
        model = STGCN(**self.config)
        for x in (torch.ones(2, 9), torch.ones(2, 8, 3), torch.ones(2, 9, 2), torch.ones(2, 9, 3, 1)):
            with self.subTest(shape=x.shape), self.assertRaisesRegex(ValueError, "shape"):
                model(x)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                model(torch.full((2, 9, 3), value))
        with self.assertRaisesRegex(ValueError, "batch item"):
            model(torch.empty(0, 9, 3))
        for x in (torch.ones(2, 9, 3, dtype=torch.long), torch.ones(2, 9, 3, dtype=torch.complex64), []):
            with self.assertRaisesRegex(TypeError, "floating-point"):
                model(x)

    @staticmethod
    def _copy_parameters(source, destination):
        with torch.no_grad():
            for source_parameter, destination_parameter in zip(
                source.parameters(), destination.parameters(), strict=True
            ):
                destination_parameter.copy_(source_parameter)


if __name__ == "__main__":
    unittest.main()
