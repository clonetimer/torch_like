"""Independent GMAN operator oracles: wrong axes, sources, gates and BN state fail."""

import importlib
import math

import pytest
import torch
from torch.nn import functional as F


def layers():
    name = "torch_like.models.forecasting.gman_layers"
    assert importlib.util.find_spec(name) is not None, "native GMAN operators are missing"
    return importlib.import_module(name)


def fc_oracle(layer, x):
    """Explicit affine/normalization math, never calls production forward."""
    x = F.linear(x, layer.linear.weight, layer.linear.bias)
    if layer.norm is not None:
        norm = layer.norm
        if norm.training:
            axes = tuple(range(x.ndim - 1))
            mean = x.mean(axes, keepdim=True)
            var = ((x - mean) ** 2).mean(axes, keepdim=True)
        else:
            mean, var = norm.running_mean, norm.running_var
        x = (x - mean) / torch.sqrt(var + 0.001) * norm.weight + norm.bias
    return torch.relu(x) if layer.activate else x


def mlp_oracle(module, x):
    for layer in module:
        x = fc_oracle(layer, x)
    return x


def attention_loop(query, key, value, heads, *, temporal, causal=False):
    """Separate per-batch/per-head loops catch batch/head and node/time mixing."""
    batch, length, nodes, width = query.shape
    depth = width // heads
    batches = []
    for b in range(batch):
        outputs = []
        for h in range(heads):
            sl = slice(h * depth, (h + 1) * depth)
            q, k, v = query[b, ..., sl], key[b, ..., sl], value[b, ..., sl]
            if temporal:
                values = []
                for n in range(nodes):
                    scores = q[:, n] @ k[:, n].T / math.sqrt(depth)
                    if causal:
                        allowed = torch.arange(k.size(0))[None, :] <= torch.arange(length)[:, None]
                        scores = scores.masked_fill(~allowed, -torch.inf)
                    values.append(scores.softmax(-1) @ v[:, n])
                outputs.append(torch.stack(values, 1))
            else:
                outputs.append(
                    torch.stack([(q[t] @ k[t].T / math.sqrt(depth)).softmax(-1) @ v[t] for t in range(length)])
                )
        batches.append(torch.cat(outputs, -1))
    return torch.stack(batches)


def attention_oracle(module, x, ste, *, temporal, causal=False):
    joined = torch.cat((x, ste), -1)
    projected = [fc_oracle(layer, joined) for layer in (module.query, module.key, module.value)]
    attended = attention_loop(*projected, module.n_heads, temporal=temporal, causal=causal)
    return mlp_oracle(module.output, attended)


def transform_oracle(module, x, history, future):
    projected = [fc_oracle(module.query, future), fc_oracle(module.key, history), fc_oracle(module.value, x)]
    attended = attention_loop(*projected, module.n_heads, temporal=True)
    return mlp_oracle(module.output, attended)


def fusion_oracle(module, spatial, temporal):
    gate = torch.sigmoid(
        F.linear(spatial, module.spatial_gate.weight)
        + F.linear(temporal, module.temporal_gate.weight, module.temporal_gate.bias)
    )
    return mlp_oracle(module.output, gate * spatial + (1 - gate) * temporal)


def block_oracle(module, x, ste):
    spatial = attention_oracle(module.spatial, x, ste, temporal=False)
    temporal = attention_oracle(module.temporal, x, ste, temporal=True, causal=module.temporal.causal)
    return x + fusion_oracle(module.fusion, spatial, temporal)


def embedding_oracle(module, spatial, calendar):
    # Literal scatter construction, not the production one_hot helper.
    encoded = spatial.new_zeros(*calendar.shape[:2], 7 + module.time_slots_per_day)
    encoded.scatter_(-1, calendar[..., :1].long(), 1)
    encoded.scatter_(-1, 7 + calendar[..., 1:].long(), 1)
    return mlp_oracle(module.spatial, spatial[None, None]) + mlp_oracle(module.temporal, encoded[:, :, None])


def assert_gradients(actual, expected, inputs, parameters):
    weight = torch.linspace(-0.7, 1.3, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
    targets = tuple(inputs) + tuple(parameters)
    actual_grad = torch.autograd.grad((actual * weight).sum(), targets, retain_graph=True)
    expected_grad = torch.autograd.grad((expected * weight).sum(), targets)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("kind,causal", [("spatial", False), ("temporal", False), ("temporal", True)])
@pytest.mark.parametrize("heads", [1, 2])
def test_attention_matches_independent_node_time_head_oracle_and_all_gradients(kind, causal, heads):
    torch.manual_seed(321)
    api = layers()
    cls = api._SpatialAttention if kind == "spatial" else api._TemporalAttention
    extras = {"causal": causal} if kind == "temporal" else {}
    module = cls(4, heads, batch_norm=False, bn_decay=0.9, **extras).double()
    x, ste = [torch.randn(2, 3, 3, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    actual = module(x, ste)
    expected = attention_oracle(module, x, ste, temporal=kind == "temporal", causal=causal)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert_gradients(actual, expected, (x, ste), module.parameters())


@pytest.mark.parametrize("heads", [1, 2])
def test_transform_uses_future_queries_history_keys_and_traffic_only_values(heads):
    torch.manual_seed(42)
    module = layers()._TransformAttention(4, heads, batch_norm=False, bn_decay=0.9).double()
    x, history = [torch.randn(2, 3, 2, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    future = torch.randn(2, 2, 2, 4, dtype=torch.float64, requires_grad=True)
    actual = module(x, history, future)
    expected = transform_oracle(module, x, history, future)
    assert actual.shape == (2, 2, 2, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert_gradients(actual, expected, (x, history, future), module.parameters())


def test_temporal_mask_blocks_future_keys_but_preserves_diagonal():
    torch.manual_seed(901)
    module = layers()._TemporalAttention(4, 2, batch_norm=False, bn_decay=0.9, causal=True).double()
    x, ste = [torch.rand(2, 3, 2, 4, dtype=torch.float64) for _ in range(2)]
    changed = x.clone()
    changed[:, 2] += 20
    torch.testing.assert_close(module(x, ste)[:, :2], module(changed, ste)[:, :2], rtol=0, atol=0)
    assert not torch.allclose(module(x, ste)[:, 2], module(changed, ste)[:, 2])


def test_gated_fusion_bias_orientation_and_residual_match_independent_math():
    torch.manual_seed(2)
    module = layers()._STAttentionBlock(4, 2, batch_norm=False, bn_decay=0.9, causal=False).double()
    with torch.no_grad():
        module.fusion.temporal_gate.bias.copy_(torch.tensor([-1.0, 0.4, 2.0, -0.3]))
    x, ste = [torch.randn(2, 3, 2, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    actual, expected = module(x, ste), block_oracle(module, x, ste)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert_gradients(actual, expected, (x, ste), module.parameters())
    with torch.no_grad():
        module.fusion.output[-1].linear.weight.zero_()
        module.fusion.output[-1].linear.bias.zero_()
    torch.testing.assert_close(module(x, ste), x, rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
def test_embedding_weekday_then_slot_and_broadcast_match_explicit_scatter(training):
    torch.manual_seed(61)
    module = layers()._STEmbedding(4, 8, batch_norm=True, bn_decay=0.9).double().train(training)
    spatial = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    calendar = torch.tensor([[[0, 7], [6, 0], [2, 4]], [[5, 6], [1, 3], [0, 0]]])
    actual = module(spatial, calendar)
    expected = embedding_oracle(module, spatial, calendar)
    assert actual.shape == (2, 3, 3, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert_gradients(actual, expected, (spatial,), module.parameters())


def test_population_bn_two_batches_and_frozen_eval_use_explicit_moments():
    norm = layers()._ChannelBatchNorm(2, decay=0.5).double()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, -0.5]))
        norm.bias.copy_(torch.tensor([0.3, -0.1]))
    first = torch.tensor([[[[1.0, 2.0], [3.0, 6.0]]]], dtype=torch.float64)
    expected = (first - torch.tensor([2.0, 4.0])) / torch.sqrt(torch.tensor([1.001, 4.001], dtype=torch.float64))
    expected = expected * norm.weight + norm.bias
    torch.testing.assert_close(norm(first), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(norm.running_mean, torch.tensor([1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(norm.running_var, torch.tensor([1.0, 2.5], dtype=torch.float64))
    second = torch.tensor([[[[5.0, 4.0], [9.0, 4.0]]]], dtype=torch.float64)
    norm(second)
    torch.testing.assert_close(norm.running_mean, torch.tensor([4.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(norm.running_var, torch.tensor([2.5, 1.25], dtype=torch.float64))
    assert norm.num_batches_tracked.item() == 2
    norm.eval()
    expected = (second - torch.tensor([4.0, 3.0])) / torch.sqrt(torch.tensor([2.501, 1.251], dtype=torch.float64))
    torch.testing.assert_close(norm(second), expected * norm.weight + norm.bias)
    assert norm.num_batches_tracked.item() == 2


def test_singleton_population_bn_is_defined_without_unbiased_variance():
    norm = layers()._ChannelBatchNorm(2, decay=0.5).double()
    x = torch.tensor([[[[2.0, -3.0]]]], dtype=torch.float64, requires_grad=True)
    actual = norm(x)
    torch.testing.assert_close(actual, torch.zeros_like(x))
    torch.testing.assert_close(norm.running_var, torch.full((2,), 0.5, dtype=torch.float64))
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("activate", [False, True])
def test_fc_normalizes_only_activated_affines(activate):
    module = layers()._FC(3, 2, activate=activate, batch_norm=True, bn_decay=0.5).double()
    x = torch.randn(2, 3, 2, 3, dtype=torch.float64)
    expected = fc_oracle(module, x)
    torch.testing.assert_close(module(x), expected)
    if not activate:
        torch.testing.assert_close(module(x), F.linear(x, module.linear.weight, module.linear.bias))


def test_output_mlp_dropout_is_before_both_affines_and_eval_is_deterministic():
    module = layers()._mlp(3, 4, 1, batch_norm=False, bn_decay=0.9, dropout=0.4).double()
    x = torch.randn(2, 3, 2, 3, dtype=torch.float64)
    torch.manual_seed(100)
    expected = x
    for layer in module:
        expected = F.dropout(expected, p=0.4, training=True)
        expected = fc_oracle(layer, expected)
    torch.manual_seed(100)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    module.eval()
    torch.testing.assert_close(module(x), mlp_oracle(module, x), rtol=0, atol=0)


def test_transform_gradcheck_uses_real_attention_path():
    torch.manual_seed(4)
    module = layers()._TransformAttention(2, 1, batch_norm=False, bn_decay=0.9).double()
    inputs = tuple(torch.randn(1, steps, 1, 2, dtype=torch.float64, requires_grad=True) for steps in (2, 2, 1))
    assert torch.autograd.gradcheck(module, inputs, eps=1e-6, atol=1e-4)
