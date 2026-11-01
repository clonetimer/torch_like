"""Independent dense math catches incorrect sparse scores, padding and gradients."""

import copy

import pytest
import torch
import torch.nn.functional as F
from test_pyraformer_graph import dense_graph, layers


def norm(x, module):
    return F.layer_norm(x, module.normalized_shape, module.weight, module.bias, module.eps)


def dense_attention(module, x, adjacency):
    b, s, d = x.shape
    q, k, v = [
        F.linear(x, p.weight).reshape(b, s, module.n_heads, -1).transpose(1, 2)
        for p in (module.q_proj, module.k_proj, module.v_proj)
    ]
    weights = (q @ k.transpose(-1, -2) / module.head_dim**0.5).masked_fill(~adjacency, -torch.inf).softmax(-1)
    context = (weights @ v).transpose(1, 2).reshape(b, s, d)
    return norm(x + F.linear(context, module.out_proj.weight, module.out_proj.bias), module.norm)


def dense_encoder(module, x, adjacency):
    state = dense_attention(module.attention, x, adjacency)
    hidden = F.gelu(F.linear(state, module.ff1.weight, module.ff1.bias))
    return norm(state + F.linear(hidden, module.ff2.weight, module.ff2.bias), module.norm)


@pytest.mark.parametrize("encoder", [False, True])
@pytest.mark.parametrize("fine,factors,inner", [(8, (2, 2, 2), 1), (23, (3, 2, 2), 3)])
def test_sparse_forward_and_all_parameter_gradients_match_dense(encoder, fine, factors, inner):
    torch.manual_seed(27)
    ops = layers()
    sizes, indices, valid, _ = ops._pyramid_graph(fine, factors, inner)
    sparse = (ops._PyramidEncoderLayer(6, 2, 9, 0.0) if encoder else ops._PyramidalAttention(6, 2, 0.0)).double()
    dense = copy.deepcopy(sparse)
    x = torch.randn(2, sum(sizes), 6, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = sparse(x, indices, valid)
    expected = (dense_encoder if encoder else dense_attention)(dense, y, dense_graph(sizes, factors, inner))
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(), (x, *sparse.parameters()))
    expected_grads = torch.autograd.grad((expected * probe).sum(), (y, *dense.parameters()))
    for a, e in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, e, rtol=2e-10, atol=2e-11)


def test_invalid_padding_keys_do_not_steal_probability_or_gradient():
    torch.manual_seed(28)
    ops = layers()
    sizes, indices, valid, _ = ops._pyramid_graph(11, (3, 2), 3)
    module = ops._PyramidalAttention(4, 2, 0).double()
    x = torch.randn(2, sum(sizes), 4, dtype=torch.float64, requires_grad=True)
    extra = torch.arange(sum(sizes)).unsqueeze(-1).expand(-1, 7)
    padded = torch.cat((indices, extra), dim=1)
    validity = torch.cat((valid, torch.zeros_like(extra, dtype=torch.bool)), dim=1)
    first, second = module(x, indices, valid), module(x, padded, validity)
    torch.testing.assert_close(first, second, rtol=1e-13, atol=1e-13)
    probe = torch.randn_like(first)
    grads = torch.autograd.grad((first * probe).sum(), (x, *module.parameters()))
    padded_grads = torch.autograd.grad((second * probe).sum(), (x, *module.parameters()))
    for a, b in zip(grads, padded_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_sparse_attention_gradcheck():
    ops = layers()
    sizes, indices, valid, _ = ops._pyramid_graph(4, (2, 2), 3)
    module = ops._PyramidalAttention(4, 2, 0).double().eval()
    x = torch.randn(1, sum(sizes), 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: module(z, indices, valid), (x,), fast_mode=True)


@pytest.mark.parametrize("dropout_name", ["attn_dropout", "proj_dropout"])
def test_both_attention_dropout_sites_are_active_and_disabled_in_eval(dropout_name):
    ops = layers()
    sizes, indices, valid, _ = ops._pyramid_graph(8, (2, 2, 2), 3)
    module = ops._PyramidalAttention(4, 2, 0).double()
    module.out_proj.bias.data.zero_()
    getattr(module, dropout_name).p = 1
    x = torch.randn(2, sum(sizes), 4, dtype=torch.float64)
    torch.testing.assert_close(module(x, indices, valid), norm(x, module.norm))
    module.eval()
    torch.testing.assert_close(module(x, indices, valid), dense_attention(module, x, dense_graph(sizes, (2, 2, 2), 3)))


def test_ffn_dropout_is_active_only_during_training():
    ops = layers()
    sizes, indices, valid, _ = ops._pyramid_graph(8, (2, 2, 2), 3)
    module = ops._PyramidEncoderLayer(4, 2, 7, 0).double()
    module.ff_dropout2.p = 1
    x = torch.randn(2, sum(sizes), 4, dtype=torch.float64)
    torch.testing.assert_close(module(x, indices, valid), norm(module.attention(x, indices, valid), module.norm))
    module.eval()
    torch.testing.assert_close(module(x, indices, valid), dense_encoder(module, x, dense_graph(sizes, (2, 2, 2), 3)))
