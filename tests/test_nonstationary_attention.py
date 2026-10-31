"""Independent mathematical oracles for the de-stationary attention operator."""

import math

import pytest
import torch
import torch.nn.functional as F

from torch_like.models.forecasting.nonstationary_attention import (
    _DeStationaryAttention,
    _DeStationaryMultiHeadAttention,
)


def _example():
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
    tau = torch.tensor([2.0], dtype=torch.float64)
    delta = torch.tensor([[0.2, -0.6, 0.9]], dtype=torch.float64)
    return q, k, v, tau, delta


def test_per_key_delta_and_tau_match_literal_dot_product_oracle():
    """Catches the dangerous but softmax-invisible per-query delta placement."""
    q, k, v, tau, delta = _example()
    scores = torch.tensor([[2.2, -0.6, 2.9], [0.2, 1.4, 2.9]], dtype=torch.float64) / math.sqrt(2)
    weights = scores.softmax(dim=-1)
    expected = torch.stack((weights[:, 0] + 3 * weights[:, 2], 2 * weights[:, 1] + 4 * weights[:, 2]), dim=-1)
    actual = _DeStationaryAttention()(q, k, v, tau, delta)
    torch.testing.assert_close(actual[0, 0], expected, rtol=1e-12, atol=1e-12)
    plain = _DeStationaryAttention()(q, k, v, tau, torch.zeros_like(delta))
    assert not torch.allclose(actual, plain)
    # Adding the same bias to every key does not change a softmax row.
    torch.testing.assert_close(actual, _DeStationaryAttention()(q, k, v, tau, delta + 17.0), rtol=1e-12, atol=1e-12)


def test_unit_tau_and_zero_delta_reduce_to_normal_scaled_dot_attention():
    torch.manual_seed(417)
    q = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    k = torch.randn(2, 3, 6, 5, dtype=torch.float64)
    v = torch.randn(2, 3, 6, 7, dtype=torch.float64)
    actual = _DeStationaryAttention()(q, k, v, torch.ones(2, dtype=q.dtype))
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_mask_true_means_allowed_and_never_softmaxes_forbidden_keys():
    q, k, v, tau, delta = _example()
    keep = torch.tensor([[True, False, False], [False, True, True]])
    actual = _DeStationaryAttention()(q, k, v, tau, delta, keep_mask=keep)
    torch.testing.assert_close(actual[0, 0, 0], v[0, 0, 0], rtol=0, atol=0)
    row = torch.softmax(torch.tensor([1.4, 2.9], dtype=q.dtype) / math.sqrt(2), dim=0)
    torch.testing.assert_close(actual[0, 0, 1], row[0] * v[0, 0, 1] + row[1] * v[0, 0, 2])


def test_causal_mask_blocks_future_keys_even_with_external_all_true_mask():
    q = torch.ones(1, 1, 3, 2, dtype=torch.float64)
    k = torch.ones(1, 1, 3, 2, dtype=torch.float64)
    v = torch.tensor([[[[1.0], [3.0], [7.0]]]], dtype=torch.float64)
    tau = torch.ones(1, dtype=torch.float64)
    attention = _DeStationaryAttention()
    original = attention(q, k, v, tau, keep_mask=torch.ones(3, 3, dtype=torch.bool), causal=True)
    changed_v = v.clone()
    changed_v[..., 2, :] = 999.0
    changed = attention(q, k, changed_v, tau, causal=True)
    torch.testing.assert_close(original[..., :2, :], changed[..., :2, :], rtol=0, atol=0)
    torch.testing.assert_close(original[0, 0, :, 0], torch.tensor([1.0, 2.0, 11.0 / 3], dtype=q.dtype))


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.tensor([[False, False, False], [True, True, True]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.float32),
    ],
)
def test_invalid_or_fully_blocked_mask_fails_explicitly(bad_mask):
    q, k, v, tau, delta = _example()
    with pytest.raises((ValueError, TypeError), match="mask|allowed|boolean"):
        _DeStationaryAttention()(q, k, v, tau, delta, keep_mask=bad_mask)


def test_batched_head_broadcasting_and_finite_all_factor_gradients():
    torch.manual_seed(311)
    q = torch.randn(2, 3, 4, 2, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 5, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 5, 3, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor([0.8, 2.2], dtype=q.dtype, requires_grad=True)
    delta = torch.randn(2, 5, dtype=q.dtype, requires_grad=True)
    keep = torch.ones(2, 1, 4, 5, dtype=torch.bool)
    keep[0, :, 0, 4] = False
    output = _DeStationaryAttention()(q, k, v, tau, delta, keep_mask=keep)
    assert output.shape == (2, 3, 4, 3)
    output.square().sum().backward()
    for tensor in (q, k, v, tau, delta):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "changed",
    [
        "tau_batch",
        "tau_rank",
        "delta_key_length",
        "delta_batch",
        "unequal_causal_lengths",
    ],
)
def test_factor_shape_and_causal_length_mismatches_fail(changed):
    q, k, v, tau, delta = _example()
    if changed == "tau_batch":
        tau = tau.repeat(2)
    elif changed == "tau_rank":
        tau = tau[:, None, None]
    elif changed == "delta_key_length":
        delta = delta[:, :-1]
    elif changed == "delta_batch":
        delta = delta.repeat(2, 1)
    if changed == "unequal_causal_lengths":
        with pytest.raises(ValueError, match="causal|length"):
            _DeStationaryAttention()(q, k, v, tau, delta, causal=True)
    else:
        with pytest.raises(ValueError, match="tau|delta"):
            _DeStationaryAttention()(q, k, v, tau, delta)


def test_multihead_projections_match_core_attention_when_set_to_identity():
    model = _DeStationaryMultiHeadAttention(d_model=2, n_heads=1, dropout=0).double()
    with torch.no_grad():
        for linear in (model.q_proj, model.k_proj, model.v_proj, model.out_proj):
            linear.weight.copy_(torch.eye(2, dtype=torch.float64))
            linear.bias.zero_()
    q, k, v, tau, delta = _example()
    actual = model(q[:, 0], k[:, 0], v[:, 0], tau, delta)
    expected = _DeStationaryAttention()(q, k, v, tau, delta)[:, 0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_multihead_rejects_invalid_hidden_dimensions():
    with pytest.raises(ValueError, match="d_model|n_heads"):
        _DeStationaryMultiHeadAttention(d_model=7, n_heads=2, dropout=0)
