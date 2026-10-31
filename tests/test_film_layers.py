"""Independent small mathematical oracles for FiLM's defining operators."""

import pytest
import torch

from torch_like.models.forecasting.film_layers import _FrequencyEnhancedLayer, _LegendreProjection


def test_bilinear_lmu_discretization_has_correct_signs_and_orientation():
    # Continuous A=[[-1,-1],[3,-3]], B=[1,-3], dt=1/2.
    layer = _LegendreProjection(order=2, window=2, pred_len=2).double()
    torch.testing.assert_close(layer.A, torch.tensor([[9, -4], [12, 1]], dtype=torch.float64) / 19)
    torch.testing.assert_close(layer.B, torch.tensor([10, -12], dtype=torch.float64) / 19)


def test_zero_initialized_recurrence_matches_hand_calculation():
    layer = _LegendreProjection(order=1, window=2, pred_len=2).double()
    x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64, requires_grad=True)
    actual = layer(x)
    expected = torch.tensor([[[[0.4, 1.04, 1.824]]]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)
    actual[..., -1].sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[0.144, 0.24, 0.4]]], dtype=torch.float64))


def test_legendre_grid_orientation_and_horizon_reconstruction():
    layer = _LegendreProjection(order=3, window=4, pred_len=2).double()
    basis = torch.tensor([[1, 0, -0.5], [1, -0.5, -0.125]], dtype=torch.float64)
    torch.testing.assert_close(layer.basis, basis, rtol=0, atol=0)
    coeff = torch.tensor([[[2.0, 4.0, 8.0]]], dtype=torch.float64)
    torch.testing.assert_close(layer.reconstruct(coeff), torch.tensor([[[-2.0, -1.0]]], dtype=torch.float64))


@pytest.mark.parametrize("length", [1, 5, 6])
def test_full_spectrum_identity_preserves_odd_even_and_single_step(length):
    layer = _FrequencyEnhancedLayer(order=2, modes=length // 2 + 1).double()
    with torch.no_grad():
        layer.weight_real.zero_()
        layer.weight_imag.zero_()
        layer.weight_real[0, 0, :].fill_(1)
        layer.weight_real[1, 1, :].fill_(1)
    x = torch.randn(2, 3, 2, length, dtype=torch.float64)
    torch.testing.assert_close(layer(x), x, rtol=1e-12, atol=1e-12)


def test_dc_cutoff_removes_time_variation_but_keeps_average():
    layer = _FrequencyEnhancedLayer(order=1, modes=1).double()
    with torch.no_grad():
        layer.weight_real.fill_(1)
        layer.weight_imag.zero_()
    x = torch.tensor([[[[1.0, 2.0, 3.0, 6.0]]]], dtype=torch.float64)
    torch.testing.assert_close(layer(x), torch.full_like(x, 3), rtol=0, atol=0)


def test_imaginary_weight_rotates_phase_and_mixes_coefficient_not_channel():
    layer = _FrequencyEnhancedLayer(order=2, modes=2).double()
    with torch.no_grad():
        layer.weight_real.zero_()
        layer.weight_imag.zero_()
        layer.weight_imag[0, 1, 1] = 1
    x = torch.zeros(1, 2, 2, 4, dtype=torch.float64)
    x[0, 0, 0] = torch.tensor([1, 0, -1, 0])
    expected = torch.zeros_like(x)
    expected[0, 0, 1] = torch.tensor([0, -1, 0, 1])
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


def test_spectral_filter_gradcheck_including_real_and_imaginary_weights():
    layer = _FrequencyEnhancedLayer(order=2, modes=2).double()
    x = torch.randn(1, 1, 2, 5, dtype=torch.float64, requires_grad=True)
    from torch.func import functional_call

    def operation(values, real, imag):
        return functional_call(layer, {"weight_real": real, "weight_imag": imag}, (values,))

    assert torch.autograd.gradcheck(operation, (x, layer.weight_real, layer.weight_imag))


def test_projection_buffers_roundtrip_without_trainable_memory_state():
    layer = _LegendreProjection(order=4, window=12, pred_len=3).double()
    assert not list(layer.parameters())
    restored = _LegendreProjection(order=4, window=12, pred_len=3).double()
    restored.load_state_dict(layer.state_dict())
    x = torch.randn(2, 3, 7, dtype=torch.float64)
    torch.testing.assert_close(restored(x), layer(x), rtol=0, atol=0)
