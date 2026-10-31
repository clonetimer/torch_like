"""FiLM forecasting semantics, numerical boundaries and trainability."""

import json

import pytest
import torch

from torch_like.models.forecasting.film import FiLM, FiLMConfig


def make_model(**changes):
    config = dict(enc_in=3, dec_in=3, c_out=3, seq_len=9, label_len=0, pred_len=4, memory_order=4, modes=3)
    config.update(changes)
    return FiLM(**config)


def set_scalar_filters(model):
    with torch.no_grad():
        for layer in model.filters:
            layer.weight_real.fill_(1)
            layer.weight_imag.zero_()
        model.scale_fusion.weight.fill_(1 / len(model.scales))
        model.scale_fusion.bias.zero_()


def test_multiscale_dc_forecast_matches_independent_rational_fixture():
    model = make_model(enc_in=1, c_out=1, seq_len=4, pred_len=2, memory_order=1, scales=(1, 2), use_norm=False)
    model.double()
    set_scalar_filters(model)
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float64)
    # W2 DC=44/25; W4 DC=6179/6561; equal mean = 443159/328050.
    torch.testing.assert_close(model(x), torch.full((1, 2, 1), 443159 / 328050, dtype=torch.float64))


def test_long_scale_selects_horizon_minus_one_not_last_filtered_state():
    model = make_model(enc_in=1, c_out=1, seq_len=8, pred_len=4, memory_order=1, scales=(2,), use_norm=False)
    model.double()
    set_scalar_filters(model)
    x = torch.arange(1, 9, dtype=torch.float64).reshape(1, 8, 1)
    # Independent closed-form states: c_n=(2/17)*sum(j*(15/17)**(n-j)).
    # Direct DFT with DC+bin1, evaluated at index3, yields this literal.
    # Wrong last-index7 selection would give 2.1023797305256284 instead.
    expected = torch.full((1, 4, 1), 0.7935631294233636, dtype=torch.float64)
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("seq_len,pred_len", [(2, 1), (2, 7), (5, 3), (8, 8), (9, 4)])
@pytest.mark.parametrize("use_norm", [False, True])
def test_short_long_odd_even_horizons_and_finite_optimizer_step(seq_len, pred_len, use_norm):
    torch.manual_seed(314)
    model = make_model(seq_len=seq_len, pred_len=pred_len, label_len=seq_len, use_norm=use_norm)
    x = torch.randn(2, seq_len, 3, requires_grad=True)
    y = model.predict(model.make_forecast_batch(x))
    assert y.shape == (2, pred_len, 3) and torch.isfinite(y).all()
    before = model.filters[0].weight_real.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.filters[0].weight_real.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.filters[0].weight_real)


def test_non_dc_imaginary_weights_receive_real_training_gradients():
    torch.manual_seed(315)
    model = make_model()
    model(torch.randn(2, 9, 3)).square().mean().backward()
    assert all(layer.weight_imag.grad[..., 1:].abs().sum() > 0 for layer in model.filters)


def test_normalization_uses_only_observed_population_statistics_and_detaches_them():
    normalized = make_model(eps=0.5).double()
    raw = make_model(use_norm=False).double()
    raw.load_state_dict(normalized.state_dict())
    x = torch.randn(2, 9, 3, dtype=torch.float64, requires_grad=True)
    mean = x.mean(1, keepdim=True).detach()
    scale = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 0.5).detach()
    expected = raw((x - mean) / scale) * scale + mean
    actual = normalized(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_constant_channels_and_translation_equivariance():
    model = make_model().double()
    offsets = torch.tensor([1.5, -4.0, 0.0], dtype=torch.float64).reshape(1, 1, 3)
    torch.testing.assert_close(model(offsets.expand(2, 9, 3)), offsets.expand(2, 4, 3))
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    torch.testing.assert_close(model(x + offsets), model(x) + offsets)


def test_channels_are_independent_and_batch_state_is_not_retained():
    model = make_model().double()
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    baseline = model(x)
    changed = x.clone()
    changed[:, :, 1] += torch.linspace(0, 20, 9)
    torch.testing.assert_close(model(changed)[..., [0, 2]], baseline[..., [0, 2]], rtol=0, atol=0)
    model(torch.randn_like(x))
    torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)


def test_future_decoder_values_and_unused_calendar_masks_cannot_leak():
    model = make_model(dec_in=2)
    x = torch.randn(2, 9, 3)
    bad = torch.full((2, 4, 2), float("nan"))
    torch.testing.assert_close(model(x, bad, bad, bad, bad, bad, bad), model(x), rtol=0, atol=0)


def test_json_configuration_and_float64_state_roundtrip():
    model = make_model(scales=[1, 3], dec_in=2, label_len=2).double()
    config = model.config()
    restored = FiLM.from_config(json.loads(json.dumps(config))).double()
    restored.load_state_dict(model.state_dict())
    assert restored.config() == config
    assert all(value.dtype == torch.float64 for value in model.state_dict().values())
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_default_config_matches_constructor_and_tuple_scales_serialize_consistently():
    common = dict(enc_in=1, dec_in=1, c_out=1, seq_len=4, label_len=0, pred_len=2)
    direct = FiLM(**common)
    configured = FiLM.from_config(FiLMConfig(**common))
    restored = FiLM.from_config(json.loads(json.dumps(direct.config())))
    assert direct.config() == configured.config() == restored.config()
    assert direct(torch.randn(1, 4, 1)).shape == (1, 2, 1)


def test_cpu_autocast_does_not_lower_recurrence_or_fft_precision():
    model = make_model()
    x = torch.randn(2, 9, 3)
    expected = model(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seq_len": 1},
        {"seq_len": True},
        {"seq_len": 2.5},
        {"pred_len": 0},
        {"pred_len": True},
        {"enc_in": 0},
        {"dec_in": 0},
        {"c_out": 2},
        {"label_len": -1},
        {"label_len": 10},
        {"label_len": True},
        {"memory_order": 0},
        {"memory_order": True},
        {"modes": 0},
        {"modes": 1.5},
        {"scales": []},
        {"scales": [1, 1]},
        {"scales": [0, 2]},
        {"scales": [True]},
        {"scales": [1.5]},
        {"scales": "124"},
        {"scales": None},
        {"use_norm": 1},
        {"eps": 0},
        {"eps": True},
        {"eps": float("nan")},
        {"eps": float("inf")},
        {"eps": 1e-100},
        {"eps": 1e100},
    ],
)
def test_invalid_settings_fail_clearly(overrides):
    with pytest.raises(ValueError):
        make_model(**overrides)


@pytest.mark.parametrize("shape", [(2, 9), (0, 9, 3), (2, 8, 3), (2, 9, 2)])
def test_invalid_input_shapes_fail_clearly(shape):
    with pytest.raises(ValueError):
        make_model()(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.complex64, torch.float64])
def test_unsupported_or_mismatched_input_dtypes_fail_clearly(dtype):
    with pytest.raises(TypeError):
        make_model()(torch.zeros(2, 9, 3, dtype=dtype))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_is_rejected(bad):
    x = torch.zeros(2, 9, 3)
    x[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        make_model()(x)


def test_wrong_input_type_and_device_are_rejected_before_computation():
    model = make_model()
    with pytest.raises(TypeError, match="Tensor"):
        model([])
    with pytest.raises(ValueError, match="device"):
        model(torch.empty(2, 9, 3, device="meta"))


@pytest.mark.parametrize("eps", [1e-45, 1e-5, 3e38])
@pytest.mark.parametrize("kind", ["constant_mean_overflow", "variance_overflow"])
def test_finite_observations_with_overflowing_statistics_raise_before_training(eps, kind):
    model = make_model(enc_in=1, c_out=1, eps=eps)
    if kind == "constant_mean_overflow":
        x = torch.full((1, 9, 1), 1e38, requires_grad=True)
    else:
        x = torch.linspace(-1e20, 1e20, 9).reshape(1, 9, 1).requires_grad_()
    assert torch.isfinite(x).all()
    with pytest.raises(ValueError, match="normalization.*numerical range"):
        model.predict(model.make_forecast_batch(x))
    assert x.grad is None and all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("eps", [1e-45, 3e38])
def test_accepted_epsilon_endpoints_keep_ordinary_constants_and_gradients_finite(eps):
    model = make_model(eps=eps)
    x = torch.full((2, 9, 3), 2.0, requires_grad=True)
    y = model(x)
    torch.testing.assert_close(y, torch.full((2, 4, 3), 2.0))
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
