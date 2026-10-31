"""Forecasting contract and mechanism tests for Non-stationary Transformers."""

import math

import numpy as np
import pytest
import torch

from torch_like.models.forecasting.common import ForecastingConfig
from torch_like.models.forecasting.nonstationary_transformer import (
    NonstationaryTransformer,
    NonstationaryTransformerConfig,
)


def make_model(**overrides):
    cfg = dict(
        enc_in=3,
        dec_in=3,
        c_out=3,
        seq_len=8,
        label_len=3,
        pred_len=4,
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=16,
        dropout=0.0,
        factor_hidden=8,
    )
    cfg.update(overrides)
    return NonstationaryTransformer(**cfg)


def test_float64_config_state_roundtrip_and_inherited_batch_helper():
    torch.manual_seed(316)
    model = make_model(label_len=0, factor_hidden=7, eps=2e-5).double().eval()
    cfg = model.config()
    assert cfg["factor_hidden"] == 7 and cfg["eps"] == 2e-5
    restored = NonstationaryTransformer.from_config(cfg).double().eval()
    restored.load_state_dict(model.state_dict())
    assert restored.config() == cfg
    x = torch.randn(2, 8, 3, dtype=torch.float64)
    batch = model.make_forecast_batch(x)
    assert batch.x_dec.shape == (2, 4, 3)
    output = model.predict(batch)
    assert output.shape == (2, 4, 3) and output.dtype == torch.float64
    torch.testing.assert_close(restored.predict(restored.make_forecast_batch(x)), output, rtol=0, atol=0)


@pytest.mark.parametrize("config_type", [ForecastingConfig, NonstationaryTransformerConfig])
def test_standard_forecasting_config_and_dataclass_acceptance(config_type):
    cfg = config_type(enc_in=2, dec_in=2, c_out=2, seq_len=6, label_len=0, pred_len=3, d_model=8, n_heads=2)
    model = NonstationaryTransformer.from_config(cfg).eval()
    assert model(torch.randn(1, 6, 2)).shape == (1, 3, 2)


def test_specialized_config_defaults_match_constructor_defaults():
    dimensions = dict(enc_in=2, dec_in=2, c_out=2, seq_len=6, label_len=0, pred_len=3)
    cfg = NonstationaryTransformerConfig(**dimensions)
    direct = NonstationaryTransformer(**dimensions)
    for name in ("d_model", "n_heads", "e_layers", "d_layers", "d_ff", "dropout", "factor_hidden", "eps"):
        assert getattr(cfg, name) == direct.config()[name]


@pytest.mark.parametrize(
    "dimension", ["enc_in", "seq_len", "label_len", "pred_len", "d_model", "n_heads", "factor_hidden"]
)
def test_numpy_integer_dimensions_work_with_public_constructor(dimension):
    cfg = make_model().config()
    cfg[dimension] = np.int64(cfg[dimension])
    model = NonstationaryTransformer(**cfg).eval()
    assert model(torch.randn(1, 8, 3)).shape == (1, 4, 3)


def test_forecast_backward_optimizer_and_both_factor_projector_gradients():
    torch.manual_seed(317)
    model = make_model().double().train()
    x = torch.randn(2, 8, 3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [p.detach().clone() for p in model.parameters()]
    output = model(x)
    assert output.shape == (2, 4, 3) and torch.isfinite(output).all()
    (output.square().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.tau_projector.conv.weight.grad.abs().sum() > 0
    assert model.delta_projector.conv.weight.grad.abs().sum() > 0
    optimizer.step()
    assert any(not torch.equal(p, previous) for p, previous in zip(model.parameters(), before, strict=True))


@pytest.mark.parametrize("label_len", [0, 1, 8])
def test_decoder_rebuilds_known_prefix_and_zero_future_without_target_leakage(label_len):
    torch.manual_seed(318)
    model = make_model(label_len=label_len).eval()
    x = torch.randn(2, 8, 3)
    baseline = model(x)
    poisoned = model(x, None, torch.full((2, label_len + 4, 3), float("nan")), None)
    torch.testing.assert_close(baseline, poisoned, rtol=0, atol=0)
    assert torch.isfinite(poisoned).all()


def test_observed_population_statistics_invert_normalization_with_correct_order():
    torch.manual_seed(319)
    model = make_model().double().eval()
    x = torch.tensor(
        [
            [
                [1.0, 2.0, -2.0],
                [3.0, 4.0, -2.0],
                [5.0, 6.0, -2.0],
                [7.0, 8.0, -2.0],
                [9.0, 10.0, -2.0],
                [11.0, 12.0, -2.0],
                [13.0, 14.0, -2.0],
                [15.0, 16.0, -2.0],
            ]
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.fill_(1.0)
    expected_mean = torch.tensor([8.0, 9.0, -2.0], dtype=x.dtype)
    expected_scale = torch.tensor([math.sqrt(21.0 + 1e-5), math.sqrt(21.0 + 1e-5), math.sqrt(1e-5)], dtype=x.dtype)
    expected = (expected_mean + expected_scale).expand(1, 4, 3)
    torch.testing.assert_close(model(x), expected, rtol=1e-12, atol=1e-12)


def test_constant_series_is_finite_and_zero_normalized_head_returns_observed_mean():
    model = make_model().double().eval()
    constants = torch.tensor([2.0, -4.0, 0.5], dtype=torch.float64)
    x = constants.reshape(1, 1, 3).expand(2, 8, 3)
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.zero_()
    torch.testing.assert_close(model(x), constants.expand(2, 4, 3), rtol=0, atol=1e-12)


def test_future_time_features_are_known_and_causally_isolated():
    torch.manual_seed(320)
    model = make_model().double().eval()
    x = torch.randn(2, 8, 3, dtype=torch.float64)
    history_markers = torch.randn(2, 8, 4, dtype=torch.float64)
    decoder_markers = torch.randn(2, 7, 4, dtype=torch.float64)
    baseline = model(x, history_markers, None, decoder_markers)
    future_changed = decoder_markers.clone()
    future_changed[:, -1, :] += 100.0
    changed = model(x, history_markers, None, future_changed)
    torch.testing.assert_close(baseline[:, :-1, :], changed[:, :-1, :], rtol=0, atol=1e-12)
    assert not torch.allclose(baseline[:, -1, :], changed[:, -1, :])
    assert not torch.allclose(baseline, model(x, None, None, None))


def test_learned_tau_and_nonconstant_delta_actually_affect_forecasts():
    torch.manual_seed(321)
    model = make_model().double().eval()
    x = torch.randn(2, 8, 3, dtype=torch.float64)
    baseline = model(x)
    with torch.no_grad():
        model.delta_projector.mlp[-1].bias[0] += 7.0
    changed_delta = model(x)
    assert not torch.allclose(changed_delta, baseline, atol=1e-7)
    with torch.no_grad():
        model.tau_projector.mlp[-1].bias += 1.5
    changed_tau = model(x)
    assert not torch.allclose(changed_tau, changed_delta, atol=1e-7)


def test_attention_masks_are_true_keep_and_malformed_masks_raise():
    model = make_model().eval()
    x = torch.randn(2, 8, 3)
    keep = torch.eye(8, dtype=torch.bool)
    selective = model(x, enc_self_mask=keep)
    assert selective.shape == (2, 4, 3)
    with pytest.raises(ValueError, match="mask|allowed"):
        model(x, enc_self_mask=torch.zeros(8, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask"):
        model(x, dec_enc_mask=torch.ones(7, 7, dtype=torch.bool))


def test_output_attention_flag_keeps_tensor_forecasting_abi():
    model = make_model(output_attention=True).eval()
    assert isinstance(model(torch.randn(1, 8, 3)), torch.Tensor)


@pytest.mark.parametrize(
    "overrides",
    [
        {"enc_in": 0},
        {"dec_in": 2},
        {"c_out": 2},
        {"seq_len": 1},
        {"label_len": -1},
        {"label_len": 9},
        {"pred_len": 0},
        {"d_model": 7},
        {"n_heads": 0},
        {"e_layers": 0},
        {"d_layers": 0},
        {"d_ff": 0},
        {"factor_hidden": 0},
        {"dropout": 1.0},
        {"eps": 0.0},
        {"eps": 1e-80},
        {"eps": 1e80},
        {"eps": float("nan")},
    ],
)
def test_invalid_config_is_rejected_before_allocation(overrides):
    with pytest.raises((ValueError, TypeError)):
        make_model(**overrides)


@pytest.mark.parametrize("input_kind", ["wrong_length", "wrong_channels", "empty_batch", "nonfinite", "half", "dtype"])
def test_invalid_observation_contract_is_rejected(input_kind):
    model = make_model()
    x = torch.randn(2, 8, 3)
    if input_kind == "wrong_length":
        x = x[:, :7]
    elif input_kind == "wrong_channels":
        x = x[:, :, :2]
    elif input_kind == "empty_batch":
        x = x[:0]
    elif input_kind == "nonfinite":
        x[0, 0, 0] = float("nan")
    elif input_kind == "half":
        x = x.half()
    else:
        x = x.double()
    with pytest.raises((ValueError, TypeError), match="input|dtype|length|finite|device|shape"):
        model(x)


@pytest.mark.parametrize("marker", ["bad_encoder", "bad_decoder", "nonfinite_encoder", "nonfinite_decoder"])
def test_invalid_time_marker_contract_is_rejected(marker):
    model = make_model().eval()
    x = torch.randn(2, 8, 3)
    enc = torch.zeros(2, 8, 4)
    dec = torch.zeros(2, 7, 4)
    if marker == "bad_encoder":
        enc = enc[:, :, :3]
    elif marker == "bad_decoder":
        dec = dec[:, :6]
    elif marker == "nonfinite_encoder":
        enc[0, 0, 0] = float("nan")
    else:
        dec[0, -1, 0] = float("nan")
    with pytest.raises((ValueError, TypeError), match="marker|shape|finite"):
        model(x, enc, None, dec)
