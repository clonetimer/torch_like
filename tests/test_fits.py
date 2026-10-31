"""Contract and mathematical tests for the independent FITS implementation."""

import math

import pytest
import torch

from torch_like.models.forecasting.common import ForecastingConfig
from torch_like.models.forecasting.fits import FITS, FITSConfig


def make_model(**overrides):
    config = dict(enc_in=3, dec_in=3, c_out=3, seq_len=9, label_len=0, pred_len=4, cut_freq=3)
    config.update(overrides)
    return FITS(**config)


@pytest.mark.parametrize("individual", [False, True])
@pytest.mark.parametrize("use_norm", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_forward_gradient_and_optimizer_step(individual, use_norm, bias):
    torch.manual_seed(171)
    model = make_model(individual=individual, use_norm=use_norm, bias=bias)
    x = torch.randn(2, 9, 3, requires_grad=True)
    output = model.predict(model.make_forecast_batch(x))
    assert output.shape == (2, 4, 3)
    assert torch.isfinite(output).all()
    assert model.reconstruct_and_forecast(x).shape == (2, 13, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [p.detach().clone() for p in model.parameters()]
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.weight_real.grad.abs().sum() > 0
    assert model.weight_imag.grad.abs().sum() > 0
    optimizer.step()
    assert any(not torch.equal(p, previous) for p, previous in zip(model.parameters(), before, strict=True))


@pytest.mark.parametrize("individual", [False, True])
def test_full_config_and_state_roundtrip(individual):
    model = make_model(individual=individual, eps=1e-4, bias=False, dec_in=2, label_len=3).double()
    config = model.config()
    assert config["individual"] == individual
    assert config["dec_in"] == 2
    assert config["eps"] == 1e-4
    restored = FITS.from_config(config).double()
    restored.load_state_dict(model.state_dict())
    assert restored.config() == config
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    assert model.make_forecast_batch(x).x_dec.shape == (2, 7, 2)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    assert all(p.dtype == torch.float64 for p in model.parameters())


@pytest.mark.parametrize("config_class", [ForecastingConfig, FITSConfig])
def test_forecast_config_compatibility(config_class):
    config = config_class(enc_in=2, dec_in=2, c_out=2, seq_len=8, label_len=0, pred_len=3)
    model = FITS.from_config(config)
    assert model.cut_freq == 5
    assert model(torch.randn(1, 8, 2)).shape == (1, 3, 2)


def slow_complex_dft_reference(model, x):
    """Explicit DFT/matrix/IDFT oracle, independent of the implementation's FFTs."""
    results = []
    positions = torch.arange(model.seq_len, dtype=x.dtype)
    frequencies = torch.arange(model.cut_freq, dtype=x.dtype)
    dft = torch.exp(-2j * math.pi * frequencies[:, None] * positions[None, :] / model.seq_len)
    output_positions = torch.arange(model.total_len, dtype=x.dtype)
    output_frequencies = torch.arange(model.total_len, dtype=x.dtype)
    idft = torch.exp(2j * math.pi * output_positions[:, None] * output_frequencies[None, :] / model.total_len)
    for sample in x:
        channels = []
        for c in range(model.enc_in):
            values = sample[:, c]
            mean = values.mean() if model.use_norm else values.new_zeros(())
            scale = torch.sqrt(values.var(unbiased=False) + model.eps) if model.use_norm else values.new_ones(())
            spectrum = dft @ ((values - mean) / scale).to(dft.dtype)
            weight = torch.complex(model.weight_real, model.weight_imag)
            if model.individual:
                weight = weight[c]
            projected = weight @ spectrum
            if model.bias_real is not None:
                bias = torch.complex(model.bias_real, model.bias_imag)
                projected = projected + (bias[c] if model.individual else bias)
            positive = projected.new_zeros(model.total_len // 2 + 1)
            positive[: model.out_freq] = projected
            # A real signal's DC and (for even lengths) Nyquist bins are real.
            positive[0] = positive[0].real.clone()
            if model.total_len % 2 == 0:
                positive[-1] = positive[-1].real.clone()
                negative = positive[1:-1].conj().flip(0)
            else:
                negative = positive[1:].conj().flip(0)
            full_spectrum = torch.cat([positive, negative])
            # IDFT factor 1/(L+H) and length compensation (L+H)/L cancel.
            extension = (idft @ full_spectrum).real / model.seq_len
            channels.append(extension * scale + mean)
        results.append(torch.stack(channels, dim=-1))
    return torch.stack(results)


@pytest.mark.parametrize("seq_len,pred_len", [(7, 4), (8, 4)])
@pytest.mark.parametrize("individual,use_norm", [(False, False), (True, True)])
def test_complex_interpolation_matches_direct_dft_oracle(seq_len, pred_len, individual, use_norm):
    model = make_model(seq_len=seq_len, pred_len=pred_len, individual=individual, use_norm=use_norm).double()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(-0.4 + index / 10, 0.7 + index / 10, parameter.numel(), dtype=torch.float64)
            parameter.copy_(values.reshape(parameter.shape))
    x = torch.linspace(-1, 3, 2 * seq_len * 3, dtype=torch.float64).reshape(2, seq_len, 3)
    x = x + torch.sin(x * 7)
    with torch.no_grad():
        expected = slow_complex_dft_reference(model, x)
        actual = model.reconstruct_and_forecast(x)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(model(x), expected[:, -pred_len:])


@pytest.mark.parametrize("seq_len,pred_len", [(2, 1), (2, 6), (3, 1), (5, 4), (8, 8), (9, 7), (16, 1)])
@pytest.mark.parametrize("default_cutoff", [False, True])
def test_maximal_cutoff_and_odd_even_inverse_fft_lengths(seq_len, pred_len, default_cutoff):
    cut_freq = None if default_cutoff else seq_len // 2 + 1
    model = make_model(seq_len=seq_len, pred_len=pred_len, cut_freq=cut_freq).double()
    assert 1 <= model.out_freq <= (seq_len + pred_len) // 2 + 1
    x = torch.randn(2, seq_len, 3, dtype=torch.float64)
    assert model(x).shape == (2, pred_len, 3)
    assert model.reconstruct_and_forecast(x).shape == (2, seq_len + pred_len, 3)
    with torch.no_grad():
        torch.testing.assert_close(model.reconstruct_and_forecast(x), slow_complex_dft_reference(model, x))


@pytest.mark.parametrize("individual", [False, True])
def test_lowpass_discards_high_frequency_without_normalization(individual):
    model = make_model(seq_len=16, pred_len=7, cut_freq=3, use_norm=False, individual=individual).double()
    time = torch.arange(16, dtype=torch.float64)
    low = torch.cos(2 * math.pi * time / 16)[None, :, None].expand(2, 16, 3)
    high = 5 * torch.sin(2 * math.pi * 5 * time / 16)[None, :, None]
    torch.testing.assert_close(model(low), model(low + high), rtol=1e-10, atol=1e-10)


def test_length_gain_preserves_dc_amplitude_without_normalization():
    model = make_model(seq_len=8, pred_len=13, cut_freq=1, use_norm=False, bias=False).double()
    with torch.no_grad():
        model.weight_real.zero_()
        model.weight_imag.zero_()
        model.weight_real[0, 0] = 1
    x = torch.full((2, 8, 3), 7.5, dtype=torch.float64)
    torch.testing.assert_close(model.reconstruct_and_forecast(x), torch.full((2, 21, 3), 7.5, dtype=torch.float64))


@pytest.mark.parametrize("individual", [False, True])
def test_constant_input_and_per_channel_translation(individual):
    model = make_model(individual=individual).double()
    constant = torch.tensor([1.5, -2, 0.0], dtype=torch.float64).reshape(1, 1, 3)
    torch.testing.assert_close(model(constant.expand(2, 9, 3)), constant.expand(2, 4, 3))
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    torch.testing.assert_close(model(x + constant), model(x) + constant)


@pytest.mark.parametrize("individual", [False, True])
def test_channels_are_independent(individual):
    model = make_model(individual=individual).double()
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    changed = x.clone()
    changed[:, :, 1] += torch.randn(2, 9, dtype=torch.float64) * 20
    torch.testing.assert_close(model(x)[..., [0, 2]], model(changed)[..., [0, 2]], rtol=0, atol=0)


def test_shared_weights_are_channel_permutation_equivariant():
    model = make_model().double()
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    permutation = [2, 0, 1]
    torch.testing.assert_close(model(x[..., permutation]), model(x)[..., permutation])


def test_individual_weights_really_are_distinct_maps():
    model = make_model(individual=True, bias=False, use_norm=False).double()
    with torch.no_grad():
        model.weight_imag.zero_()
        model.weight_real.zero_()
        model.weight_real[0, 1, 1] = 1
        model.weight_real[1, 1, 1] = 2
    x = torch.randn(2, 9, 1, dtype=torch.float64).expand(2, 9, 3)
    output = model(x)
    torch.testing.assert_close(output[..., 1], output[..., 0] * 2)
    assert torch.count_nonzero(output[..., 2]) == 0


def test_future_decoder_values_do_not_enter_forecast():
    model = make_model()
    x = torch.randn(2, 9, 3)
    batch = model.make_forecast_batch(x)
    expected = model.predict(batch)
    batch.x_dec.fill_(float("nan"))
    batch.x_mark_dec.fill_(float("nan"))
    batch.x_mark_enc.fill_(float("nan"))
    torch.testing.assert_close(model.predict(batch), expected, rtol=0, atol=0)


def test_standard_dropout_argument_does_not_change_linear_architecture():
    model = make_model(dropout=0.9)
    x = torch.randn(2, 9, 3)
    train_output = model.train()(x)
    torch.testing.assert_close(model.eval()(x), train_output, rtol=0, atol=0)


def test_cpu_autocast_does_not_lower_complex_fft_precision():
    model = make_model()
    x = torch.randn(2, 9, 3)
    expected = model(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_input_gradient_matches_finite_difference():
    model = make_model(enc_in=1, dec_in=1, c_out=1, seq_len=4, pred_len=3, cut_freq=2).double()
    x = torch.randn(1, 4, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (x,), atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("individual", [False, True])
def test_joint_input_and_all_parameter_gradients_match_finite_difference(individual):
    torch.manual_seed(427)
    model = make_model(
        enc_in=2,
        dec_in=2,
        c_out=2,
        seq_len=4,
        pred_len=3,
        cut_freq=2,
        individual=individual,
    ).double()
    names = tuple(name for name, _ in model.named_parameters())
    assert set(names) == {"weight_real", "weight_imag", "bias_real", "bias_imag"}
    parameters = tuple(parameter.detach().requires_grad_(True) for parameter in model.parameters())
    x = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)

    def functional_forecast(history, *values):
        state = dict(zip(names, values, strict=True))
        return torch.func.functional_call(model, state, (history,), strict=True)

    # Simultaneously perturb the observed history and every real/imaginary
    # weight/bias element, including cross terms and zero DC-imaginary paths.
    assert torch.autograd.gradcheck(functional_forecast, (x, *parameters), atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"enc_in": 0}, "enc_in"),
        ({"dec_in": 0}, "dec_in"),
        ({"c_out": 0}, "c_out"),
        ({"c_out": 1}, "c_out == enc_in"),
        ({"seq_len": 1}, "seq_len"),
        ({"seq_len": 2.5}, "seq_len"),
        ({"seq_len": True}, "seq_len"),
        ({"label_len": -1}, "label_len"),
        ({"label_len": 10}, "label_len"),
        ({"pred_len": 0}, "pred_len"),
        ({"cut_freq": 0}, "cut_freq"),
        ({"cut_freq": 6}, "cut_freq"),
        ({"cut_freq": 1.5}, "cut_freq"),
        ({"cut_freq": True}, "cut_freq"),
        ({"eps": 0}, "eps"),
        ({"eps": -1}, "eps"),
        ({"eps": float("nan")}, "eps"),
        ({"eps": float("inf")}, "eps"),
        ({"eps": True}, "eps"),
        ({"individual": 1}, "individual"),
        ({"use_norm": 0}, "use_norm"),
        ({"bias": 1}, "bias"),
    ],
)
def test_invalid_constructor_parameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_model(**overrides)


@pytest.mark.parametrize("shape", [(2, 9), (2, 8, 3), (2, 9, 2), (0, 9, 3)])
def test_invalid_input_shape(shape):
    with pytest.raises(ValueError):
        make_model()(torch.ones(shape))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float16, torch.bfloat16, torch.complex64])
def test_unsupported_input_dtypes(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        make_model()(torch.ones(2, 9, 3, dtype=dtype))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_is_rejected(value):
    x = torch.ones(2, 9, 3)
    x[0, 1, 2] = value
    with pytest.raises(ValueError, match="finite"):
        make_model()(x)


def test_non_tensor_input_is_rejected():
    with pytest.raises(TypeError, match="torch.Tensor"):
        make_model()([1, 2, 3])


def test_model_input_dtype_mismatch_has_clear_error():
    with pytest.raises(TypeError, match="match model dtype"):
        make_model().double()(torch.ones(2, 9, 3))


def test_model_input_device_mismatch_has_clear_error():
    with pytest.raises(ValueError, match="same device"):
        make_model()(torch.empty(2, 9, 3, device="meta"))
