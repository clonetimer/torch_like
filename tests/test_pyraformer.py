"""Forecasting/CSCM contracts: independent math, data isolation and failure boundaries."""

import copy
import importlib
import importlib.util
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_pyraformer_attention import dense_encoder, norm
from test_pyraformer_graph import dense_graph, layers
from torch.utils._python_dispatch import TorchDispatchMode


def public():
    name = "torch_like.models.forecasting.pyraformer"
    assert importlib.util.find_spec(name) is not None, "Pyraformer public model is missing"
    return importlib.import_module(name)


def config(**updates):
    result = dict(
        enc_in=2,
        dec_in=2,
        c_out=2,
        seq_len=22,
        label_len=2,
        pred_len=3,
        d_model=6,
        n_heads=2,
        e_layers=2,
        d_ff=9,
        dropout=0.0,
        window_size=(3, 2, 2),
        d_bottleneck=3,
    )
    result.update(updates)
    return result


def cscm_oracle(module, x):
    """Unfold/einsum construction and explicit BN, independent of module forward."""
    state = F.linear(x, module.down.weight, module.down.bias).transpose(1, 2)
    levels, stats = [], []
    for conv, bn, factor in zip(module.convs, module.bns, module.factors, strict=True):
        windows = state.unfold(-1, factor, factor)
        state = torch.einsum("bcsk,ock->bos", windows, conv.weight) + conv.bias[None, :, None]
        if module.training:
            mean, variance = state.mean((0, 2)), state.var((0, 2), unbiased=False)
            samples = state.shape[0] * state.shape[-1]
            stats.append(
                (
                    0.9 * bn.running_mean + 0.1 * mean.detach(),
                    0.9 * bn.running_var + 0.1 * variance.detach() * samples / (samples - 1),
                )
            )
        else:
            mean, variance = bn.running_mean, bn.running_var
        state = (state - mean[None, :, None]) / (variance[None, :, None] + bn.eps).sqrt()
        state = state * bn.weight[None, :, None] + bn.bias[None, :, None]
        state = F.elu(state)
        levels.append(state.transpose(1, 2))
    coarse = F.linear(torch.cat(levels, 1), module.up.weight, module.up.bias)
    return norm(torch.cat((x, coarse), 1), module.norm), stats


@pytest.mark.parametrize("training", [False, True])
def test_cscm_matches_explicit_convolution_batchnorm_and_gradients(training):
    torch.manual_seed(30)
    ops = layers()
    assert hasattr(ops, "_BottleneckCSCM"), "Bottleneck CSCM is missing"
    module = ops._BottleneckCSCM(6, 3, (3, 2, 2)).double().train(training)
    oracle = copy.deepcopy(module)
    x = torch.randn(2, 23, 6, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, stats = cscm_oracle(oracle, y)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * probe).sum(), (x, *module.parameters()))
    references = torch.autograd.grad((expected * probe).sum(), (y, *oracle.parameters()))
    for a, e in zip(grads, references, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    if training:
        for bn, (mean, variance) in zip(module.bns, stats, strict=True):
            torch.testing.assert_close(bn.running_mean, mean)
            torch.testing.assert_close(bn.running_var, variance)
            assert bn.num_batches_tracked.item() == 1


def test_cscm_drops_tail_from_coarse_construction_but_preserves_fine_nodes():
    ops = layers()
    assert hasattr(ops, "_BottleneckCSCM"), "Bottleneck CSCM is missing"
    module = ops._BottleneckCSCM(4, 3, (3, 2, 2)).double().eval()
    x = torch.randn(2, 23, 4, dtype=torch.float64)
    changed = x.clone()
    changed[:, 21:, 0] += 3
    before, after = module(x), module(changed)
    torch.testing.assert_close(before[:, 23:], after[:, 23:], rtol=0, atol=0)
    assert not torch.allclose(before[:, 21:23], after[:, 21:23])


def position_oracle(length, width, dtype):
    # Scalar formula, not the production table or its vectorized construction.
    return torch.tensor(
        [
            [
                math.sin(t / 10000 ** (2 * (j // 2) / width))
                if j % 2 == 0
                else math.cos(t / 10000 ** (2 * (j // 2) / width))
                for j in range(width)
            ]
            for t in range(length)
        ],
        dtype=dtype,
    ).unsqueeze(0)


def forecast_oracle(model, observed, history_marks, future_marks):
    values = torch.cat((observed, torch.zeros_like(observed[:, :1])), 1)
    circular = torch.cat((values[:, -1:], values, values[:, :1]), 1).transpose(1, 2)
    embedded = F.conv1d(circular, model.value_embedding.weight, model.value_embedding.bias).transpose(1, 2)
    marks = torch.cat((history_marks, future_marks[:, model.label_len : model.label_len + 1]), 1)
    embedded = embedded + F.linear(marks, model.time_embedding.weight, model.time_embedding.bias)
    embedded = embedded + position_oracle(model.seq_len + 1, embedded.shape[-1], observed.dtype)
    state, _ = cscm_oracle(model.cscm, embedded)
    adjacency = dense_graph((23, 7, 3, 1), (3, 2, 2), 3)
    for layer in model.encoder:
        state = dense_encoder(layer, state, adjacency)
    # Literal last ancestors for history22 + one zero-value token.
    flat = state[:, [22, 29, 32, 33], :].flatten(1)
    return F.linear(flat, model.head.weight).reshape(observed.shape[0], 3, 2)


def test_full_forecast_and_input_parameter_gradients_match_independent_dense_oracle():
    torch.manual_seed(31)
    model = public().Pyraformer(**config()).double().eval()
    oracle = copy.deepcopy(model)
    x = torch.randn(2, 22, 2, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    history, future = torch.randn(2, 22, 4, dtype=torch.float64), torch.randn(2, 5, 4, dtype=torch.float64)
    actual = model(x, history, None, future)
    expected = forecast_oracle(oracle, y, history, future)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    # Position buffer starts in model default precision; align that one deliberate
    # rounding difference before demanding double-precision gradient equality.
    model.position.copy_(position_oracle(23, 6, torch.float64))
    actual = model(x, history, None, future)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    probe = torch.randn_like(actual)
    a_grads = torch.autograd.grad((actual * probe).sum(), (x, *model.parameters()))
    e_grads = torch.autograd.grad((expected * probe).sum(), (y, *oracle.parameters()))
    for a, e in zip(a_grads, e_grads, strict=True):
        torch.testing.assert_close(a, e, rtol=3e-9, atol=3e-10)


def test_decoder_values_prefix_marks_and_later_future_marks_are_not_consumed():
    model = public().Pyraformer(**config()).eval()
    x, history, future = torch.randn(2, 22, 2), torch.randn(2, 22, 4), torch.randn(2, 5, 4)
    expected = model(x, history, torch.randn(2, 5, 2), future)
    ignored = future.clone()
    ignored[:, :2] = float("nan")
    ignored[:, 3:] = float("inf")
    actual = model(x, history, torch.full((2, 5, 2), float("nan")), ignored)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed = future.clone()
    changed[:, 2, 0] += 2
    assert not torch.allclose(model(x, history, None, changed), expected)


def test_missing_markers_are_exactly_zero_markers():
    model = public().Pyraformer(**config(label_len=0, dec_in=1)).eval()
    x = torch.randn(2, 22, 2)
    torch.testing.assert_close(model(x), model(x, torch.zeros(2, 22, 4), None, torch.zeros(2, 3, 4)), rtol=0, atol=0)


def test_marker_gradient_only_reaches_observed_and_first_future_covariates():
    model = public().Pyraformer(**config()).double().eval()
    x = torch.randn(2, 22, 2, dtype=torch.float64)
    history = torch.randn(2, 22, 4, dtype=torch.float64, requires_grad=True)
    future = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    model(x, history, None, future).square().sum().backward()
    assert history.grad.abs().sum() > 0 and future.grad[:, 2].abs().sum() > 0
    assert torch.count_nonzero(future.grad[:, :2]) == torch.count_nonzero(future.grad[:, 3:]) == 0


@pytest.mark.parametrize("batch,length", [(2, 7), (1, 15)])
def test_train_batchnorm_accepts_more_than_one_value_per_channel(batch, length):
    model = public().Pyraformer(**config(seq_len=length, window_size=(2, 2, 2))).train()
    assert torch.isfinite(model(torch.randn(batch, length, 2))).all()
    assert all(bn.num_batches_tracked.item() == 1 for bn in model.cscm.bns)


def test_degenerate_training_rejects_before_any_batchnorm_state_mutation():
    model = public().Pyraformer(**config()).train()
    before = {k: v.clone() for k, v in model.named_buffers()}
    with pytest.raises(ValueError, match="BatchNorm"):
        model(torch.randn(1, 22, 2))
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    model.eval()
    assert torch.isfinite(model(torch.randn(1, 22, 2))).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_training_produces_finite_gradients_and_an_optimizer_update(dtype):
    torch.manual_seed(33)
    model = public().Pyraformer(**config()).to(dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(3, 22, 2, dtype=dtype)
    history = torch.randn(3, 22, 4, dtype=dtype)
    before = {name: p.clone() for name, p in model.named_parameters()}
    loss = model(x, history).square().mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())


def test_json_roundtrip_and_config_dataclass_produce_identical_state_behavior():
    mod = public()
    model = mod.Pyraformer.from_config(mod.PyraformerConfig(**config())).eval()
    restored = mod.Pyraformer.from_config(json.loads(json.dumps(model.config()))).eval()
    restored.load_state_dict(model.state_dict())
    assert model.config()["window_size"] == [3, 2, 2]
    assert model.config() == restored.config()
    x = torch.randn(2, 22, 2)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


def test_odd_width_and_beyond_reference_position_limit_are_supported():
    model = public().Pyraformer(**config(seq_len=5001, d_model=5, n_heads=1)).eval()
    expected = position_oracle(5002, 5, model.position.dtype)
    torch.testing.assert_close(model.position, expected)
    with torch.no_grad():
        assert model(torch.randn(1, 5001, 2)).shape == (1, 3, 2)


def test_default_config_and_constructor_agree_and_default_window_is_valid():
    mod = public()
    required = dict(enc_in=1, dec_in=1, c_out=1, seq_len=63, label_len=0, pred_len=1)
    direct = mod.Pyraformer(**required).eval()
    via_config = mod.Pyraformer.from_config(mod.PyraformerConfig(**required)).eval()
    assert direct.config() == via_config.config()
    assert direct(torch.randn(1, 63, 1)).shape == (1, 1, 1)


def test_numpy_integer_factors_cannot_overflow_the_nonempty_level_guard():
    with pytest.raises(ValueError, match="coarse|seq_len"):
        public().Pyraformer(**config(window_size=(np.int8(16), np.int8(16), np.int8(2))))


def test_numpy_sequence_length_does_not_wrap_when_appending_prediction_token():
    model = public().Pyraformer(**config(seq_len=np.int8(127))).eval()
    assert model(torch.randn(1, 127, 2)).shape == (1, 3, 2)


@pytest.mark.parametrize(
    "integer_type", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
)
def test_all_accepted_integer_types_match_python_int_geometry_and_forecasts(integer_type):
    native = config(seq_len=63, d_model=64, n_heads=4, d_ff=96, d_bottleneck=16, window_size=(4, 4, 4), inner_size=3)
    typed = {name: integer_type(value) if isinstance(value, int) else value for name, value in native.items()}
    typed["window_size"] = tuple(integer_type(value) for value in native["window_size"])
    torch.manual_seed(341)
    reference = public().Pyraformer(**native).eval()
    torch.manual_seed(341)
    model = public().Pyraformer(**typed).eval()
    assert model.head.in_features == 256
    x = torch.randn(2, 63, 2)
    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("enc_in", 0),
        ("enc_in", True),
        ("dec_in", 0),
        ("c_out", 0),
        ("c_out", 3),
        ("seq_len", 0),
        ("seq_len", True),
        ("seq_len", 10),
        ("label_len", -1),
        ("label_len", 23),
        ("label_len", 0.5),
        ("label_len", True),
        ("pred_len", 0),
        ("pred_len", 1.5),
        ("d_model", 1),
        ("d_model", 5),
        ("n_heads", 0),
        ("n_heads", True),
        ("e_layers", 0),
        ("e_layers", 1.5),
        ("d_ff", 0),
        ("d_bottleneck", 0),
        ("window_size", (2, 2)),
        ("window_size", (2, 2, 2, 2)),
        ("window_size", "2,2,2"),
        ("window_size", (2, True, 2)),
        ("window_size", (2, 1, 2)),
        ("window_size", (2, 2.5, 2)),
        ("inner_size", 0),
        ("inner_size", 2),
        ("inner_size", True),
        ("dropout", -0.1),
        ("dropout", 1),
        ("dropout", float("nan")),
        ("dropout", True),
        ("activation", "relu"),
        ("output_attention", True),
    ],
)
def test_rejects_invalid_or_unsupported_configuration(key, value):
    with pytest.raises(ValueError, match=key if key != "seq_len" else "seq_len|coarse"):
        public().Pyraformer(**config(**{key: value}))


@pytest.mark.parametrize("which", ["enc_self_mask", "dec_self_mask", "dec_enc_mask"])
def test_external_masks_are_not_silently_ignored(which):
    model = public().Pyraformer(**config()).eval()
    with pytest.raises(ValueError, match="mask"):
        model(torch.randn(2, 22, 2), **{which: torch.zeros(22, 22)})


@pytest.mark.parametrize("shape", [(2, 21, 2), (2, 22, 3), (22, 2), (0, 22, 2)])
def test_rejects_wrong_or_empty_observation_shape(shape):
    with pytest.raises(ValueError, match="shape|batch"):
        public().Pyraformer(**config()).eval()(torch.randn(*shape))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.float64])
def test_rejects_unsupported_or_mismatched_input_dtype(dtype):
    with pytest.raises(TypeError, match="dtype|float32|float64"):
        public().Pyraformer(**config()).eval()(torch.ones(2, 22, 2, dtype=dtype))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_observations_before_state_mutation(value):
    model = public().Pyraformer(**config())
    x = torch.randn(2, 22, 2)
    x[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        model(x)
    assert all(bn.num_batches_tracked.item() == 0 for bn in model.cscm.bns)


@pytest.mark.parametrize("future", [False, True])
@pytest.mark.parametrize("problem", ["shape", "dtype", "nonfinite", "type", "device"])
def test_marker_contract(future, problem):
    model = public().Pyraformer(**config()).eval()
    x = torch.randn(2, 22, 2)
    shape = (2, 5 if future else 22, 4)
    marks = torch.zeros(shape)
    if problem == "shape":
        marks = marks[..., :3]
    elif problem == "dtype":
        marks = marks.double()
    elif problem == "nonfinite":
        marks[:, 2 if future else 0] = float("nan")
    elif problem == "type":
        marks = marks.tolist()
    elif problem == "device":
        marks = marks.to("meta")
    with pytest.raises((TypeError, ValueError), match="mark"):
        model(x, **{"x_mark_dec" if future else "x_mark_enc": marks})


def test_rejects_non_tensor_and_wrong_device_observations():
    model = public().Pyraformer(**config()).eval()
    with pytest.raises(TypeError, match="Tensor"):
        model([[[1.0]]])
    with pytest.raises(ValueError, match="device"):
        model(torch.empty(2, 22, 2, device="meta"))


def test_autocast_cannot_silently_change_supported_compute_precision():
    model = public().Pyraformer(**config()).eval()
    x = torch.randn(2, 22, 2)
    expected = model(x)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = model(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_real_execution_never_allocates_square_total_node_tensor():
    total = 34

    class NoDenseSquare(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            outputs = torch.utils._pytree.tree_leaves(result)
            for value in outputs:
                if isinstance(value, torch.Tensor):
                    assert list(value.shape).count(total) < 2, (func, value.shape)
            return result

    with NoDenseSquare():
        model = public().Pyraformer(**config()).double()
        result = model(torch.randn(2, 22, 2, dtype=torch.float64))
        result.square().mean().backward()
