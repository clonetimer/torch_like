"""GMAN public input ownership, calendar boundaries and complete forecast math."""

import importlib
import json

import numpy as np
import pytest
import torch
from test_gman_layers import assert_gradients, block_oracle, embedding_oracle, mlp_oracle, transform_oracle


def api():
    name = "torch_like.models.forecasting.gman"
    assert importlib.util.find_spec(name) is not None, "GMAN forecaster is missing"
    return importlib.import_module(name)


def config(**overrides):
    result = dict(
        enc_in=3,
        dec_in=3,
        c_out=3,
        seq_len=4,
        label_len=2,
        pred_len=2,
        d_model=4,
        n_heads=2,
        e_layers=1,
        time_slots_per_day=8,
        spatial_embedding=[[-0.5, 0.2, 0.7, 0.0], [0.3, -0.8, 0.1, 0.9], [0.6, 0.4, -0.2, -0.3]],
    )
    result.update(overrides)
    return result


def calendar(batch, length, *, start=0, slots=8, dtype=torch.long):
    indices = torch.arange(start, start + length)
    return torch.stack(((indices // slots) % 7, indices % slots), -1).unsqueeze(0).repeat(batch, 1, 1).to(dtype)


def inputs(model, *, batch=2, dtype=torch.float32):
    x = torch.randn(batch, model.seq_len, model.enc_in, dtype=dtype)
    past = calendar(batch, model.seq_len, slots=model.time_slots_per_day)
    future = calendar(
        batch, model.label_len + model.pred_len, start=model.seq_len - model.label_len, slots=model.time_slots_per_day
    )
    return x, past, future


def full_oracle(model, x, past, future):
    ste = embedding_oracle(model.embedding, model.spatial_embedding, torch.cat((past, future[:, model.label_len :]), 1))
    historical, predicted = ste[:, : model.seq_len], ste[:, model.seq_len :]
    hidden = mlp_oracle(model.input_projection, x[..., None])
    for block in model.encoder:
        hidden = block_oracle(block, hidden, historical)
    hidden = transform_oracle(model.transform, hidden, historical, predicted)
    for block in model.decoder:
        hidden = block_oracle(block, hidden, predicted)
    return mlp_oracle(model.readout, hidden).squeeze(-1)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batch_norm", [False, True])
def test_complete_forecast_and_all_gradients_match_independent_math(masked, batch_norm):
    torch.manual_seed(71)
    model = api().GMAN(**config(temporal_mask=masked, use_batch_norm=batch_norm, e_layers=2)).double().eval()
    x, past, future = inputs(model, dtype=torch.float64)
    x.requires_grad_()
    actual = model(x, past, None, future)
    expected = full_oracle(model, x, past, future)
    assert actual.shape == (2, 2, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    assert_gradients(actual, expected, (x,), model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_training_optimizer_and_batch_prediction_are_real(dtype):
    torch.manual_seed(26)
    model = api().GMAN(**config()).to(dtype=dtype)
    x, past, future = inputs(model, dtype=dtype)
    x.requires_grad_()
    before = model.transform.value.linear.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch = model.make_forecast_batch(x, x_mark_enc=past, x_mark_dec=future)
    prediction = model.predict(batch)
    (prediction - torch.randn_like(prediction)).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.spatial_embedding.grad is None
    optimizer.step()
    assert not torch.equal(before, model.transform.value.linear.weight)
    model.eval()
    torch.testing.assert_close(model.predict(batch), model(x, past, None, future), rtol=0, atol=0)


def test_model_forecast_is_equivariant_to_joint_sensor_and_embedding_permutation():
    torch.manual_seed(415)
    model = api().GMAN(**config()).double().eval()
    x, past, future = inputs(model, dtype=torch.float64)
    order = torch.tensor([2, 0, 1])
    other = api().GMAN(**config(spatial_embedding=model.spatial_embedding[order])).double().eval()
    other.load_state_dict(
        {key: value for key, value in model.state_dict().items() if key != "spatial_embedding"}, strict=False
    )
    torch.testing.assert_close(
        other(x[:, :, order], past, None, future), model(x, past, None, future)[:, :, order], rtol=1e-9, atol=1e-11
    )


def test_spatial_embedding_and_every_future_calendar_are_actual_inputs():
    torch.manual_seed(38)
    model = api().GMAN(**config()).double().eval()
    # Expose upstream dependence: an entirely inactive random output ReLU
    # would legitimately hide both original and perturbed spatial embeddings.
    with torch.no_grad():
        model.readout[0].linear.bias.fill_(2.0)
    x, past, future = inputs(model, dtype=torch.float64)
    expected = model(x, past, None, future)
    for step in range(model.pred_len):
        changed = future.clone()
        changed[:, model.label_len + step, 0] = 6
        assert not torch.allclose(model(x, past, None, changed), expected, rtol=1e-9, atol=1e-11)
    with torch.no_grad():
        model.spatial_embedding[1] += torch.tensor([0.3, -0.5, 2.0, -0.9])
    assert not torch.allclose(model(x, past, None, future), expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("label_len", [0, 2, 4])
def test_decoder_values_and_unused_prefix_marks_cannot_leak_into_prediction(label_len):
    model = api().GMAN(**config(label_len=label_len)).eval()
    x, past, future = inputs(model)
    expected = model(x, past, None, future)
    poisoned = future.clone()
    poisoned[:, :label_len] = -900
    for decoder in (torch.full((2, label_len + 2, 3), torch.nan), object(), torch.tensor(0.0)):
        torch.testing.assert_close(model(x, past, decoder, poisoned), expected, rtol=0, atol=0)


def test_noncontiguous_input_and_calendar_preserve_semantics():
    model = api().GMAN(**config()).eval()
    x, past, future = inputs(model)
    views = [value.repeat_interleave(2, dim=-1)[..., ::2] for value in (x, past, future)]
    assert all(not value.is_contiguous() for value in views)
    torch.testing.assert_close(model(views[0], views[1], None, views[2]), model(x, past, None, future), rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
def test_singleton_node_time_batch_and_hidden_channel_are_defined(training):
    model = (
        api()
        .GMAN(
            **config(
                enc_in=1,
                dec_in=1,
                c_out=1,
                seq_len=1,
                label_len=0,
                pred_len=1,
                d_model=1,
                n_heads=1,
                time_slots_per_day=1,
                spatial_embedding=[[0.0]],
            )
        )
        .train(training)
    )
    x = torch.ones(1, 1, 1, requires_grad=True)
    marks = torch.tensor([[[6, 0]]])
    result = model.predict(model.make_forecast_batch(x, x_mark_enc=marks, x_mark_dec=marks))
    assert result.shape == (1, 1, 1) and torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_metr_like_depth_has_finite_cold_eval_and_explicit_mask_dropout_choices():
    torch.manual_seed(17)
    model = api().GMAN(**config(e_layers=5, temporal_mask=True, dropout=0.1)).eval()
    x, past, future = inputs(model)
    result = model(x, past, None, future)
    assert result.shape == (2, 2, 3) and torch.isfinite(result).all()
    torch.testing.assert_close(result, model(x, past, None, future), rtol=0, atol=0)


def test_owned_spatial_embedding_and_json_config_do_not_alias_constructor_or_caller():
    spatial = torch.tensor(config()["spatial_embedding"], requires_grad=True)
    model = api().GMAN(**config(spatial_embedding=spatial))
    stored = model.spatial_embedding.clone()
    with torch.no_grad():
        spatial.add_(100)
    torch.testing.assert_close(model.spatial_embedding, stored, rtol=0, atol=0)
    cfg = json.loads(json.dumps(model.config()))
    cfg["spatial_embedding"][0][0] += 200
    torch.testing.assert_close(model.spatial_embedding, stored, rtol=0, atol=0)
    with torch.no_grad():
        model.spatial_embedding.add_(0.125)
    assert model.config()["spatial_embedding"] == model.spatial_embedding.tolist()
    assert not model.spatial_embedding.requires_grad
    assert "spatial_embedding" not in dict(model.named_parameters())


@pytest.mark.parametrize("integer_type", [np.int8, np.uint8, np.int64])
def test_fixed_width_dimensions_are_canonical_before_arithmetic(integer_type):
    model = (
        api()
        .GMAN(
            **config(
                d_model=integer_type(64),
                n_heads=integer_type(4),
                time_slots_per_day=np.uint8(250),
                spatial_embedding=np.zeros((3, 64)),
                e_layers=integer_type(1),
                bn_decay=np.float32(0.9),
                dropout=np.float32(0),
            )
        )
        .eval()
    )
    x, past, future = inputs(model)
    assert model(x, past, None, future).shape == (2, 2, 3)
    assert json.loads(json.dumps(model.config()))["d_model"] == 64


def test_fixed_width_label_and_horizon_sum_does_not_wrap():
    model = api().GMAN(**config(seq_len=np.int8(120), label_len=np.int8(100), pred_len=np.int8(30))).eval()
    x, past, future = inputs(model, batch=1)
    assert future.size(1) == 130
    assert model.predict(model.make_forecast_batch(x, x_mark_enc=past, x_mark_dec=future)).shape == (1, 30, 3)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_calendar_integer_types_validate_after_safe_promotion(dtype):
    model = api().GMAN(**config(time_slots_per_day=288)).eval()
    x, past, future = inputs(model)
    past[..., 1] = 100
    future[..., 1] = 100
    expected = model(x, past, None, future)
    torch.testing.assert_close(model(x, past.to(dtype), None, future.to(dtype)), expected, rtol=0, atol=0)


def test_autocast_does_not_silently_change_portable_float_precision():
    model = api().GMAN(**config()).eval()
    x, past, future = inputs(model)
    expected = model(x, past, None, future)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(x, past, None, future)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"enc_in": 0}, "enc_in"),
        ({"dec_in": 0}, "dec_in"),
        ({"c_out": 0}, "c_out"),
        ({"seq_len": 0}, "seq_len"),
        ({"pred_len": 0}, "pred_len"),
        ({"d_model": 0}, "d_model"),
        ({"n_heads": 0}, "n_heads"),
        ({"e_layers": 0}, "e_layers"),
        ({"time_slots_per_day": 0}, "time_slots"),
        ({"seq_len": True}, "seq_len"),
        ({"d_model": 4.5}, "d_model"),
        ({"e_layers": np.bool_(True)}, "e_layers"),
        ({"label_len": -1}, "label_len"),
        ({"label_len": 5}, "label_len"),
        ({"label_len": True}, "label_len"),
        ({"n_heads": 3}, "divisible"),
        ({"c_out": 2}, "c_out"),
        ({"temporal_mask": 1}, "temporal_mask"),
        ({"use_batch_norm": 0}, "use_batch_norm"),
        ({"activation": "gelu"}, "relu"),
        ({"output_attention": True}, "output_attention"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": float("nan")}, "dropout"),
        ({"dropout": float("inf")}, "dropout"),
        ({"dropout": True}, "dropout"),
        ({"bn_decay": -0.1}, "bn_decay"),
        ({"bn_decay": 1.0}, "bn_decay"),
        ({"bn_decay": float("nan")}, "bn_decay"),
        ({"bn_decay": True}, "bn_decay"),
    ],
)
def test_invalid_configuration_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        api().GMAN(**config(**overrides))


@pytest.mark.parametrize(
    "spatial",
    [
        None,
        [[1.0]],
        [[1.0], [2.0, 3.0]],
        "bad",
        torch.ones(3, 4, dtype=torch.bool),
        torch.ones(3, 4, dtype=torch.complex64),
        torch.eye(3).to_sparse(),
        torch.full((3, 4), torch.nan),
        torch.full((3, 4), torch.inf),
        [[1e50] * 4] * 3,
        [[-1e-100] * 4] * 3,
    ],
)
def test_invalid_or_unrepresentable_spatial_embeddings_are_rejected(spatial):
    with pytest.raises(ValueError, match="spatial_embedding"):
        api().GMAN(**config(spatial_embedding=spatial))


@pytest.mark.parametrize("which", ["history", "future"])
@pytest.mark.parametrize(
    "kind", ["missing", "float", "bool", "shape", "batch", "weekday", "slot", "negative", "device"]
)
def test_invalid_calendar_fails_before_any_bn_state_change(which, kind):
    model = api().GMAN(**config())
    x, past, future = inputs(model)
    mark = past if which == "history" else future
    index = 0 if which == "history" else model.label_len
    if kind == "missing":
        mark = None
    elif kind == "float":
        mark = mark.float()
    elif kind == "bool":
        mark = mark.bool()
    elif kind == "shape":
        mark = mark.repeat(1, 1, 2)
    elif kind == "batch":
        mark = mark[:1]
    elif kind == "weekday":
        mark[:, index, 0] = 7
    elif kind == "slot":
        mark[:, index, 1] = 8
    elif kind == "negative":
        mark[:, index, 1] = -1
    else:
        mark = mark.to(device="meta")
    if which == "history":
        past = mark
    else:
        future = mark
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises((TypeError, ValueError), match="x_mark|calendar"):
        model(x, past, None, future)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind",
    ["object", "rank", "length", "nodes", "empty", "integer", "half", "complex", "double", "nan", "inf", "device"],
)
def test_invalid_observations_are_rejected(kind):
    model = api().GMAN(**config())
    x, past, future = inputs(model)
    if kind == "object":
        x = x.tolist()
    elif kind == "rank":
        x = x.unsqueeze(-1)
    elif kind == "length":
        x = x[:, :3]
    elif kind == "nodes":
        x = x[:, :, :2]
    elif kind == "empty":
        x = x[:0]
    elif kind in ("integer", "half", "complex", "double"):
        x = x.to(
            {"integer": torch.long, "half": torch.float16, "complex": torch.complex64, "double": torch.float64}[kind]
        )
    elif kind in ("nan", "inf"):
        x[0, 0, 0] = float(kind)
    else:
        x = x.to(device="meta")
    with pytest.raises((TypeError, ValueError), match="x_enc|input|observations|batch|device|dtype"):
        model(x, past, None, future)


@pytest.mark.parametrize("name", ["enc_self_mask", "dec_self_mask", "dec_enc_mask"])
def test_external_masks_are_not_silently_ignored(name):
    model = api().GMAN(**config())
    x, past, future = inputs(model)
    with pytest.raises(ValueError, match="mask"):
        model(x, past, None, future, **{name: torch.ones(1, 1)})


@pytest.mark.parametrize("which", ["history", "future", "both"])
def test_make_forecast_batch_requires_explicit_calendar_instead_of_zero_defaults(which):
    model = api().GMAN(**config())
    x, past, future = inputs(model)
    with pytest.raises(ValueError, match="calendar|x_mark"):
        model.make_forecast_batch(
            x, x_mark_enc=None if which != "future" else past, x_mark_dec=None if which != "history" else future
        )


def test_config_dataclass_constructs_a_working_model_with_same_defaults():
    cls = api()
    model = cls.GMAN.from_config(cls.GMANConfig(**config())).eval()
    x, past, future = inputs(model)
    batch = model.make_forecast_batch(x, x_mark_enc=past, x_mark_dec=future, x_exog_enc=torch.ones(2, 4, 1))
    assert model.predict(batch).shape == (2, 2, 3)
    assert batch.x_exog_enc.shape == (2, 4, 1)
    assert model.config()["dropout"] == 0 and model.config()["temporal_mask"] is False


@pytest.mark.parametrize("as_dataclass", [False, True])
def test_computed_tensor_embedding_config_constructs_detached_owned_prediction(as_dataclass):
    cls = api()
    leaf = torch.ones(3, 4, requires_grad=True)
    computed = leaf * 0.5
    cfg = config(spatial_embedding=computed)
    model = cls.GMAN.from_config(cls.GMANConfig(**cfg) if as_dataclass else cfg).eval()
    expected_embedding = model.spatial_embedding.clone()
    assert not model.spatial_embedding.requires_grad
    with torch.no_grad():
        computed.add_(10)
        leaf.add_(20)
    torch.testing.assert_close(model.spatial_embedding, expected_embedding, rtol=0, atol=0)
    x, past, future = inputs(model)
    prediction = model(x, past, None, future)
    assert prediction.shape == (2, 2, 3) and torch.isfinite(prediction).all()
    prediction.sum().backward()
    assert leaf.grad is None


@pytest.mark.parametrize(
    "dtype_arg,wanted",
    [
        ("float32", torch.float32),
        ("float64", torch.float64),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_explicit_construction_dtype_is_owned_without_changing_process_default(dtype_arg, wanted):
    original = torch.get_default_dtype()
    model = api().GMAN(**config(dtype=dtype_arg)).eval()
    assert torch.get_default_dtype() == original
    assert model.spatial_embedding.dtype == wanted
    assert all(p.dtype == wanted for p in model.parameters())
    assert model.config()["dtype"] == str(wanted).removeprefix("torch.")
    assert json.loads(json.dumps(model.config())) == model.config()
    x, past, future = inputs(model, dtype=wanted)
    assert model(x, past, None, future).dtype == wanted
    other_dtype = torch.float64 if wanted == torch.float32 else torch.float32
    model.to(dtype=other_dtype)
    assert model.config()["dtype"] == str(other_dtype).removeprefix("torch.")


@pytest.mark.parametrize("dtype_arg", ["float16", torch.bfloat16, True])
def test_unsupported_construction_dtype_is_rejected(dtype_arg):
    with pytest.raises(ValueError, match="dtype"):
        api().GMAN(**config(dtype=dtype_arg))


def test_unsupported_default_dtype_is_rejected_without_changing_it():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with pytest.raises(ValueError, match="dtype"):
            api().GMAN(**config())
        assert torch.get_default_dtype() == torch.float16
    finally:
        torch.set_default_dtype(original)
