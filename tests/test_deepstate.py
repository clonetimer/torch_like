"""DeepState public contracts, with independent conditional Gaussian oracles."""

import importlib
import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_deepstate_layers import dense_joint_oracle


def api():
    name = "torch_like.models.forecasting.deepstate"
    assert importlib.util.find_spec(name) is not None, "native DeepState model is missing"
    return importlib.import_module(name)


def config(**overrides):
    values = dict(
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=4,
        label_len=1,
        pred_len=3,
        d_model=4,
        e_layers=2,
        time_feature_dim=0,
        scaling=False,
    )
    return values | overrides


def batch_for(model, batch_size=2):
    dtype = next(model.parameters()).dtype
    b, t, h, prefix = batch_size, model.seq_len, model.pred_len, model.label_len
    d = model.time_feature_dim
    k = len(model.seasonal_periods)
    past = torch.randn(b, t, d, dtype=dtype) if d else None
    future = torch.randn(b, prefix + h, d, dtype=dtype) if d else None

    def indices(length):
        if not k:
            return None
        return (
            torch.stack([torch.arange(length).remainder(p) for p in model.seasonal_periods], -1)[None]
            .expand(b, -1, -1)
            .clone()
        )

    cats = torch.zeros(b, len(model.cardinalities), dtype=torch.long) if model.cardinalities else None
    return api().DeepStateBatch(
        x_enc=torch.randn(b, t, 1, dtype=dtype),
        x_mark_enc=past,
        x_dec=None,
        x_mark_dec=future,
        y_true=torch.randn(b, h, 1, dtype=dtype),
        seasonal_indices_enc=indices(t),
        seasonal_indices_dec=indices(prefix + h),
        feat_static_cat=cats,
    )


def zero_parameters(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"time_feature_dim": 2},
        {"time_feature_dim": 2, "seasonal_periods": (2, 3), "cardinalities": (3, 2), "embedding_dimensions": (2, 1)},
    ],
)
def test_public_prediction_distribution_and_optimizer(dtype, extra):
    torch.manual_seed(71)
    model = api().DeepState(**config(**extra), dtype=dtype)
    batch = batch_for(model)
    distribution = model.predict_distribution(batch)
    assert distribution["distribution"] == "gaussian"
    assert distribution["mean"].shape == distribution["scale"].shape == (2, 3, 1)
    assert distribution["mean"].dtype == distribution["scale"].dtype == dtype
    assert torch.isfinite(distribution["mean"]).all() and (distribution["scale"] > 0).all()
    torch.testing.assert_close(model.predict(batch), distribution["mean"], rtol=0, atol=0)
    loss = model.fit_step(batch) + model.context_nll(batch)
    assert loss.ndim == 0 and loss.dtype == dtype and torch.isfinite(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    before = model.predict(batch).detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.predict(batch))
    assert model.sample(batch, num_samples=5).shape == (5, 2, 3, 1)


def constant_joint(model):
    """Zero neural weights imply literal midpoint bounds, independently of helpers."""
    t = model.seq_len + model.pred_len
    return dense_joint_oracle(
        mean=torch.zeros(1, 2, dtype=torch.float64),
        cov=torch.eye(2, dtype=torch.float64)[None] * 0.5000005,
        emission=torch.ones(1, t, 2, dtype=torch.float64),
        transition=torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        innovation=torch.full((1, t, 1), 0.0050005, dtype=torch.float64),
        noise_std=torch.full((1, t, 1), 0.5000005, dtype=torch.float64),
        residual=torch.zeros(1, t, 1, dtype=torch.float64),
    )


def test_conditional_future_nll_and_boundary_match_dense_joint_gaussian():
    model = zero_parameters(api().DeepState(**config(dtype="float64")))
    batch = batch_for(model, 1)
    mean, cov = constant_joint(model)
    y = torch.cat((batch.x_enc, batch.y_true), 1)[..., 0]
    full_lp = torch.distributions.MultivariateNormal(mean, cov).log_prob(y)
    context_lp = torch.distributions.MultivariateNormal(mean[:, :4], cov[:, :4, :4]).log_prob(y[:, :4])
    torch.testing.assert_close(model.fit_step(batch), (context_lp - full_lp).sum() / 3, rtol=1e-10, atol=1e-10)
    cross = cov[:, 4:, :4]
    conditional_mean = mean[:, 4:] + (
        cross @ torch.linalg.solve(cov[:, :4, :4], y[:, :4, None] - mean[:, :4, None])
    ).squeeze(-1)
    conditional_cov = cov[:, 4:, 4:] - cross @ torch.linalg.solve(cov[:, :4, :4], cross.transpose(-1, -2))
    dist = model.predict_distribution(batch)
    torch.testing.assert_close(dist["mean"][..., 0], conditional_mean, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        dist["scale"][..., 0].square(), conditional_cov.diagonal(dim1=-2, dim2=-1), rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(model.context_nll(batch), -context_lp.sum() / 4, rtol=1e-10, atol=1e-10)


def test_missing_observations_use_joint_observed_subset_and_global_future_count():
    model = zero_parameters(api().DeepState(**config(dtype="float64")))
    batch = batch_for(model, 1)
    past_mask = torch.tensor([[[True], [False], [True], [True]]])
    future_mask = torch.tensor([[[False], [True], [True]]])
    batch = replace(batch, observed_enc=past_mask, observed_dec=future_mask)
    mean, cov = constant_joint(model)
    y = torch.cat((batch.x_enc, batch.y_true), 1)[..., 0]

    def density(ids):
        return torch.distributions.MultivariateNormal(mean[:, ids], cov[:, ids][:, :, ids]).log_prob(y[:, ids])

    expected = (density([0, 2, 3]) - density([0, 2, 3, 5, 6])).sum() / 2
    torch.testing.assert_close(model.fit_step(batch), expected, rtol=1e-10, atol=1e-10)
    doubled = replace(
        batch,
        **{
            name: torch.cat((getattr(batch, name), getattr(batch, name)), 0)
            for name in ("x_enc", "y_true", "observed_enc", "observed_dec")
        },
    )
    doubled.observed_dec[1] = False
    doubled.y_true[1] = float("nan")
    torch.testing.assert_close(model.fit_step(doubled), expected, rtol=1e-10, atol=1e-10)


def test_public_samples_preserve_temporal_covariance():
    torch.manual_seed(902)
    model = zero_parameters(api().DeepState(**config(dtype="float64")))
    batch = batch_for(model, 1)
    mean, cov = constant_joint(model)
    cross = cov[:, 4:, :4]
    expected_cov = cov[:, 4:, 4:] - cross @ torch.linalg.solve(cov[:, :4, :4], cross.transpose(-1, -2))
    samples = model.sample(batch, num_samples=50000)[:, 0, :, 0]
    torch.testing.assert_close(samples.mean(0), model.predict(batch)[0, :, 0], rtol=0, atol=0.025)
    torch.testing.assert_close(torch.cov(samples.T), expected_cov[0], rtol=0.04, atol=0.025)


def test_covariate_network_ignores_target_values_and_scale():
    model = api().DeepState(**config(time_feature_dim=2, scaling=True))
    batch = batch_for(model)
    captured = []
    handle = model.rnn.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    model.predict(batch)
    model.predict(replace(batch, x_enc=batch.x_enc * 70, y_true=batch.y_true * 99))
    handle.remove()
    assert len(captured) == 4
    torch.testing.assert_close(captured[0], captured[2], rtol=0, atol=0)
    torch.testing.assert_close(captured[1], captured[3], rtol=0, atol=0)


def test_decoder_targets_and_prefixes_are_ignored():
    model = api().DeepState(**config(time_feature_dim=2, seasonal_periods=(3,)))
    batch = batch_for(model)
    expected = model.predict_distribution(batch)
    changed = replace(
        batch,
        x_dec=torch.full((2, 4, 1), float("nan")),
        y_true=torch.full((2, 3, 1), float("inf")),
        x_mark_dec=batch.x_mark_dec.clone(),
        seasonal_indices_dec=batch.seasonal_indices_dec.clone(),
    )
    changed.x_mark_dec[:, :1] = float("nan")
    changed.seasonal_indices_dec[:, :1] = -99
    for name in ("mean", "scale"):
        torch.testing.assert_close(model.predict_distribution(changed)[name], expected[name], rtol=0, atol=0)


def test_scaling_uses_observed_context_only_and_has_correct_jacobian():
    model = api().DeepState(**config(scaling=True, dtype="float64"))
    batch = batch_for(model)
    batch.x_enc = torch.tensor([[[1.0], [2.0], [3.0], [4.0]], [[2.0], [3.0], [4.0], [5.0]]], dtype=torch.float64)
    factor = 7.0
    changed = replace(batch, x_enc=batch.x_enc * factor, y_true=batch.y_true * factor)
    for name in ("mean", "scale"):
        torch.testing.assert_close(
            model.predict_distribution(changed)[name],
            model.predict_distribution(batch)[name] * factor,
            rtol=1e-11,
            atol=1e-11,
        )
    torch.testing.assert_close(
        model.fit_step(changed) - model.fit_step(batch), torch.tensor(np.log(factor)), rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        model.context_nll(changed) - model.context_nll(batch), torch.tensor(np.log(factor)), rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize("missing_value", [float("nan"), float("inf"), -1e30])
def test_masked_values_have_no_value_or_gradient_effect(missing_value):
    model = api().DeepState(**config(scaling=True, dtype="float64"))
    batch = batch_for(model)
    batch.observed_enc = torch.ones_like(batch.x_enc, dtype=torch.bool)
    batch.observed_dec = torch.ones_like(batch.y_true, dtype=torch.bool)
    batch.observed_enc[:, 1] = False
    batch.observed_dec[:, 1] = False
    clean = replace(batch, x_enc=batch.x_enc.clone(), y_true=batch.y_true.clone())
    batch.x_enc[:, 1] = missing_value
    batch.y_true[:, 1] = missing_value
    batch.x_enc.requires_grad_()
    batch.y_true.requires_grad_()
    loss = model.fit_step(batch) + model.context_nll(batch)
    torch.testing.assert_close(loss, model.fit_step(clean) + model.context_nll(clean), rtol=0, atol=0)
    loss.backward()
    for value in (batch.x_enc, batch.y_true):
        assert torch.isfinite(value.grad).all() and (value.grad[:, 1] == 0).all()
    assert batch.x_enc.grad.abs().sum() > 0


def test_all_missing_context_is_legal_and_row_scaling_is_independent():
    model = api().DeepState(**config(scaling=True))
    batch = batch_for(model)
    batch.x_enc[0] = float("nan")
    batch.observed_enc = torch.isfinite(batch.x_enc)
    before = model.predict_distribution(batch)
    assert torch.isfinite(before["mean"]).all()
    changed = replace(batch, x_enc=batch.x_enc.clone())
    changed.x_enc[1] *= 100
    torch.testing.assert_close(model.predict(changed)[0], before["mean"][0], rtol=0, atol=0)
    missing = replace(batch, observed_enc=torch.zeros_like(batch.x_enc, dtype=torch.bool))
    with pytest.raises(ValueError, match="observed"):
        model.context_nll(missing)
    with pytest.raises(ValueError, match="observed"):
        model.fit_step(replace(batch, observed_dec=torch.zeros_like(batch.y_true, dtype=torch.bool)))


def test_zero_only_context_scale_is_one_and_context_requires_no_future():
    model = api().DeepState(**config(scaling=True, time_feature_dim=2, seasonal_periods=(3,)))
    batch = batch_for(model)
    batch.x_enc.zero_()
    nofuture = replace(batch, x_mark_dec=None, seasonal_indices_dec=None, y_true=None)
    assert torch.isfinite(model.context_nll(nofuture))
    assert torch.isfinite(model.predict(batch)).all()


@pytest.mark.parametrize(
    "overrides",
    [
        {"enc_in": 2},
        {"dec_in": 2},
        {"c_out": 2},
        {"seq_len": 0},
        {"pred_len": True},
        {"label_len": -1},
        {"label_len": 5},
        {"d_model": 1.2},
        {"e_layers": 0},
        {"num_samples": 0},
        {"distribution": "student_t"},
        {"dropout": 0.1},
        {"dropout": False},
        {"time_feature_dim": -1},
        {"time_feature_dim": False},
        {"seasonal_periods": [1]},
        {"seasonal_periods": [True]},
        {"seasonal_periods": "24"},
        {"cardinalities": [3], "embedding_dimensions": []},
        {"cardinalities": [0], "embedding_dimensions": [2]},
        {"cardinalities": [3], "embedding_dimensions": [False]},
        {"add_trend": 1},
        {"scaling": 0},
        {"minimum_scale": 0},
        {"minimum_scale": float("nan")},
        {"dtype": "float16"},
        {"dtype": torch.bfloat16},
        {"noise_std_bounds": (0.0, 1.0)},
        {"noise_std_bounds": (1.0, 0.1)},
        {"noise_std_bounds": (1e-30, 1.0)},
        {"innovation_bounds": (1e-6, float("inf"))},
        {"prior_cov_bounds": (1e-50, 1.0)},
        {"noise_std_bounds": (1e-6, 1e30)},
        {"prior_cov_bounds": (True, 1.0)},
    ],
)
def test_invalid_configuration_fails_before_execution(overrides):
    with pytest.raises((ValueError, TypeError)):
        api().DeepState(**config(**overrides))


def test_mutable_configuration_is_owned_and_json_roundtrip_uses_current_dtype():
    periods = [3]
    cards = [2]
    dims = [2]
    bounds = [1e-6, 1.0]
    model = api().DeepState.from_config(
        api().DeepStateConfig(
            **config(seasonal_periods=periods, cardinalities=cards, embedding_dimensions=dims, noise_std_bounds=bounds)
        )
    )
    periods[0] = 99
    cards[0] = 99
    dims[0] = 99
    bounds[0] = 0
    cfg = model.double().config()
    assert cfg["seasonal_periods"] == [3] and cfg["cardinalities"] == [2] and cfg["noise_std_bounds"] == [1e-6, 1.0]
    assert cfg["dtype"] == "float64" and cfg == json.loads(json.dumps(cfg))
    restored = api().DeepState.from_config(cfg)
    restored.load_state_dict(model.state_dict())
    batch = batch_for(model)
    torch.testing.assert_close(model.predict(batch), restored.predict(batch), rtol=0, atol=0)
    cfg["seasonal_periods"][0] = 30
    assert model.config()["seasonal_periods"] == [3]


@pytest.mark.parametrize(
    "field", ["x_enc", "x_mark_enc", "x_mark_dec", "seasonal_indices_enc", "seasonal_indices_dec", "feat_static_cat"]
)
def test_required_inputs_are_not_synthesized(field):
    model = api().DeepState(
        **config(time_feature_dim=2, seasonal_periods=(3,), cardinalities=(2,), embedding_dimensions=(2,))
    )
    batch = replace(batch_for(model), **{field: None})
    with pytest.raises((ValueError, TypeError), match=field):
        model.predict(batch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("x_enc", torch.zeros(0, 4, 1)),
        ("x_enc", torch.zeros(2, 3, 1)),
        ("x_enc", torch.ones(2, 4, 1, dtype=torch.long)),
        ("x_enc", torch.zeros(2, 4, 1, dtype=torch.float64)),
        ("x_enc", torch.full((2, 4, 1), float("nan"))),
        ("x_enc", torch.zeros(2, 4, 1).to_sparse()),
        ("x_dec", torch.zeros(2, 3, 1)),
        ("observed_enc", torch.ones(2, 4, 1)),
        ("observed_enc", torch.ones(2, 4, dtype=torch.bool)),
        ("x_mark_enc", torch.full((2, 4, 2), float("nan"))),
        ("x_mark_dec", torch.zeros(2, 3, 2)),
        ("seasonal_indices_enc", torch.zeros(2, 4, 1)),
        ("seasonal_indices_enc", torch.full((2, 4, 1), 3)),
        ("seasonal_indices_dec", torch.full((2, 4, 1), -1)),
        ("feat_static_cat", torch.full((2, 1), 2)),
        ("feat_static_cat", torch.zeros(2, 1, dtype=torch.bool)),
        ("enc_self_mask", torch.ones(1)),
        ("x_exog_enc", torch.ones(2, 4, 1)),
        ("x_target_dec", torch.ones(2, 4, 1)),
    ],
)
def test_bad_batch_fails_explicitly_without_mutating_state(field, value):
    model = api().DeepState(
        **config(time_feature_dim=2, seasonal_periods=(3,), cardinalities=(2,), embedding_dimensions=(2,))
    )
    batch = replace(batch_for(model), **{field: value})
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises((ValueError, TypeError)):
        model.predict(batch)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("num_samples", [0, -1, True, 1.5])
def test_invalid_sample_count_is_rejected(num_samples):
    model = api().DeepState(**config())
    with pytest.raises(ValueError, match="num_samples"):
        model.sample(batch_for(model), num_samples=num_samples)


def test_batch_helper_preserves_supplied_extras_without_creating_calendar():
    model = api().DeepState(**config(time_feature_dim=2, seasonal_periods=(3,)))
    supplied = batch_for(model)
    batch = model.make_probabilistic_forecast_batch(**supplied.__dict__)
    assert type(batch) is api().DeepStateBatch
    assert batch.seasonal_indices_enc is supplied.seasonal_indices_enc
    assert batch.x_dec is None
    missing = model.make_probabilistic_forecast_batch(supplied.x_enc)
    assert missing.x_mark_enc is None and missing.seasonal_indices_enc is None
    with pytest.raises(ValueError):
        model.predict(missing)


def test_amp_does_not_change_numerical_dtype_or_prediction():
    model = api().DeepState(**config())
    batch = batch_for(model)
    expected = model.predict_distribution(batch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.predict_distribution(batch)
        loss = model.fit_step(batch)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    for name in ("mean", "scale"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_distribution_rejects_scale_underflow_in_output_dtype(dtype):
    model = api().DeepState(
        **config(
            seq_len=2,
            label_len=0,
            pred_len=2,
            d_model=2,
            e_layers=1,
            add_trend=False,
            scaling=True,
            minimum_scale=1e-40,
            noise_std_bounds=(1e-6, 1e-6),
            innovation_bounds=(1e-6, 1e-6),
            prior_cov_bounds=(1e-6, 1e-6),
            dtype=dtype,
        )
    )
    batch = api().DeepStateBatch(x_enc=torch.full((1, 2, 1), 1e-40, dtype=dtype))
    if dtype == torch.float32:
        with pytest.raises(ValueError, match="scale"):
            model.predict_distribution(batch)
    else:
        distribution = model.predict_distribution(batch)
        assert torch.isfinite(distribution["scale"]).all() and (distribution["scale"] > 0).all()
        assert torch.isfinite(torch.distributions.Normal(distribution["mean"], distribution["scale"]).entropy()).all()
