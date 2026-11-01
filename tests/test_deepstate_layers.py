"""Independent Gaussian/state-space oracles for the native DeepState core."""

import importlib
import importlib.util
import math

import pytest
import torch


def get_layers():
    name = "torch_like.models.forecasting.deepstate_layers"
    assert importlib.util.find_spec(name) is not None, "native DeepState numerical core is missing"
    return importlib.import_module(name)


def tiny_system():
    dtype = torch.float64
    return dict(
        mean=torch.tensor([[0.3, -0.2]], dtype=dtype),
        cov=torch.tensor([[[1.2, 0.2], [0.2, 0.7]]], dtype=dtype),
        emission=torch.tensor([[[1.0, 0.5], [0.2, 1.0], [1.0, -0.3]]], dtype=dtype),
        transition=torch.tensor([[1.0, 0.4], [0.0, 0.8]], dtype=dtype),
        innovation=torch.tensor([[[0.2], [0.3], [0.1]]], dtype=dtype),
        noise_std=torch.tensor([[[0.4], [0.6], [0.3]]], dtype=dtype),
        residual=torch.tensor([[[0.1], [-0.2], [0.05]]], dtype=dtype),
    )


def dense_joint_oracle(mean, cov, emission, transition, innovation, noise_std, residual):
    """Build each output directly as an affine map of independent normal noises.

    Unlike the production recurrence this never filters or propagates covariance.
    Observation/state noise occupy separate columns; one state-noise column is
    shared by every latent component at each time.
    """
    batch, horizon, state_dim = emission.shape
    coefficients = mean.new_zeros(batch, state_dim, state_dim + 2 * horizon)
    coefficients[:, :, :state_dim] = torch.linalg.cholesky(cov)
    state_mean = mean
    means, rows = [], []
    for t in range(horizon):
        a = emission[:, t]
        row = (a.unsqueeze(1) @ coefficients).squeeze(1)
        row[:, state_dim + 2 * t] += noise_std[:, t, 0]
        means.append((a * state_mean).sum(-1) + residual[:, t, 0])
        rows.append(row)
        coefficients = transition @ coefficients
        coefficients[:, :, state_dim + 2 * t + 1] += a * innovation[:, t]
        state_mean = (transition @ state_mean.unsqueeze(-1)).squeeze(-1)
    matrix = torch.stack(rows, dim=1)
    return torch.stack(means, dim=1), matrix @ matrix.transpose(-1, -2)


@pytest.mark.parametrize("trend", [False, True])
def test_structural_coefficients_use_explicit_calendar_and_reference_trend(trend):
    layers = get_layers()
    structure = layers._ISSM(periods=(3,), add_trend=trend, dtype=torch.float64)
    ids = torch.tensor([[[0], [2]]])
    reference = torch.zeros(1, 2, 1, dtype=torch.float64)
    actual = structure.emission(ids, reference)
    if trend:
        expected = torch.tensor([[[1, 1, 1, 0, 0], [1, 1, 0, 0, 1]]], dtype=torch.float64)
        transition = torch.tensor(
            [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
            dtype=torch.float64,
        )
    else:
        expected = torch.tensor([[[1, 1, 0, 0], [1, 0, 0, 1]]], dtype=torch.float64)
        transition = torch.eye(4, dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(structure.transition, transition, rtol=0, atol=0)
    assert structure.state_dim == expected.shape[-1]
    assert "transition" in structure.state_dict()


def test_no_seasonal_block_needs_no_calendar():
    layers = get_layers()
    structure = layers._ISSM(periods=(), add_trend=False, dtype=torch.float64)
    actual = structure.emission(None, torch.zeros(2, 3, 1, dtype=torch.float64))
    torch.testing.assert_close(actual, torch.ones(2, 3, 1, dtype=torch.float64))


def test_scalar_kalman_update_then_transition_has_literal_solution():
    layers = get_layers()
    target = torch.tensor([[[2.0]]], dtype=torch.float64)
    lp, mean, cov = layers._kalman_filter(
        target,
        torch.ones_like(target, dtype=torch.bool),
        mean=torch.zeros(1, 1, dtype=torch.float64),
        cov=torch.tensor([[[4.0]]], dtype=torch.float64),
        emission=torch.ones_like(target),
        transition=torch.ones(1, 1, dtype=torch.float64),
        innovation=torch.full_like(target, 0.3),
        noise_std=torch.ones_like(target),
        residual=torch.zeros_like(target),
    )
    torch.testing.assert_close(mean, torch.tensor([[1.6]], dtype=torch.float64))
    torch.testing.assert_close(cov, torch.tensor([[[0.89]]], dtype=torch.float64))
    torch.testing.assert_close(lp, torch.tensor([[-0.5 * (math.log(10 * math.pi) + 0.8)]], dtype=torch.float64))


@pytest.mark.parametrize("missing_value", [float("nan"), float("inf"), -1e200])
def test_missing_middle_value_skips_update_but_still_transitions(missing_value):
    layers = get_layers()
    target = torch.tensor([[[2.0], [missing_value], [1.0]]], dtype=torch.float64)
    lp, mean, cov = layers._kalman_filter(
        target,
        torch.tensor([[[True], [False], [True]]]),
        mean=torch.zeros(1, 1, dtype=torch.float64),
        cov=torch.ones(1, 1, 1, dtype=torch.float64),
        emission=torch.ones_like(target),
        transition=torch.full((1, 1), 2.0, dtype=torch.float64),
        innovation=torch.full_like(target, 0.5),
        noise_std=torch.ones_like(target),
        residual=torch.zeros_like(target),
    )
    torch.testing.assert_close(mean, torch.tensor([[106 / 41]], dtype=torch.float64))
    torch.testing.assert_close(cov, torch.tensor([[[633 / 164]]], dtype=torch.float64))
    expected = torch.tensor(
        [[-0.5 * (math.log(4 * math.pi) + 2), 0.0, -0.5 * (math.log(20.5 * math.pi) + 9 / 10.25)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(lp, expected)


def test_composite_innovation_is_rank_one_across_structural_blocks():
    layers = get_layers()
    structure = layers._ISSM(periods=(2,), add_trend=True, dtype=torch.float64)
    target = torch.zeros(1, 1, 1, dtype=torch.float64)
    emission = structure.emission(torch.tensor([[[0]]]), target)
    _, _, cov = layers._kalman_filter(
        target,
        torch.zeros_like(target, dtype=torch.bool),
        mean=torch.zeros(1, 4, dtype=torch.float64),
        cov=torch.zeros(1, 4, 4, dtype=torch.float64),
        emission=emission,
        transition=structure.transition,
        innovation=torch.full_like(target, 0.3),
        noise_std=torch.ones_like(target),
        residual=target,
    )
    expected = torch.tensor([[[0.09, 0.09, 0.09, 0], [0.09, 0.09, 0.09, 0], [0.09, 0.09, 0.09, 0], [0, 0, 0, 0]]])
    torch.testing.assert_close(cov, expected.double())


def test_sequential_likelihood_matches_independent_dense_joint_gaussian():
    layers = get_layers()
    system = tiny_system()
    target = torch.tensor([[[0.5], [1.2], [-0.1]]], dtype=torch.float64)
    expected_mean, expected_cov = dense_joint_oracle(**system)
    expected = torch.distributions.MultivariateNormal(expected_mean, expected_cov).log_prob(target.squeeze(-1))
    lp, _, _ = layers._kalman_filter(target, torch.ones_like(target, dtype=torch.bool), **system)
    torch.testing.assert_close(lp.sum(1), expected, rtol=1e-12, atol=1e-12)


def test_missing_likelihood_is_joint_density_of_observed_subset():
    layers = get_layers()
    system = tiny_system()
    target = torch.tensor([[[0.5], [float("nan")], [-0.1]]], dtype=torch.float64)
    expected_mean, expected_cov = dense_joint_oracle(**system)
    keep = torch.tensor([0, 2])
    expected = torch.distributions.MultivariateNormal(
        expected_mean[:, keep], expected_cov[:, keep][:, :, keep]
    ).log_prob(target[:, keep, 0])
    lp, _, _ = layers._kalman_filter(target, torch.isfinite(target), **system)
    torch.testing.assert_close(lp.sum(1), expected, rtol=1e-12, atol=1e-12)


def test_marginals_emit_before_nonidentity_transition():
    layers = get_layers()
    system = tiny_system()
    expected_mean, expected_cov = dense_joint_oracle(**system)
    actual_mean, actual_var = layers._marginals(**system)
    torch.testing.assert_close(actual_mean[..., 0], expected_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_var[..., 0], expected_cov.diagonal(dim1=-2, dim2=-1), rtol=1e-12, atol=1e-12)


def test_filter_likelihood_has_correct_float64_gradients():
    layers = get_layers()
    system = tiny_system()
    target = torch.tensor([[[0.5], [1.2], [-0.1]]], dtype=torch.float64, requires_grad=True)
    mean = system["mean"].clone().requires_grad_()
    factor = torch.linalg.cholesky(system["cov"]).detach().requires_grad_()
    noise = system["noise_std"].clone().requires_grad_()
    innovation = system["innovation"].clone().requires_grad_()
    residual = system["residual"].clone().requires_grad_()

    def objective(m, f, sigma, gamma, b, y):
        lp, _, _ = layers._kalman_filter(
            y,
            torch.ones_like(y, dtype=torch.bool),
            mean=m,
            cov=f @ f.transpose(-1, -2),
            emission=system["emission"],
            transition=system["transition"],
            innovation=gamma,
            noise_std=sigma,
            residual=b,
        )
        return lp.sum()

    assert torch.autograd.gradcheck(objective, (mean, factor, noise, innovation, residual, target))


def test_masked_nan_has_zero_finite_input_gradient():
    layers = get_layers()
    system = tiny_system()
    target = torch.tensor([[[0.5], [float("nan")], [-0.1]]], dtype=torch.float64, requires_grad=True)
    lp, mean, cov = layers._kalman_filter(target, torch.isfinite(target), **system)
    (lp.sum() + mean.sum() + cov.sum()).backward()
    assert torch.isfinite(target.grad).all()
    assert target.grad[0, 1, 0] == 0


def test_joint_samples_retain_cross_time_covariance():
    layers = get_layers()
    torch.manual_seed(761)
    system = tiny_system()
    expected_mean, expected_cov = dense_joint_oracle(**system)
    samples = layers._sample_paths(**system, num_samples=60000)
    assert samples.shape == (60000, 1, 3, 1)
    empirical_mean = samples[:, 0, :, 0].mean(0)
    empirical_cov = torch.cov(samples[:, 0, :, 0].T)
    torch.testing.assert_close(empirical_mean, expected_mean[0], rtol=0, atol=0.025)
    torch.testing.assert_close(empirical_cov, expected_cov[0], rtol=0.04, atol=0.025)
    assert empirical_cov[0, 1] > 0.3


def test_cholesky_preserves_valid_rows_and_bounds_repair():
    layers = get_layers()
    valid = torch.tensor([[2.0, 0.3], [0.3, 0.5]], dtype=torch.float64)
    singular = torch.ones(2, 2, dtype=torch.float64)
    covariance = torch.stack((valid, singular))
    factor = layers._safe_cholesky(covariance)
    torch.testing.assert_close(factor[0], torch.linalg.cholesky(valid), rtol=0, atol=0)
    repaired = factor[1] @ factor[1].T
    assert torch.isfinite(repaired).all()
    assert (repaired - singular).abs().max() <= 101 * torch.finfo(torch.float64).eps
    invalid = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]], dtype=torch.float64)
    with pytest.raises(ValueError, match="covariance"):
        layers._safe_cholesky(invalid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joseph_filter_long_context_is_symmetric_and_finite(dtype):
    layers = get_layers()
    structure = layers._ISSM(periods=(4,), add_trend=True, dtype=dtype)
    target = torch.ones(2, 128, 1, dtype=dtype)
    indices = torch.arange(128).remainder(4).view(1, 128, 1).expand(2, -1, -1)
    lp, mean, cov = layers._kalman_filter(
        target,
        torch.ones_like(target, dtype=torch.bool),
        mean=torch.zeros(2, 6, dtype=dtype),
        cov=torch.eye(6, dtype=dtype).expand(2, -1, -1).clone(),
        emission=structure.emission(indices, target),
        transition=structure.transition,
        innovation=torch.full_like(target, 1e-4),
        noise_std=torch.full_like(target, 1e-4),
        residual=torch.zeros_like(target),
    )
    assert torch.isfinite(lp).all() and torch.isfinite(mean).all() and torch.isfinite(cov).all()
    torch.testing.assert_close(cov, cov.transpose(-1, -2), rtol=0, atol=0)
    assert torch.linalg.eigvalsh(cov.double()).min() >= -1e-6


def test_residual_lstm_continuation_matches_concatenation():
    layers = get_layers()
    torch.manual_seed(920)
    network = layers._ResidualLSTM(input_size=3, hidden_size=4, num_layers=2).double()
    x = torch.randn(2, 7, 3, dtype=torch.float64)
    combined, combined_state = network(x)
    past, state = network(x[:, :4])
    future, final_state = network(x[:, 4:], state)
    torch.testing.assert_close(torch.cat((past, future), 1), combined, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(final_state, combined_state, strict=True):
        for value, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, wanted, rtol=1e-12, atol=1e-12)
    first, _ = network.layers[0](x)
    second, _ = network.layers[1](first)
    torch.testing.assert_close(combined, first + second, rtol=1e-12, atol=1e-12)
