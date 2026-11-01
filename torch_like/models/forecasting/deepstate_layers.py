"""Native structural Gaussian state-space primitives for DeepState.

Mathematical/API reference: GluonTS d60556ca71167a3ec1ea807b491dd982d1de2112,
mx/model/deepstate/issm.py and mx/distribution/lds.py. Adapted to PyTorch with
explicit masking, covariance symmetrization and bounded sampling jitter.
Gluon Time Series
Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _ISSM(nn.Module):
    """Fixed transition with caller-indexed seasonal emission/innovation vectors."""

    def __init__(self, periods: tuple[int, ...], add_trend: bool, dtype: torch.dtype):
        super().__init__()
        self.periods = tuple(periods)
        self.nonseasonal_dim = 2 if add_trend else 1
        self.state_dim = self.nonseasonal_dim + sum(periods)
        transition = torch.eye(self.state_dim, dtype=dtype)
        if add_trend:
            transition[0, 1] = 1
        self.register_buffer("transition", transition)

    def emission(self, indices: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        parts = [reference.new_ones(*reference.shape[:2], self.nonseasonal_dim)]
        for column, period in enumerate(self.periods):
            parts.append(F.one_hot(indices[..., column].long(), num_classes=period).to(dtype=reference.dtype))
        return torch.cat(parts, dim=-1)


class _ResidualLSTM(nn.Module):
    """Zero-zoneout reference subset; upper layers retain residual connections."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.LSTM(input_size if index == 0 else hidden_size, hidden_size, batch_first=True)
            for index in range(num_layers)
        )

    def forward(self, inputs: torch.Tensor, states=None):
        output = inputs
        final_states = []
        for index, layer in enumerate(self.layers):
            previous = output
            output, state = layer(output, None if states is None else states[index])
            if index:
                output = output + previous
            final_states.append(state)
        return output, final_states


def _observation(mean, cov, emission, noise_std, residual):
    projected_cov = (cov @ emission.unsqueeze(-1)).squeeze(-1)
    variance = (emission * projected_cov).sum(-1) + noise_std.square()
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("DeepState observation variance must be positive and finite; rescale inputs or use float64")
    return (emission * mean).sum(-1) + residual, variance, projected_cov


def _transition(mean, cov, transition, emission, innovation):
    next_mean = (transition @ mean.unsqueeze(-1)).squeeze(-1)
    direction = emission * innovation
    next_cov = transition @ cov @ transition.T + direction.unsqueeze(-1) * direction.unsqueeze(-2)
    return next_mean, (next_cov + next_cov.transpose(-1, -2)) * 0.5


def _kalman_filter(
    targets: torch.Tensor,
    observed: torch.Tensor,
    mean: torch.Tensor,
    cov: torch.Tensor,
    emission: torch.Tensor,
    transition: torch.Tensor,
    innovation: torch.Tensor,
    noise_std: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score/update current observations, then transition, including after the last.

    Inputs have already been validated by the public model. Returned state is
    ready for the next observation, not the posterior before its transition.
    Accumulate in float64, including for float32 networks: seasonal covariance
    cancellation can otherwise exceed the configured small observation noise.
    Keep the posterior in float64 across the history/future boundary.
    """
    targets, mean, cov, emission, transition, innovation, noise_std, residual = (
        value.double() for value in (targets, mean, cov, emission, transition, innovation, noise_std, residual)
    )
    clean = torch.where(observed, targets, torch.zeros_like(targets))
    identity = torch.eye(mean.shape[-1], dtype=mean.dtype, device=mean.device)
    log_prob = []
    for t in range(targets.shape[1]):
        a = emission[:, t]
        sigma = noise_std[:, t, 0]
        mu, variance, projected_cov = _observation(mean, cov, a, sigma, residual[:, t, 0])
        delta = clean[:, t, 0] - mu
        gain = projected_cov / variance.unsqueeze(-1)
        filtered_mean = mean + gain * delta.unsqueeze(-1)
        imka = identity - gain.unsqueeze(-1) * a.unsqueeze(-2)
        filtered_cov = imka @ cov @ imka.transpose(-1, -2)
        filtered_cov = filtered_cov + (gain * sigma.unsqueeze(-1)).unsqueeze(-1) * (
            gain * sigma.unsqueeze(-1)
        ).unsqueeze(-2)
        present = observed[:, t, 0]
        mean = torch.where(present.unsqueeze(-1), filtered_mean, mean)
        cov = torch.where(present[:, None, None], filtered_cov, cov)
        lp = -0.5 * (math.log(2 * math.pi) + variance.log() + delta.square() / variance)
        log_prob.append(torch.where(present, lp, torch.zeros_like(lp)))
        mean, cov = _transition(mean, cov, transition, a, innovation[:, t])
    return torch.stack(log_prob, dim=1), mean, cov


def _marginals(
    mean: torch.Tensor,
    cov: torch.Tensor,
    emission: torch.Tensor,
    transition: torch.Tensor,
    innovation: torch.Tensor,
    noise_std: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic future marginal means/variances without conditioning on targets."""
    mean, cov, emission, transition, innovation, noise_std, residual = (
        value.double() for value in (mean, cov, emission, transition, innovation, noise_std, residual)
    )
    means, variances = [], []
    for t in range(emission.shape[1]):
        a = emission[:, t]
        mu, variance, _ = _observation(mean, cov, a, noise_std[:, t, 0], residual[:, t, 0])
        means.append(mu)
        variances.append(variance)
        mean, cov = _transition(mean, cov, transition, a, innovation[:, t])
    return torch.stack(means, dim=1).unsqueeze(-1), torch.stack(variances, dim=1).unsqueeze(-1)


def _safe_cholesky(cov: torch.Tensor) -> torch.Tensor:
    """Repair roundoff only, and never perturb a batch row that decomposes as-is."""
    if not torch.isfinite(cov).all():
        raise ValueError("DeepState sampling covariance must be finite")
    symmetric = (cov + cov.transpose(-1, -2)) * 0.5
    factor, info = torch.linalg.cholesky_ex(symmetric)
    if not info.any():
        return factor
    precision = torch.finfo(cov.dtype)
    magnitude = symmetric.diagonal(dim1=-2, dim2=-1).abs().amax(-1).clamp_min(precision.tiny)
    identity = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    failed = info != 0
    for multiplier in (1, 10, 100):
        jitter = multiplier * precision.eps * magnitude
        candidate = symmetric + torch.where(failed, jitter, torch.zeros_like(jitter))[:, None, None] * identity
        factor, info = torch.linalg.cholesky_ex(candidate)
        if not info.any():
            return factor
    raise ValueError("DeepState sampling covariance is not positive definite within bounded roundoff jitter")


def _sample_paths(
    mean: torch.Tensor,
    cov: torch.Tensor,
    emission: torch.Tensor,
    transition: torch.Tensor,
    innovation: torch.Tensor,
    noise_std: torch.Tensor,
    residual: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Draw one evolving state per path, with independent scalar step innovations."""
    mean, cov, emission, transition, innovation, noise_std, residual = (
        value.double() for value in (mean, cov, emission, transition, innovation, noise_std, residual)
    )
    batch, horizon, state_dim = emission.shape
    normal = torch.randn(num_samples, batch, state_dim, 1, dtype=mean.dtype, device=mean.device)
    state = mean.unsqueeze(0) + (_safe_cholesky(cov).unsqueeze(0) @ normal).squeeze(-1)
    observation_noise = torch.randn(num_samples, batch, horizon, dtype=mean.dtype, device=mean.device)
    state_noise = torch.randn(num_samples, batch, horizon, dtype=mean.dtype, device=mean.device)
    samples = []
    for t in range(horizon):
        a = emission[:, t]
        value = (state * a).sum(-1) + residual[:, t, 0] + observation_noise[:, :, t] * noise_std[:, t, 0]
        samples.append(value)
        state = state @ transition.T + a * innovation[:, t] * state_noise[:, :, t, None]
    return torch.stack(samples, dim=2).unsqueeze(-1)
