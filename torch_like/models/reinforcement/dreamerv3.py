"""DreamerV3-style discrete world model and imagination actor-critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.reinforcement.common import (
    ReinforcementBatch,
    ReinforcementConfig,
    ReinforcementModel,
    build_mlp,
)


@dataclass
class DreamerV3Config(ReinforcementConfig):
    deter_dim: int = 128
    stochastic_dim: int = 16
    classes: int = 16
    imagination_horizon: int = 15
    free_nats: float = 1.0
    kl_balance: float = 0.8
    discount: float = 0.99


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


class DiscreteRSSM(nn.Module):
    """Recurrent state-space model with straight-through categorical latents."""

    def __init__(self, state_dim, action_dim, deter_dim, stochastic_dim, classes, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stochastic_dim = stochastic_dim
        self.classes = classes
        self.stoch_size = stochastic_dim * classes
        self.encoder = build_mlp(state_dim, hidden_dim, hidden_dim, 1, 0.0)
        self.recurrent = nn.GRUCell(self.stoch_size + action_dim, deter_dim)
        self.prior = build_mlp(deter_dim, hidden_dim, self.stoch_size, 1, 0.0)
        self.posterior = build_mlp(deter_dim + hidden_dim, hidden_dim, self.stoch_size, 1, 0.0)

    def initial_state(self, batch_size, *, device=None, dtype=None):
        parameter = next(self.parameters())
        device, dtype = device or parameter.device, dtype or parameter.dtype
        deter = torch.zeros(batch_size, self.deter_dim, device=device, dtype=dtype)
        logits = torch.zeros(batch_size, self.stochastic_dim, self.classes, device=device, dtype=dtype)
        stoch = F.one_hot(logits.argmax(-1), self.classes).to(dtype).flatten(1)
        return {"deter": deter, "stoch": stoch, "logits": logits}

    def _distribution(self, logits):
        return torch.distributions.OneHotCategoricalStraightThrough(logits=logits)

    def _sample(self, logits, deterministic=False):
        distribution = self._distribution(logits)
        sample = (
            F.one_hot(logits.argmax(-1), self.classes).to(logits.dtype) if deterministic else distribution.rsample()
        )
        return sample.flatten(1)

    def imagine_step(self, previous, action, deterministic=False):
        deter = self.recurrent(torch.cat([previous["stoch"], action], dim=-1), previous["deter"])
        logits = self.prior(deter).view(-1, self.stochastic_dim, self.classes)
        return {"deter": deter, "stoch": self._sample(logits, deterministic), "logits": logits}

    def observe_step(self, previous, action, observation, deterministic=False):
        prior = self.imagine_step(previous, action, deterministic)
        embedding = self.encoder(observation)
        logits = self.posterior(torch.cat([prior["deter"], embedding], dim=-1)).view(
            -1, self.stochastic_dim, self.classes
        )
        posterior = {"deter": prior["deter"], "stoch": self._sample(logits, deterministic), "logits": logits}
        return posterior, prior

    def observe(self, observations, actions, deterministic=False):
        if observations.dim() != 3 or actions.dim() != 3 or observations.shape[:2] != actions.shape[:2]:
            raise ValueError("observations and actions must have matching (batch, time, feature) shapes")
        state = self.initial_state(observations.size(0), device=observations.device, dtype=observations.dtype)
        posteriors, priors = [], []
        for index in range(observations.size(1)):
            state, prior = self.observe_step(state, actions[:, index], observations[:, index], deterministic)
            posteriors.append(state)
            priors.append(prior)

        def stack(items, key):
            return torch.stack([item[key] for item in items], dim=1)

        posterior = {key: stack(posteriors, key) for key in ("deter", "stoch", "logits")}
        prior = {key: stack(priors, key) for key in ("deter", "stoch", "logits")}
        return posterior, prior

    @staticmethod
    def features(state):
        return torch.cat([state["deter"], state["stoch"]], dim=-1)


class DreamerV3(ReinforcementModel):
    """Composable DreamerV3 world model with latent imagination interfaces."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        d_model: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        deter_dim: int = 128,
        stochastic_dim: int = 16,
        classes: int = 16,
        imagination_horizon: int = 15,
        free_nats: float = 1.0,
        kl_balance: float = 0.8,
        discount: float = 0.99,
    ):
        super().__init__()
        if min(state_dim, action_dim, d_model, deter_dim, stochastic_dim, classes, imagination_horizon) <= 0:
            raise ValueError("DreamerV3 dimensions and horizon must be positive")
        if hidden_layers < 0 or not 0 <= kl_balance <= 1 or not 0 < discount <= 1 or free_nats < 0:
            raise ValueError("Invalid DreamerV3 optimization configuration")
        self.state_dim, self.action_dim = state_dim, action_dim
        self.imagination_horizon, self.free_nats, self.kl_balance, self.discount = (
            imagination_horizon,
            free_nats,
            kl_balance,
            discount,
        )
        self._config = DreamerV3Config(
            state_dim,
            action_dim,
            d_model,
            hidden_layers,
            dropout,
            deter_dim,
            stochastic_dim,
            classes,
            imagination_horizon,
            free_nats,
            kl_balance,
            discount,
        )
        self.rssm = DiscreteRSSM(state_dim, action_dim, deter_dim, stochastic_dim, classes, d_model)
        feature_dim = deter_dim + stochastic_dim * classes
        self.observation_head = build_mlp(feature_dim, d_model, state_dim, hidden_layers, dropout)
        self.reward_head = build_mlp(feature_dim, d_model, 1, hidden_layers, dropout)
        self.continue_head = build_mlp(feature_dim, d_model, 1, hidden_layers, dropout)
        self.actor = build_mlp(feature_dim, d_model, action_dim * 2, hidden_layers, dropout)
        self.critic = build_mlp(feature_dim, d_model, 1, hidden_layers, dropout)

    def _actor_distribution(self, features):
        mean, raw_std = self.actor(features).chunk(2, dim=-1)
        std = F.softplus(raw_std) + 0.1
        return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)

    def policy(self, features, deterministic=False):
        distribution = self._actor_distribution(features)
        action = distribution.mean if deterministic else distribution.rsample()
        return torch.tanh(action)

    def value(self, features):
        return symexp(self.critic(features).squeeze(-1))

    def forward(self, states, actions, deterministic=False):
        posterior, prior = self.rssm.observe(states, actions, deterministic)
        features = self.rssm.features(posterior)
        return {
            "posterior": posterior,
            "prior": prior,
            "features": features,
            "reconstructed_states": symexp(self.observation_head(features)),
            "predicted_rewards": symexp(self.reward_head(features).squeeze(-1)),
            "continue_logits": self.continue_head(features).squeeze(-1),
            "values": self.value(features),
        }

    def imagine(self, start_state, horizon=None, deterministic=False):
        horizon = self.imagination_horizon if horizon is None else horizon
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        state = {key: value[:, -1] if value.dim() > 2 else value for key, value in start_state.items()}
        trajectories = {key: [] for key in ("features", "actions", "rewards", "continues", "values")}
        for _ in range(horizon):
            features = self.rssm.features(state)
            action = self.policy(features, deterministic)
            state = self.rssm.imagine_step(state, action, deterministic)
            next_features = self.rssm.features(state)
            trajectories["features"].append(next_features)
            trajectories["actions"].append(action)
            trajectories["rewards"].append(symexp(self.reward_head(next_features).squeeze(-1)))
            trajectories["continues"].append(torch.sigmoid(self.continue_head(next_features).squeeze(-1)))
            trajectories["values"].append(self.value(next_features))
        return {key: torch.stack(value, dim=1) for key, value in trajectories.items()}

    def _kl_loss(self, posterior_logits, prior_logits):
        posterior = torch.distributions.Categorical(logits=posterior_logits)
        prior = torch.distributions.Categorical(logits=prior_logits)
        dynamics = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=posterior_logits.detach()), prior
        ).sum(-1)
        representation = torch.distributions.kl_divergence(
            posterior, torch.distributions.Categorical(logits=prior_logits.detach())
        ).sum(-1)
        kl = self.kl_balance * dynamics + (1 - self.kl_balance) * representation
        return kl.clamp_min(self.free_nats).mean()

    def world_model_loss(self, batch: ReinforcementBatch):
        if batch.actions is None or batch.rewards is None:
            raise ValueError("DreamerV3 world_model_loss requires actions and rewards")
        outputs = self(batch.states, batch.actions)
        target_states = batch.states.to(outputs["reconstructed_states"])
        target_rewards = batch.rewards.to(outputs["predicted_rewards"])
        target_continues = (
            torch.ones_like(target_rewards) if batch.dones is None else 1 - batch.dones.to(target_rewards)
        )
        reconstruction = F.mse_loss(symlog(outputs["reconstructed_states"]), symlog(target_states))
        reward = F.mse_loss(symlog(outputs["predicted_rewards"]), symlog(target_rewards))
        continuation = F.binary_cross_entropy_with_logits(outputs["continue_logits"], target_continues)
        kl = self._kl_loss(outputs["posterior"]["logits"], outputs["prior"]["logits"])
        return reconstruction + reward + continuation + kl

    def actor_critic_loss(self, posterior, horizon=None):
        trajectory = self.imagine(posterior, horizon)
        rewards, continues, values = trajectory["rewards"], trajectory["continues"], trajectory["values"]
        target = values[:, -1].detach()
        returns = []
        for index in range(values.size(1) - 1, -1, -1):
            target = rewards[:, index] + self.discount * continues[:, index] * target
            returns.append(target)
        returns = torch.stack(list(reversed(returns)), dim=1)
        actor_loss = -returns.mean()
        critic_loss = F.mse_loss(symlog(values), symlog(returns.detach()))
        return {"actor_loss": actor_loss, "critic_loss": critic_loss, "imagined_returns": returns}

    def loss_step(self, batch: ReinforcementBatch):
        return self.world_model_loss(batch)
