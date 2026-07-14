"""Soft Actor-Critic style continuous-control actor-critic model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from torch_like.models.reinforcement.common import (
    ReinforcementBatch,
    ReinforcementConfig,
    ReinforcementModel,
    build_mlp,
)


@dataclass
class SACActorCriticConfig(ReinforcementConfig):
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    alpha: float = 0.2
    gamma: float = 0.99


class SACActorCritic(ReinforcementModel):
    """Gaussian policy with twin Q networks for Soft Actor-Critic."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        d_model: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        alpha: float = 0.2,
        gamma: float = 0.99,
    ):
        super().__init__()
        if min(state_dim, action_dim, d_model) <= 0 or hidden_layers < 0:
            raise ValueError("SACActorCritic dimensions must be positive and hidden_layers must be non-negative")
        self._config = SACActorCriticConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            hidden_layers=hidden_layers,
            dropout=dropout,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            alpha=alpha,
            gamma=gamma,
        )
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.alpha = alpha
        self.gamma = gamma
        self.actor = build_mlp(state_dim, d_model, 2 * action_dim, hidden_layers, dropout)
        self.q1 = build_mlp(state_dim + action_dim, d_model, 1, hidden_layers, dropout)
        self.q2 = build_mlp(state_dim + action_dim, d_model, 1, hidden_layers, dropout)

    def policy(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(states).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_action(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy(states)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = Normal(mean, std).rsample()
        action = torch.tanh(pre_tanh)
        normal = Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + eps)
        return action, log_prob.sum(dim=-1)

    def q_values(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([states, actions], dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)

    def min_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.q_values(states, actions)
        return torch.minimum(q1, q2)

    def forward(self, states: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample_action(states, deterministic=deterministic)

    def critic_loss(
        self,
        batch: ReinforcementBatch,
        target_model: Optional["SACActorCritic"] = None,
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        if batch.actions is None or batch.rewards is None or batch.next_states is None or batch.dones is None:
            raise ValueError("critic_loss requires actions, rewards, next_states, and dones")
        discount = self.gamma if gamma is None else gamma
        entropy_weight = self.alpha if alpha is None else alpha
        rewards = batch.rewards.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        dones = batch.dones.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        current_q1, current_q2 = self.q_values(batch.states, batch.actions)
        target = self if target_model is None else target_model
        with torch.no_grad():
            next_actions, next_log_probs = self.sample_action(batch.next_states)
            next_q = target.min_q(batch.next_states, next_actions)
            soft_target = rewards + discount * (1.0 - dones) * (next_q - entropy_weight * next_log_probs)
        return F.mse_loss(current_q1, soft_target) + F.mse_loss(current_q2, soft_target)

    def actor_loss(self, states: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        entropy_weight = self.alpha if alpha is None else alpha
        actions, log_probs = self.sample_action(states)
        q = self.min_q(states, actions)
        return (entropy_weight * log_probs - q).mean()

    def loss_step(self, batch: ReinforcementBatch) -> torch.Tensor:
        return self.critic_loss(batch)

    def gaussian_entropy_target(self) -> float:
        return -float(self.action_dim)
