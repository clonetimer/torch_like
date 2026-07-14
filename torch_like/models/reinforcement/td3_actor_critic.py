"""TD3-style deterministic actor-critic for continuous control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torch_like.models.reinforcement.common import (
    ReinforcementBatch,
    ReinforcementConfig,
    ReinforcementModel,
    build_mlp,
)


@dataclass
class TD3ActorCriticConfig(ReinforcementConfig):
    action_limit: float = 1.0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    gamma: float = 0.99


class TD3ActorCritic(ReinforcementModel):
    """Deterministic policy with twin critics and target-policy smoothing."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        d_model: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        action_limit: float = 1.0,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        gamma: float = 0.99,
    ):
        super().__init__()
        if min(state_dim, action_dim, d_model) <= 0 or hidden_layers < 0:
            raise ValueError("TD3ActorCritic dimensions must be positive and hidden_layers must be non-negative")
        if action_limit <= 0:
            raise ValueError("action_limit must be positive")
        if policy_noise < 0 or noise_clip < 0:
            raise ValueError("policy_noise and noise_clip must be non-negative")
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self._config = TD3ActorCriticConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            hidden_layers=hidden_layers,
            dropout=dropout,
            action_limit=action_limit,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            gamma=gamma,
        )
        self.actor = build_mlp(state_dim, d_model, action_dim, hidden_layers, dropout)
        self.q1 = build_mlp(state_dim + action_dim, d_model, 1, hidden_layers, dropout)
        self.q2 = build_mlp(state_dim + action_dim, d_model, 1, hidden_layers, dropout)

    def policy(self, states: torch.Tensor) -> torch.Tensor:
        return self.action_limit * torch.tanh(self.actor(states))

    def target_actions(
        self,
        states: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        policy_noise: Optional[float] = None,
        noise_clip: Optional[float] = None,
    ) -> torch.Tensor:
        actions = self.policy(states)
        sigma = self.policy_noise if policy_noise is None else policy_noise
        clip = self.noise_clip if noise_clip is None else noise_clip
        if sigma > 0:
            if noise is None:
                noise = torch.randn_like(actions) * sigma
            noise = noise.clamp(-clip, clip)
            actions = actions + noise
        return actions.clamp(-self.action_limit, self.action_limit)

    def q_values(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([states, actions], dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)

    def min_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.q_values(states, actions)
        return torch.minimum(q1, q2)

    def actor_loss(self, states: torch.Tensor) -> torch.Tensor:
        actions = self.policy(states)
        q1, _ = self.q_values(states, actions)
        return -q1.mean()

    def critic_loss(
        self,
        batch: ReinforcementBatch,
        target_model: Optional["TD3ActorCritic"] = None,
        gamma: Optional[float] = None,
        policy_noise: Optional[float] = None,
        noise_clip: Optional[float] = None,
    ) -> torch.Tensor:
        if batch.actions is None or batch.rewards is None or batch.next_states is None or batch.dones is None:
            raise ValueError("critic_loss requires actions, rewards, next_states, and dones")
        discount = self.gamma if gamma is None else gamma
        rewards = batch.rewards.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        dones = batch.dones.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        current_q1, current_q2 = self.q_values(batch.states, batch.actions)
        target = self if target_model is None else target_model
        with torch.no_grad():
            next_actions = target.target_actions(
                batch.next_states,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
            )
            next_q = target.min_q(batch.next_states, next_actions)
            target_q = rewards + discount * (1.0 - dones) * next_q
        return F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy(states)

    def loss_step(self, batch: ReinforcementBatch) -> torch.Tensor:
        return self.critic_loss(batch)
