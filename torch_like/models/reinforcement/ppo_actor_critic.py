"""PPO actor-critic model with clipped surrogate objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch_like.models.reinforcement.actor_critic import ActorCritic, ActorCriticConfig
from torch_like.models.reinforcement.common import ReinforcementBatch


@dataclass
class PPOActorCriticConfig(ActorCriticConfig):
    clip_ratio: float = 0.2


class PPOActorCritic(ActorCritic):
    """Actor-critic core exposing the PPO clipped policy objective."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        d_model: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        clip_ratio: float = 0.2,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            hidden_layers=hidden_layers,
            dropout=dropout,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
        self._config = PPOActorCriticConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            hidden_layers=hidden_layers,
            dropout=dropout,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            clip_ratio=clip_ratio,
        )
        self.clip_ratio = clip_ratio

    def ppo_loss(
        self,
        batch: ReinforcementBatch,
        clip_ratio: float | None = None,
        entropy_coef: float | None = None,
        value_coef: float | None = None,
    ) -> torch.Tensor:
        if batch.actions is None or batch.old_log_probs is None or batch.returns is None or batch.advantages is None:
            raise ValueError("ppo_loss requires actions, old_log_probs, returns, and advantages")
        clip = self.clip_ratio if clip_ratio is None else clip_ratio
        entropy_weight = self.entropy_coef if entropy_coef is None else entropy_coef
        value_weight = self.value_coef if value_coef is None else value_coef

        old_log_probs = batch.old_log_probs.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        advantages = batch.advantages.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        returns = batch.returns.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)

        log_probs, entropy, values = self.evaluate_actions(batch.states, batch.actions)
        ratio = torch.exp(log_probs - old_log_probs)
        unclipped = ratio * advantages.detach()
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages.detach()
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_bonus = entropy.mean()
        return policy_loss + value_weight * value_loss - entropy_weight * entropy_bonus

    def loss_step(self, batch: ReinforcementBatch) -> torch.Tensor:
        return self.ppo_loss(batch)
