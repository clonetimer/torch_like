"""Actor-critic policy/value model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from torch_like.models.reinforcement.common import (
    ReinforcementBatch,
    ReinforcementConfig,
    ReinforcementModel,
    build_mlp,
)


@dataclass
class ActorCriticConfig(ReinforcementConfig):
    entropy_coef: float = 0.01
    value_coef: float = 0.5


class ActorCritic(ReinforcementModel):
    """Discrete-action actor-critic core with policy and value heads."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        d_model: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        super().__init__()
        self._config = ActorCriticConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            hidden_layers=hidden_layers,
            dropout=dropout,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.encoder = build_mlp(state_dim, d_model, d_model, hidden_layers, dropout)
        self.policy_head = torch.nn.Linear(d_model, action_dim)
        self.value_head = torch.nn.Linear(d_model, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(states)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def policy_logits(self, states: torch.Tensor) -> torch.Tensor:
        logits, _ = self(states)
        return logits

    def value(self, states: torch.Tensor) -> torch.Tensor:
        _, values = self(states)
        return values

    def distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.policy_logits(states))

    def act(self, states: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self(states)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())
        entropy = dist.entropy()
        return log_probs, entropy, values

    def actor_critic_loss(
        self,
        batch: ReinforcementBatch,
        entropy_coef: float | None = None,
        value_coef: float | None = None,
    ) -> torch.Tensor:
        if batch.actions is None or batch.returns is None or batch.advantages is None:
            raise ValueError("actor_critic_loss requires actions, returns, and advantages")
        entropy_weight = self.entropy_coef if entropy_coef is None else entropy_coef
        value_weight = self.value_coef if value_coef is None else value_coef
        advantages = batch.advantages.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        returns = batch.returns.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        log_probs, entropy, values = self.evaluate_actions(batch.states, batch.actions)
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_bonus = entropy.mean()
        return policy_loss + value_weight * value_loss - entropy_weight * entropy_bonus

    def loss_step(self, batch: ReinforcementBatch) -> torch.Tensor:
        return self.actor_critic_loss(batch)
