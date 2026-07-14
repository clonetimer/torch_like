"""Deep Q-Network style value model."""

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
class DQNConfig(ReinforcementConfig):
    gamma: float = 0.99


class DQN(ReinforcementModel):
    """Q-network for discrete-action value-based reinforcement learning."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        d_model: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        gamma: float = 0.99,
    ):
        super().__init__()
        self._config = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            hidden_layers=hidden_layers,
            dropout=dropout,
            gamma=gamma,
        )
        self.gamma = gamma
        self.network = build_mlp(state_dim, d_model, action_dim, hidden_layers, dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    def q_values(self, states: torch.Tensor) -> torch.Tensor:
        return self(states)

    def greedy_action(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_values(states).argmax(dim=-1)

    def td_loss(
        self,
        batch: ReinforcementBatch,
        target_model: Optional["DQN"] = None,
        gamma: Optional[float] = None,
    ) -> torch.Tensor:
        if batch.actions is None or batch.rewards is None or batch.next_states is None or batch.dones is None:
            raise ValueError("td_loss requires actions, rewards, next_states, and dones")
        discount = self.gamma if gamma is None else gamma
        actions = batch.actions.long().view(-1, 1)
        rewards = batch.rewards.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)
        dones = batch.dones.to(device=batch.states.device, dtype=batch.states.dtype).view(-1)

        q_selected = self.q_values(batch.states).gather(dim=-1, index=actions).squeeze(-1)
        target_network = self if target_model is None else target_model
        with torch.no_grad():
            next_q = target_network.q_values(batch.next_states).max(dim=-1).values
            target = rewards + discount * (1.0 - dones) * next_q
        return F.mse_loss(q_selected, target)

    def loss_step(self, batch: ReinforcementBatch) -> torch.Tensor:
        return self.td_loss(batch)
