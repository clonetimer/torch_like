"""Common reinforcement-learning configs and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torch_like.models.base import BaseModel


@dataclass
class ReinforcementConfig:
    state_dim: int = 4
    action_dim: int = 2
    d_model: int = 128
    hidden_layers: int = 2
    dropout: float = 0.0


@dataclass
class ReinforcementBatch:
    states: torch.Tensor
    actions: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    next_states: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    returns_to_go: Optional[torch.Tensor] = None
    timesteps: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    old_log_probs: Optional[torch.Tensor] = None


def build_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int, dropout: float
) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    in_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend([torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()])
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(torch.nn.Linear(in_dim, output_dim))
    return torch.nn.Sequential(*layers)


class ReinforcementModel(BaseModel):
    """Base class for policy/value/Q-network style RL models."""

    def make_reinforcement_batch(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        returns_to_go: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
    ) -> ReinforcementBatch:
        return ReinforcementBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            returns=returns,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            advantages=advantages,
            old_log_probs=old_log_probs,
        )
