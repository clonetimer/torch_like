"""Common generative-domain configs and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torch_like.models.base import BaseModel


@dataclass
class GenerativeConfig:
    in_channels: int = 3
    image_size: int = 32
    d_model: int = 64
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    dropout: float = 0.0


@dataclass
class GenerativeBatch:
    x_start: torch.Tensor
    timesteps: torch.Tensor
    noise: Optional[torch.Tensor] = None


class GenerativeModel(BaseModel):
    """Base class for generative models."""

    def predict(self, batch: GenerativeBatch) -> torch.Tensor:
        return self(batch.x_start, batch.timesteps)

    def make_generative_batch(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> GenerativeBatch:
        return GenerativeBatch(x_start=x_start, timesteps=timesteps, noise=noise)
