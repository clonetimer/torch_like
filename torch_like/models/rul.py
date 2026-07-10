"""Common RUL configs and model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torch_like.models.base import BaseModel


@dataclass
class RULConfig:
    enc_in: int = 2
    seq_len: int = 2048
    d_model: int = 256
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 1024
    dropout: float = 0.1


@dataclass
class RULBatch:
    x: torch.Tensor
    x_mark: Optional[torch.Tensor] = None


class RULModel(BaseModel):
    """Base class for RUL regression models."""

    def predict(self, batch: RULBatch) -> torch.Tensor:
        return self(batch.x, batch.x_mark)

    def make_rul_batch(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> RULBatch:
        if x_mark is None:
            x_mark = torch.zeros(x.size(0), x.size(1), 4, device=x.device, dtype=x.dtype)
        return RULBatch(x=x, x_mark=x_mark)
