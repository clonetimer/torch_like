"""Common RUL configs and model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

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
    targets: Optional[torch.Tensor] = None


class RULModel(BaseModel):
    """Base class for RUL regression models."""

    def predict(self, batch: RULBatch) -> torch.Tensor:
        return self(batch.x, batch.x_mark)

    def loss_step(self, batch: RULBatch, loss: str = "mse") -> torch.Tensor:
        if batch.targets is None:
            raise ValueError("RULBatch.targets is required for loss_step")
        predictions = self(batch.x, batch.x_mark)
        targets = batch.targets.to(device=predictions.device, dtype=predictions.dtype)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"RUL predictions and targets must have the same shape: {predictions.shape} != {targets.shape}"
            )
        if loss == "mse":
            return F.mse_loss(predictions, targets)
        if loss == "mae":
            return F.l1_loss(predictions, targets)
        if loss == "huber":
            return F.huber_loss(predictions, targets)
        raise ValueError("loss must be one of: mse, mae, huber")

    def make_rul_batch(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> RULBatch:
        if x_mark is None:
            x_mark = torch.zeros(x.size(0), x.size(1), 4, device=x.device, dtype=x.dtype)
        return RULBatch(x=x, x_mark=x_mark, targets=targets)
