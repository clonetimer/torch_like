"""Common tabular-domain configs and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torch_like.models.base import BaseModel


@dataclass
class TabularConfig:
    input_dim: int
    output_dim: int
    d_model: int = 64
    dropout: float = 0.1
    categories: tuple[int, ...] = ()


@dataclass
class TabularBatch:
    x_cont: torch.Tensor
    x_cat: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None


class TabularModel(BaseModel):
    """Base class for tabular prediction models."""

    def predict(self, batch: TabularBatch) -> torch.Tensor:
        return self(batch.x_cont, batch.x_cat)

    def make_tabular_batch(
        self,
        x_cont: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> TabularBatch:
        return TabularBatch(x_cont=x_cont, x_cat=x_cat, y=y)

    def loss_step(self, batch: TabularBatch) -> torch.Tensor:
        if batch.y is None:
            raise ValueError("loss_step requires batch.y")
        pred = self.predict(batch)
        if pred.size(-1) > 1 and batch.y.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            return F.cross_entropy(pred, batch.y.long())
        target = batch.y.to(device=pred.device, dtype=pred.dtype)
        if target.shape != pred.shape:
            target = target.reshape_as(pred)
        return F.mse_loss(pred, target)
