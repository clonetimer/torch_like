"""Common recommendation-domain configs and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torch_like.models.base import BaseModel


@dataclass
class RecommendationConfig:
    output_dim: int = 1
    d_model: int = 64
    dropout: float = 0.1


@dataclass
class RecommendationBatch:
    user_ids: Optional[torch.Tensor] = None
    item_ids: Optional[torch.Tensor] = None
    x_cat: Optional[torch.Tensor] = None
    x_cont: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None


class RecommendationModel(BaseModel):
    """Base class for recommender and CTR prediction models."""

    def predict(self, batch: RecommendationBatch) -> torch.Tensor:
        return self(
            user_ids=batch.user_ids,
            item_ids=batch.item_ids,
            x_cat=batch.x_cat,
            x_cont=batch.x_cont,
        )

    def make_recommendation_batch(
        self,
        user_ids: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        x_cont: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> RecommendationBatch:
        return RecommendationBatch(user_ids=user_ids, item_ids=item_ids, x_cat=x_cat, x_cont=x_cont, y=y)

    def loss_step(self, batch: RecommendationBatch) -> torch.Tensor:
        if batch.y is None:
            raise ValueError("loss_step requires batch.y")
        logits = self.predict(batch)
        if logits.size(-1) > 1 and batch.y.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            return F.cross_entropy(logits, batch.y.long())
        target = batch.y.to(device=logits.device, dtype=logits.dtype)
        if target.shape != logits.shape:
            target = target.reshape_as(logits)
        return F.binary_cross_entropy_with_logits(logits, target)
