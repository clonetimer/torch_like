"""Neural Collaborative Filtering model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.recommendation.common import RecommendationConfig, RecommendationModel


@dataclass
class NCFConfig(RecommendationConfig):
    num_users: int = 1
    num_items: int = 1
    mf_dim: int = 32
    mlp_layers: tuple[int, ...] = (64, 32, 16)


class NCF(RecommendationModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        output_dim: int = 1,
        d_model: int = 64,
        dropout: float = 0.1,
        mf_dim: int = 32,
        mlp_layers: tuple[int, ...] = (64, 32, 16),
    ):
        super().__init__()
        if min(num_users, num_items, output_dim, d_model, mf_dim) <= 0:
            raise ValueError("NCF dimensions must be positive")
        self.num_users = num_users
        self.num_items = num_items
        self.output_dim = output_dim
        self._config = NCFConfig(
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            num_users=num_users,
            num_items=num_items,
            mf_dim=mf_dim,
            mlp_layers=tuple(mlp_layers),
        )
        self.user_mf = nn.Embedding(num_users, mf_dim)
        self.item_mf = nn.Embedding(num_items, mf_dim)
        self.user_mlp = nn.Embedding(num_users, d_model)
        self.item_mlp = nn.Embedding(num_items, d_model)
        layers: list[nn.Module] = []
        current = 2 * d_model
        for width in mlp_layers:
            layers.extend([nn.Linear(current, width), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            current = width
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(mf_dim + current, output_dim)

    def forward(
        self,
        user_ids: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        x_cont: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_cat, x_cont
        if user_ids is None or item_ids is None:
            raise ValueError("NCF requires user_ids and item_ids")
        user_ids = user_ids.long()
        item_ids = item_ids.long()
        mf = self.user_mf(user_ids) * self.item_mf(item_ids)
        mlp_input = torch.cat([self.user_mlp(user_ids), self.item_mlp(item_ids)], dim=-1)
        mlp = self.mlp(mlp_input)
        return self.head(torch.cat([mf, mlp], dim=-1))
