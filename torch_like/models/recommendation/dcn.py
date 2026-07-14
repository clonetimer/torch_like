"""Deep & Cross Network for feature crossing recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.recommendation.common import RecommendationConfig, RecommendationModel


@dataclass
class DCNConfig(RecommendationConfig):
    categories: tuple[int, ...] = ()
    num_continuous: int = 0
    embedding_dim: int = 16
    cross_layers: int = 3
    mlp_layers: tuple[int, ...] = (64, 32)


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        projection = torch.sum(x * self.weight, dim=-1, keepdim=True)
        return x0 * projection + self.bias + x


class DCN(RecommendationModel):
    def __init__(
        self,
        categories: tuple[int, ...] = (),
        num_continuous: int = 0,
        output_dim: int = 1,
        d_model: int = 64,
        dropout: float = 0.1,
        embedding_dim: int = 16,
        cross_layers: int = 3,
        mlp_layers: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        if not categories and num_continuous <= 0:
            raise ValueError("DCN requires categorical or continuous features")
        if min(output_dim, d_model, embedding_dim, cross_layers) <= 0 or num_continuous < 0:
            raise ValueError("DCN dimensions must be positive and num_continuous must be non-negative")
        self.categories = tuple(categories)
        self.num_continuous = num_continuous
        self.output_dim = output_dim
        self._config = DCNConfig(
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            categories=self.categories,
            num_continuous=num_continuous,
            embedding_dim=embedding_dim,
            cross_layers=cross_layers,
            mlp_layers=tuple(mlp_layers),
        )
        self.cat_embeddings = nn.ModuleList([nn.Embedding(size, embedding_dim) for size in self.categories])
        input_dim = len(self.categories) * embedding_dim + num_continuous
        self.cross = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_layers)])
        layers: list[nn.Module] = []
        current = input_dim
        for width in mlp_layers:
            layers.extend([nn.Linear(current, width), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            current = width
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim + current, output_dim)

    def _features(self, x_cat: torch.Tensor | None, x_cont: torch.Tensor | None) -> torch.Tensor:
        features: list[torch.Tensor] = []
        if self.categories:
            if x_cat is None or x_cat.shape[-1] != len(self.categories):
                raise ValueError(f"x_cat must have shape (batch, {len(self.categories)})")
            features.append(
                torch.cat([emb(x_cat[:, idx].long()) for idx, emb in enumerate(self.cat_embeddings)], dim=-1)
            )
        if self.num_continuous > 0:
            if x_cont is None or x_cont.shape[-1] != self.num_continuous:
                raise ValueError(f"x_cont must have shape (batch, {self.num_continuous})")
            features.append(x_cont.float())
        return torch.cat(features, dim=-1)

    def forward(
        self,
        user_ids: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        x_cont: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del user_ids, item_ids
        x0 = self._features(x_cat, x_cont)
        crossed = x0
        for layer in self.cross:
            crossed = layer(x0, crossed)
        deep = self.deep(x0)
        return self.head(torch.cat([crossed, deep], dim=-1))
