"""DeepFM model for CTR-style recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.recommendation.common import RecommendationConfig, RecommendationModel


@dataclass
class DeepFMConfig(RecommendationConfig):
    categories: tuple[int, ...] = ()
    num_continuous: int = 0
    fm_dim: int = 16
    mlp_layers: tuple[int, ...] = (64, 32)


class DeepFM(RecommendationModel):
    def __init__(
        self,
        categories: tuple[int, ...],
        num_continuous: int = 0,
        output_dim: int = 1,
        d_model: int = 64,
        dropout: float = 0.1,
        fm_dim: int = 16,
        mlp_layers: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        if not categories:
            raise ValueError("DeepFM requires at least one categorical field")
        if min(output_dim, d_model, fm_dim) <= 0 or num_continuous < 0:
            raise ValueError("DeepFM dimensions must be positive and num_continuous must be non-negative")
        self.categories = tuple(categories)
        self.num_continuous = num_continuous
        self.output_dim = output_dim
        self._config = DeepFMConfig(
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            categories=self.categories,
            num_continuous=num_continuous,
            fm_dim=fm_dim,
            mlp_layers=tuple(mlp_layers),
        )
        self.linear_embeddings = nn.ModuleList([nn.Embedding(size, output_dim) for size in self.categories])
        self.fm_embeddings = nn.ModuleList([nn.Embedding(size, fm_dim) for size in self.categories])
        deep_input_dim = len(self.categories) * fm_dim + num_continuous
        layers: list[nn.Module] = []
        current = deep_input_dim
        for width in mlp_layers:
            layers.extend([nn.Linear(current, width), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            current = width
        self.deep = nn.Sequential(*layers)
        self.deep_head = nn.Linear(current, output_dim)
        self.continuous_linear = nn.Linear(num_continuous, output_dim, bias=False) if num_continuous > 0 else None
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def _check_inputs(
        self, x_cat: torch.Tensor | None, x_cont: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x_cat is None or x_cat.shape[-1] != len(self.categories):
            raise ValueError(f"DeepFM requires x_cat with shape (batch, {len(self.categories)})")
        if self.num_continuous > 0:
            if x_cont is None or x_cont.shape[-1] != self.num_continuous:
                raise ValueError(f"x_cont must have shape (batch, {self.num_continuous})")
        return x_cat.long(), x_cont

    def forward(
        self,
        user_ids: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        x_cont: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del user_ids, item_ids
        x_cat, x_cont = self._check_inputs(x_cat, x_cont)
        linear_terms = [emb(x_cat[:, idx]) for idx, emb in enumerate(self.linear_embeddings)]
        linear = torch.stack(linear_terms, dim=0).sum(dim=0) + self.bias
        if self.continuous_linear is not None and x_cont is not None:
            linear = linear + self.continuous_linear(x_cont.float())
        fm_vectors = torch.stack([emb(x_cat[:, idx]) for idx, emb in enumerate(self.fm_embeddings)], dim=1)
        summed = fm_vectors.sum(dim=1)
        fm = 0.5 * (summed.pow(2) - fm_vectors.pow(2).sum(dim=1)).sum(dim=-1, keepdim=True)
        if self.output_dim > 1:
            fm = fm.expand(-1, self.output_dim)
        deep_input = fm_vectors.flatten(1)
        if x_cont is not None:
            deep_input = torch.cat([deep_input, x_cont.float()], dim=-1)
        return linear + fm + self.deep_head(self.deep(deep_input))
