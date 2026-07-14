"""TabTransformer-style contextual categorical embedding model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.tabular.common import TabularConfig, TabularModel


@dataclass
class TabTransformerConfig(TabularConfig):
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256


class TabTransformer(TabularModel):
    """Transformer over categorical feature tokens with continuous MLP fusion."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        dropout: float = 0.1,
        categories: tuple[int, ...] = (),
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
    ):
        super().__init__()
        if min(input_dim, output_dim, d_model, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("TabTransformer dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not categories:
            raise ValueError("TabTransformer requires at least one categorical feature")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.categories = tuple(categories)
        self._config = TabTransformerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            categories=self.categories,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
        )
        self.cat_embeddings = nn.ModuleList([nn.Embedding(size, d_model) for size in self.categories])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        fused_dim = len(self.categories) * d_model + input_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, output_dim),
        )

    def categorical_tokens(self, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.shape != (x_cat.size(0), len(self.categories)):
            raise ValueError(f"x_cat must have shape (batch, {len(self.categories)})")
        return torch.stack([emb(x_cat[:, idx].long()) for idx, emb in enumerate(self.cat_embeddings)], dim=1)

    def contextual_embeddings(self, x_cat: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.categorical_tokens(x_cat))

    def forward_features(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        if x_cont.dim() != 2 or x_cont.size(-1) != self.input_dim:
            raise ValueError(f"TabTransformer expects x_cont with shape (batch, {self.input_dim})")
        if x_cat is None:
            raise ValueError("TabTransformer requires x_cat")
        contextual = self.contextual_embeddings(x_cat).flatten(start_dim=1)
        return torch.cat([contextual, x_cont.float()], dim=-1)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.forward_features(x_cont, x_cat))
