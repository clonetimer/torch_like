"""FT-Transformer-style tabular model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.tabular.common import TabularConfig, TabularModel


@dataclass
class FTTransformerConfig(TabularConfig):
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256


class FTTransformer(TabularModel):
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
            raise ValueError("FTTransformer dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.categories = tuple(categories)
        self._config = FTTransformerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            categories=self.categories,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cont_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.cont_bias = nn.Parameter(torch.zeros(input_dim, d_model))
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def tokenize(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        if x_cont.dim() != 2 or x_cont.size(-1) != self.input_dim:
            raise ValueError(f"FTTransformer expects x_cont with shape (batch, {self.input_dim})")
        cont_tokens = x_cont.float().unsqueeze(-1) * self.cont_weight.unsqueeze(0) + self.cont_bias.unsqueeze(0)
        tokens = [self.cls_token.expand(x_cont.size(0), -1, -1), cont_tokens]
        if self.categories:
            if x_cat is None or x_cat.shape != (x_cont.size(0), len(self.categories)):
                raise ValueError(f"x_cat must have shape (batch, {len(self.categories)})")
            cat_tokens = torch.stack([emb(x_cat[:, idx].long()) for idx, emb in enumerate(self.cat_embeddings)], dim=1)
            tokens.append(cat_tokens)
        return torch.cat(tokens, dim=1)

    def forward_features(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.tokenize(x_cont, x_cat)
        encoded = self.encoder(tokens)
        return encoded[:, 0]

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.forward_features(x_cont, x_cat))
