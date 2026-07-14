"""SAINT-style row/column attention model for tabular data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.tabular.common import TabularConfig, TabularModel


@dataclass
class SAINTConfig(TabularConfig):
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    use_row_attention: bool = True


class _SAINTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, use_row_attention: bool):
        super().__init__()
        self.use_row_attention = use_row_attention
        self.column_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.column_norm = nn.LayerNorm(d_model)
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.row_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        column_out, _ = self.column_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.column_norm(tokens + self.dropout(column_out))
        if self.use_row_attention:
            row_tokens = tokens.transpose(0, 1)
            row_out, _ = self.row_attn(row_tokens, row_tokens, row_tokens, need_weights=False)
            tokens = self.row_norm((row_tokens + self.dropout(row_out)).transpose(0, 1))
        tokens = self.ffn_norm(tokens + self.dropout(self.ffn(tokens)))
        return tokens


class SAINT(TabularModel):
    """Self-attention and inter-sample attention model for tabular prediction."""

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
        use_row_attention: bool = True,
    ):
        super().__init__()
        if min(input_dim, output_dim, d_model, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("SAINT dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.categories = tuple(categories)
        self._config = SAINTConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            categories=self.categories,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            use_row_attention=use_row_attention,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cont_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.cont_bias = nn.Parameter(torch.zeros(input_dim, d_model))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(size, d_model) for size in self.categories])
        self.blocks = nn.ModuleList(
            [_SAINTBlock(d_model, n_heads, d_ff, dropout, use_row_attention) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def tokenize(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        if x_cont.dim() != 2 or x_cont.size(-1) != self.input_dim:
            raise ValueError(f"SAINT expects x_cont with shape (batch, {self.input_dim})")
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
        for block in self.blocks:
            tokens = block(tokens)
        return tokens[:, 0]

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.forward_features(x_cont, x_cat))
