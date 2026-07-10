"""Crossformer-inspired cross-dimension forecasting model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class DSWEmbedding(nn.Module):
    def __init__(self, seg_len: int, d_model: int):
        super().__init__()
        self.seg_len = seg_len
        self.proj = nn.Linear(seg_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.seg_len, step=self.seg_len)
        return self.proj(patches)


class TwoStageAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.cross_time = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.cross_dim = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, variates, segments, dim = x.shape
        x_time = x.reshape(batch * variates, segments, dim)
        x_time = self.cross_time(x_time)
        x = x_time.reshape(batch, variates, segments, dim)
        x_dim = x.permute(0, 2, 1, 3).reshape(batch * segments, variates, dim)
        x_dim = self.cross_dim(x_dim)
        return x_dim.reshape(batch, segments, variates, dim).permute(0, 2, 1, 3)


class Crossformer(ForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        seg_len: int = 6,
    ):
        super().__init__()
        del dec_in, label_len, d_layers, embed, freq, output_attention
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.seg_len = seg_len
        if seq_len < seg_len:
            raise ValueError("seq_len must be >= seg_len")
        self.num_segments = math.floor(seq_len / seg_len)
        if self.num_segments < 1:
            raise ValueError("Crossformer requires at least one segment")
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=0,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation=activation,
            output_attention=False,
        )

        self.embedding = DSWEmbedding(seg_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, enc_in, self.num_segments, d_model))
        self.blocks = nn.ModuleList(
            [TwoStageAttentionBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.num_segments * d_model, pred_len)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        if x_enc.dim() != 3:
            raise ValueError("Crossformer expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        usable_len = self.num_segments * self.seg_len
        x = x_enc[:, -usable_len:, :].transpose(1, 2)
        x = self.embedding(x) + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        batch, variates, segments, dim = x.shape
        x = x.reshape(batch, variates, segments * dim)
        x = self.head(x).transpose(1, 2)
        x = self.output_proj(x)
        return x
