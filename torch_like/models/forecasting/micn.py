"""MICN-inspired multiscale local-global convolution forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.nn.decomposition import SeriesDecomp


class MICNBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, kernel_sizes: list[int], dropout: float):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_ff, kernel_size=kernel, padding=kernel // 2),
                    nn.GELU(),
                    nn.Conv1d(d_ff, d_model, kernel_size=1),
                )
                for kernel in kernel_sizes
            ]
        )
        self.global_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        local = torch.stack([branch(x_t) for branch in self.branches], dim=0).mean(dim=0)
        global_feat = self.global_mixer(x_t)
        mixed = (local + global_feat).transpose(1, 2)
        return self.norm(x + self.dropout(mixed))


class MICN(ForecastingModel):
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
        moving_avg: int = 3,
        conv_kernel_sizes: list[int] | None = None,
    ):
        super().__init__()
        del dec_in, label_len, n_heads, d_layers, embed, freq, activation, output_attention
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=0,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=1,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
        )
        kernel_sizes = conv_kernel_sizes or [3, 5, 7]
        self.input_proj = nn.Linear(enc_in, d_model)
        self.decomp = SeriesDecomp(moving_avg)
        self.blocks = nn.ModuleList([MICNBlock(d_model, d_ff, kernel_sizes, dropout) for _ in range(e_layers)])
        self.head = nn.Linear(seq_len * d_model, pred_len * c_out)

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
            raise ValueError("MICN expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        seasonal, trend = self.decomp(x_enc)
        x = self.input_proj(seasonal + trend)
        for block in self.blocks:
            x = block(x)
        x = self.head(x.reshape(x.size(0), -1))
        return x.view(x.size(0), self.pred_len, self.c_out)
