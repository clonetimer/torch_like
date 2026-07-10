"""MLCNN-inspired multi-level convolutional forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class MultiLevelConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_sizes: list[int], dropout: float):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=kernel, padding=kernel // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel in kernel_sizes
            ]
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.stack([branch(x) for branch in self.branches], dim=0).mean(dim=0)
        return self.norm(x + mixed)


class MLCNN(ForecastingModel):
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
        e_layers: int = 3,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        kernel_sizes: list[int] | None = None,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, d_ff, embed, freq, activation, output_attention
        kernel_sizes = kernel_sizes or [3, 5, 7]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=1,
            e_layers=e_layers,
            d_layers=1,
            d_ff=128,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
        )

        self.input_proj = nn.Conv1d(enc_in, d_model, kernel_size=1)
        self.blocks = nn.ModuleList([MultiLevelConvBlock(d_model, kernel_sizes, dropout) for _ in range(e_layers)])
        self.temporal_head = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Linear(d_model, c_out)

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
            raise ValueError("MLCNN expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        x = self.input_proj(x_enc.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        x = self.temporal_head(x).transpose(1, 2)
        return self.output_proj(x)
