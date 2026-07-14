"""ModernTCN-style large-kernel convolutional time-series forecaster."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class ModernTCNConfig(ForecastingConfig):
    patch_size: int = 8
    patch_stride: int = 4
    kernel_size: int = 13
    num_blocks: int = 3
    use_revin: bool = True


class ModernTCNBlock(nn.Module):
    def __init__(self, channels: int, num_variables: int, d_ff: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.large_kernel = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.feature_mixer = nn.Sequential(
            nn.Conv1d(channels, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, channels, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.variable_mixer = nn.Sequential(
            nn.Conv1d(num_variables, num_variables, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, variables, hidden, patches)
        batch_size, variables, hidden, patches = x.shape
        temporal = x.reshape(batch_size * variables, hidden, patches)
        temporal = self.feature_mixer(self.norm(self.large_kernel(temporal)))
        x = x + temporal.reshape(batch_size, variables, hidden, patches)
        mixed = self.variable_mixer(x.permute(0, 3, 1, 2).reshape(batch_size * patches, variables, hidden))
        return x + mixed.reshape(batch_size, patches, variables, hidden).permute(0, 2, 3, 1)


class ModernTCN(ForecastingModel):
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
        patch_size: int = 8,
        patch_stride: int = 4,
        kernel_size: int = 13,
        num_blocks: int | None = None,
        use_revin: bool = True,
    ):
        super().__init__()
        del n_heads, d_layers, embed, freq, activation, output_attention
        if min(enc_in, c_out, seq_len, pred_len, d_model, patch_size, patch_stride) <= 0:
            raise ValueError("model dimensions, patch_size, and patch_stride must be positive")
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.use_revin = use_revin
        block_count = num_blocks or e_layers
        self._config = ModernTCNConfig(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            patch_size=patch_size,
            patch_stride=patch_stride,
            kernel_size=kernel_size,
            num_blocks=block_count,
            use_revin=use_revin,
        )
        self.patch_embedding = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_stride)
        self.blocks = nn.ModuleList(
            [ModernTCNBlock(d_model, enc_in, d_ff, kernel_size, dropout) for _ in range(block_count)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)
        self.channel_projection = nn.Identity() if enc_in == c_out else nn.Linear(enc_in, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        del x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"ModernTCN expects x_enc with shape (batch, {self.seq_len}, {self.enc_in})")
        mean = x_enc.mean(dim=1, keepdim=True)
        std = x_enc.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x = (x_enc - mean) / std if self.use_revin else x_enc
        x = x.transpose(1, 2).reshape(x.size(0) * self.enc_in, 1, self.seq_len)
        if self.seq_len < self.patch_size:
            x = F.pad(x, (self.patch_size - self.seq_len, 0))
        x = self.patch_embedding(x)
        x = x.reshape(x_enc.size(0), self.enc_in, -1, x.size(-1))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.mean(dim=-1))
        out = self.head(x).transpose(1, 2)
        if self.use_revin:
            out = out * std + mean
        return self.channel_projection(out)
