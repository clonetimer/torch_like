"""ConvTimeNet-style pure convolutional forecasting backbone.

This is a framework-oriented integration inspired by ConvTimeNet's modern
convolutional design for general time series analysis. It keeps the
torch_like forecasting contract and avoids depending on paper-specific
training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class ConvTimeNetConfig(ForecastingConfig):
    patch_len: int = 8
    stride: int = 4
    large_kernel: int = 13
    multiscale_kernels: tuple[int, ...] = (3, 5, 7)
    use_revin: bool = True


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std * self.weight + self.bias, mean, std

    def denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != mean.size(-1):
            return x
        return (x - self.bias) / self.weight.clamp_min(self.eps) * std + mean


class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, enc_in: int, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.local_gate = nn.Conv1d(enc_in, enc_in, kernel_size=3, padding=1, groups=enc_in)
        self.patch_proj = nn.Conv1d(enc_in, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x = x.transpose(1, 2)
        if x.size(-1) < self.patch_len:
            x = F.pad(x, (self.patch_len - x.size(-1), 0))
        gate = torch.sigmoid(self.local_gate(x))
        h = self.patch_proj(x * gate)
        return h.transpose(1, 2)


class LargeKernelConvMixer(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("large_kernel must be odd to preserve sequence length")
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.depthwise(h)
        h = F.gelu(self.pointwise(h))
        return self.dropout(h.transpose(1, 2))


class MultiScaleConvMixer(nn.Module):
    def __init__(self, d_model: int, kernels: Sequence[int], dropout: float):
        super().__init__()
        if not kernels:
            raise ValueError("multiscale_kernels must not be empty")
        for kernel in kernels:
            if kernel % 2 == 0:
                raise ValueError("multiscale_kernels must be odd to preserve sequence length")
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=d_model,
                )
                for kernel in kernels
            ]
        )
        self.fuse = nn.Conv1d(d_model * len(kernels), d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        branches = [branch(h) for branch in self.branches]
        h = F.gelu(self.fuse(torch.cat(branches, dim=1)))
        return self.dropout(h.transpose(1, 2))


class ConvTimeNetBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        large_kernel: int,
        multiscale_kernels: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        self.large_norm = nn.LayerNorm(d_model)
        self.large_mixer = LargeKernelConvMixer(d_model, large_kernel, dropout)
        self.multi_norm = nn.LayerNorm(d_model)
        self.multi_mixer = MultiScaleConvMixer(d_model, multiscale_kernels, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.large_mixer(self.large_norm(x))
        x = x + self.multi_mixer(self.multi_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class ConvTimeNet(ForecastingModel):
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
        patch_len: int = 8,
        stride: int = 4,
        large_kernel: int = 13,
        multiscale_kernels: Sequence[int] = (3, 5, 7),
        use_revin: bool = True,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, embed, freq, activation, output_attention
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_revin = use_revin
        kernel_tuple = tuple(multiscale_kernels)
        self._config = ConvTimeNetConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            patch_len=patch_len,
            stride=stride,
            large_kernel=large_kernel,
            multiscale_kernels=kernel_tuple,
            use_revin=use_revin,
        )

        self.revin = RevIN(enc_in) if use_revin else None
        self.patch_embedding = AdaptivePatchEmbedding(enc_in, d_model, patch_len, stride)
        self.blocks = nn.ModuleList(
            [
                ConvTimeNetBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    large_kernel=large_kernel,
                    multiscale_kernels=kernel_tuple,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * c_out),
        )

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
            raise ValueError("ConvTimeNet expects x_enc with shape (batch, seq_len, enc_in)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input features, got {x_enc.size(-1)}")

        mean = std = None
        if self.revin is not None:
            x_enc, mean, std = self.revin.normalize(x_enc)
        h = self.patch_embedding(x_enc)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        pooled = torch.cat([h.mean(dim=1), h[:, -1]], dim=-1)
        out = self.head(pooled).view(x_enc.size(0), self.pred_len, self.c_out)
        if self.revin is not None and mean is not None and std is not None:
            out = self.revin.denormalize(out, mean[:, :1, :], std[:, :1, :])
        return out
