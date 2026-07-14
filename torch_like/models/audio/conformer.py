"""Conformer-style convolution-augmented Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.audio.common import AudioConfig, AudioModel


@dataclass
class ConformerConfig(AudioConfig):
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    conv_kernel_size: int = 15


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).transpose(1, 2)
        x = F.glu(self.pointwise_in(x), dim=1)
        x = F.silu(self.batch_norm(self.depthwise(x)))
        x = self.pointwise_out(x).transpose(1, 2)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, conv_kernel_size: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(
            attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class Conformer(AudioModel):
    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 1000,
        d_model: int = 256,
        dropout: float = 0.1,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        conv_kernel_size: int = 15,
    ):
        super().__init__()
        if min(input_dim, output_dim, d_model, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("Conformer dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._config = ConformerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            conv_kernel_size=conv_kernel_size,
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [ConformerBlock(d_model, n_heads, d_ff, conv_kernel_size, dropout) for _ in range(n_layers)]
        )
        self.output_projection = nn.Linear(d_model, output_dim)

    def _padding_mask(self, lengths: torch.Tensor | None, max_len: int, device: torch.device) -> torch.Tensor | None:
        if lengths is None:
            return None
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        return positions >= lengths.to(device=device).unsqueeze(1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(f"Conformer expects x with shape (batch, time, {self.input_dim})")
        mask = self._padding_mask(lengths, x.size(1), x.device)
        out = self.input_projection(x.float())
        for layer in self.layers:
            out = layer(out, key_padding_mask=mask)
        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.output_projection(self.encode(x, lengths=lengths))
