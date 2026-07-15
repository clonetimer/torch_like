"""Portable temporal backbone adapters for RUL regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.rul import RULConfig, RULModel


def _validate_common(enc_in: int, seq_len: int, d_model: int) -> None:
    if min(enc_in, seq_len, d_model) <= 0:
        raise ValueError("enc_in, seq_len, and d_model must be positive")


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, d_ff: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.mixer = nn.Sequential(
            nn.Conv1d(channels, d_ff, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(self.depthwise(x)))


class TCNRUL(RULModel):
    """Dilated residual temporal convolution baseline for scalar RUL."""

    def __init__(
        self,
        enc_in: int = 2,
        seq_len: int = 2048,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ):
        super().__init__()
        del n_heads
        _validate_common(enc_in, seq_len, d_model)
        if e_layers <= 0 or d_ff <= 0 or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("e_layers/d_ff must be positive and kernel_size must be a positive odd integer")
        self.enc_in = enc_in
        self.seq_len = seq_len
        self._config = RULConfig(enc_in, seq_len, d_model, 1, e_layers, d_ff, dropout)
        self.input_projection = nn.Conv1d(enc_in, d_model, 1)
        self.blocks = nn.Sequential(
            *[_TemporalBlock(d_model, kernel_size, 2**index, d_ff, dropout) for index in range(e_layers)]
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        if x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"TCNRUL expects x with shape (batch, {self.seq_len}, {self.enc_in})")
        return self.head(self.blocks(self.input_projection(x.transpose(1, 2)))).squeeze(-1)


class PatchTSTRUL(RULModel):
    """Channel-independent PatchTST encoder with a scalar RUL head."""

    def __init__(
        self,
        enc_in: int = 2,
        seq_len: int = 2048,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        patch_len: int = 32,
        patch_stride: int = 16,
    ):
        super().__init__()
        _validate_common(enc_in, seq_len, d_model)
        if min(n_heads, e_layers, d_ff, patch_len, patch_stride) <= 0:
            raise ValueError("Transformer and patch dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if patch_len > seq_len:
            raise ValueError("patch_len cannot exceed seq_len")
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.num_patches = (seq_len - patch_len) // patch_stride + 1
        self._config = RULConfig(enc_in, seq_len, d_model, n_heads, e_layers, d_ff, dropout)
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, e_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        if x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"PatchTSTRUL expects x with shape (batch, {self.seq_len}, {self.enc_in})")
        patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        batch, channels, patch_count, patch_len = patches.shape
        encoded = self.patch_projection(patches.reshape(batch * channels, patch_count, patch_len))
        encoded = self.encoder(encoded + self.position)
        encoded = self.norm(encoded.mean(dim=1)).reshape(batch, channels, -1).mean(dim=1)
        return self.head(encoded).squeeze(-1)


class TransformerRUL(RULModel):
    """Encoder-only Transformer baseline for multichannel RUL regression."""

    def __init__(
        self,
        enc_in: int = 2,
        seq_len: int = 2048,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        _validate_common(enc_in, seq_len, d_model)
        if min(n_heads, e_layers, d_ff) <= 0 or d_model % n_heads:
            raise ValueError("n_heads/e_layers/d_ff must be positive and d_model divisible by n_heads")
        self.enc_in = enc_in
        self.seq_len = seq_len
        self._config = RULConfig(enc_in, seq_len, d_model, n_heads, e_layers, d_ff, dropout)
        self.input_projection = nn.Linear(enc_in, d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, e_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        if x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"TransformerRUL expects x with shape (batch, {self.seq_len}, {self.enc_in})")
        encoded = self.encoder(self.input_projection(x) + self.position)
        return self.head(self.norm(encoded.mean(dim=1))).squeeze(-1)


__all__ = ["TCNRUL", "PatchTSTRUL", "TransformerRUL"]
