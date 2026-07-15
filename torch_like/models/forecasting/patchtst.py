"""PatchTST: patch-based channel-independent Transformer for forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        return self.proj(patches)


class PatchTST(ForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        patch_len: int = 16,
        stride: int | None = None,
    ):
        super().__init__()
        del dec_in, d_layers, embed, freq, output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.patch_len = patch_len
        self.stride = stride or patch_len
        if self.stride != self.patch_len:
            raise ValueError("Current PatchTST implementation requires stride == patch_len")
        if seq_len < patch_len:
            raise ValueError("seq_len must be >= patch_len")
        self.num_patches = math.floor((seq_len - patch_len) / self.stride) + 1
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
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

        self.patch_embed = PatchEmbedding(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * d_model, pred_len),
        )
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
            raise ValueError("PatchTST expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        x = x_enc.transpose(1, 2)  # (B, C, L)
        x = self.patch_embed(x)  # (B, C, P, D)
        x = x + self.pos_embed
        batch, channels, patches, dim = x.shape
        x = self.dropout(x).reshape(batch * channels, patches, dim)
        x = self.encoder(x)
        x = x.reshape(batch, channels, patches, dim)
        x = self.head(x).transpose(1, 2)  # (B, pred_len, C)
        x = self.output_proj(x)
        return x
