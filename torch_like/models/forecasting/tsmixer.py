"""TSMixer: all-MLP baseline for multivariate time series forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class TimeMixing(nn.Module):
    def __init__(self, seq_len: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.net = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return residual + self.net(x.transpose(1, 2)).transpose(1, 2)


class FeatureMixing(nn.Module):
    def __init__(self, enc_in: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(enc_in)
        self.net = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, enc_in),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        return residual + self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, enc_in: int, time_hidden_dim: int, feature_hidden_dim: int, dropout: float):
        super().__init__()
        self.time_mixing = TimeMixing(seq_len, time_hidden_dim, dropout)
        self.feature_mixing = FeatureMixing(enc_in, feature_hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x


class TSMixer(ForecastingModel):
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
        time_hidden_dim: int | None = None,
        feature_hidden_dim: int | None = None,
    ):
        super().__init__()
        del dec_in, d_model, n_heads, d_layers, embed, freq, activation, output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.time_hidden_dim = time_hidden_dim or d_ff
        self.feature_hidden_dim = feature_hidden_dim or d_ff
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=64,
            n_heads=4,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
        )

        self.input_norm = nn.LayerNorm(enc_in)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(seq_len, enc_in, self.time_hidden_dim, self.feature_hidden_dim, dropout)
                for _ in range(e_layers)
            ]
        )
        self.temporal_head = nn.Linear(seq_len, pred_len)
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
            raise ValueError("TSMixer expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        x = self.input_norm(x_enc)
        for block in self.blocks:
            x = block(x)
        x = self.temporal_head(x.transpose(1, 2)).transpose(1, 2)
        return self.output_proj(x)
