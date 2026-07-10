"""FEDformer: frequency enhanced decomposed Transformer for forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.nn.decomposition import SeriesDecomp


class FourierCrossMixing(nn.Module):
    def __init__(self, d_model: int, modes: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        self.weight_real = nn.Parameter(torch.randn(modes, d_model) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(modes, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        freq = torch.fft.rfft(x, dim=1)
        top_modes = min(self.modes, freq.size(1))
        mixed = freq.clone()
        mixed[:, :top_modes, :] = freq[:, :top_modes, :] * torch.complex(
            self.weight_real[:top_modes].unsqueeze(0),
            self.weight_imag[:top_modes].unsqueeze(0),
        )
        out = torch.fft.irfft(mixed, n=seq_len, dim=1)
        return self.dropout(out)


class FEDformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, modes: int, moving_avg: int, dropout: float, activation: str):
        super().__init__()
        self.mix = FourierCrossMixing(d_model=d_model, modes=modes, dropout=dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seasonal, trend1 = self.decomp1(x + self.mix(x))
        seasonal = self.norm1(seasonal)
        seasonal2, trend2 = self.decomp2(seasonal + self.ffn(seasonal))
        seasonal2 = self.norm2(seasonal2)
        return seasonal2, trend1 + trend2


class FEDformer(ForecastingModel):
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
        moving_avg: int = 25,
        modes: int = 32,
    ):
        super().__init__()
        del dec_in, d_layers, embed, freq, output_attention, n_heads
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
            d_ff=d_ff,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation=activation,
            output_attention=False,
        )
        self.input_proj = nn.Linear(enc_in, d_model)
        self.blocks = nn.ModuleList(
            [
                FEDformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    modes=modes,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )
        self.seasonal_head = nn.Linear(seq_len, pred_len)
        self.trend_head = nn.Linear(seq_len, pred_len)
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
            raise ValueError("FEDformer expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        hidden = self.input_proj(x_enc)
        trend_acc = torch.zeros_like(hidden)
        for block in self.blocks:
            hidden, trend = block(hidden)
            trend_acc = trend_acc + trend
        seasonal = self.seasonal_head(hidden.transpose(1, 2)).transpose(1, 2)
        trend = self.trend_head(trend_acc.transpose(1, 2)).transpose(1, 2)
        return self.output_proj(seasonal + trend)
