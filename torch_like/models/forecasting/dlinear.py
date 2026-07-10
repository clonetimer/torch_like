"""DLinear: decomposition-linear baseline for long-term forecasting.

This implementation follows the LTSF-Linear/DLinear idea: decompose the input
series into seasonal and trend components, then use linear temporal projection
heads to forecast each component. It is intentionally simple because DLinear's
framework value is as a strong, transparent baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.nn.decomposition import SeriesDecomp


@dataclass
class DLinearConfig(ForecastingConfig):
    moving_avg: int = 25
    individual: bool = True
    use_norm: bool = False


class DLinear(ForecastingModel):
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
        e_layers: int = 1,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.0,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        moving_avg: int = 25,
        individual: bool = True,
        use_norm: bool = False,
    ):
        super().__init__()
        del dec_in, d_model, n_heads, e_layers, d_layers, d_ff, embed, freq, activation, output_attention
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.individual = individual
        self.use_norm = use_norm
        self._config = DLinearConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            dropout=dropout,
            moving_avg=moving_avg,
            individual=individual,
            use_norm=use_norm,
        )

        self.decomposition = SeriesDecomp(moving_avg)
        if individual:
            self.seasonal_linear = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
            self.trend_linear = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
        else:
            self.seasonal_linear = nn.Linear(seq_len, pred_len)
            self.trend_linear = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _project(self, x: torch.Tensor, heads: nn.ModuleList | nn.Linear) -> torch.Tensor:
        # x: (B, C, L) -> (B, pred_len, C)
        if self.individual:
            outputs = [heads[i](x[:, i, :]) for i in range(self.enc_in)]
            return torch.stack(outputs, dim=-1)
        return heads(x).transpose(1, 2)

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
            raise ValueError("DLinear expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        means = stdev = None
        x = x_enc
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        out = self._project(seasonal, self.seasonal_linear) + self._project(trend, self.trend_linear)
        out = self.output_proj(self.dropout(out))

        if self.use_norm and means is not None and stdev is not None:
            out = out * stdev[:, 0, : self.c_out].unsqueeze(1) + means[:, 0, : self.c_out].unsqueeze(1)
        return out
