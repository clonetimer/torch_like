"""SOFTS-style series-core fusion forecaster.

This framework-oriented implementation follows the SOFTS direction: each
variable is encoded as a series token, a compact core state aggregates global
cross-variate information, and the core is redistributed back to each series
for efficient multivariate forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class SOFTSConfig(ForecastingConfig):
    core_dim: int = 32
    use_revin: bool = True
    pooling: str = "attention"


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


class STARBlock(nn.Module):
    """Series-core aggregate-redistribute block."""

    def __init__(self, d_model: int, d_ff: int, core_dim: int, dropout: float, pooling: str):
        super().__init__()
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be 'attention' or 'mean'")
        self.pooling = pooling
        self.series_norm = nn.LayerNorm(d_model)
        self.series_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.core_proj = nn.Linear(d_model, core_dim)
        self.core_score = nn.Linear(core_dim, 1)
        self.redistribute = nn.Sequential(
            nn.Linear(d_model + core_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D), where C is variable count.
        series = x + self.dropout(self.series_mlp(self.series_norm(x)))
        core_candidates = torch.tanh(self.core_proj(series))
        if self.pooling == "mean":
            core = core_candidates.mean(dim=1, keepdim=True)
        else:
            weights = torch.softmax(self.core_score(core_candidates), dim=1)
            core = (weights * core_candidates).sum(dim=1, keepdim=True)
        core = core.expand(-1, series.size(1), -1)
        fused = torch.cat([series, core], dim=-1)
        return self.out_norm(series + self.dropout(self.redistribute(fused)))


class SOFTS(ForecastingModel):
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
        core_dim: int = 32,
        use_revin: bool = True,
        pooling: str = "attention",
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, embed, freq, activation, output_attention
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_revin = use_revin
        self._config = SOFTSConfig(
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
            core_dim=core_dim,
            use_revin=use_revin,
            pooling=pooling,
        )

        self.revin = RevIN(enc_in) if use_revin else None
        self.series_embedding = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                STARBlock(d_model=d_model, d_ff=d_ff, core_dim=core_dim, dropout=dropout, pooling=pooling)
                for _ in range(e_layers)
            ]
        )
        self.horizon_head = nn.Linear(d_model, pred_len)
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
            raise ValueError("SOFTS expects x_enc with shape (batch, seq_len, enc_in)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input features, got {x_enc.size(-1)}")

        mean = std = None
        if self.revin is not None:
            x_enc, mean, std = self.revin.normalize(x_enc)

        x = x_enc.transpose(1, 2)
        h = self.series_embedding(x)
        for block in self.blocks:
            h = block(h)
        out = self.horizon_head(h).transpose(1, 2)
        out = self.output_proj(out)

        if self.revin is not None and mean is not None and std is not None:
            out = self.revin.denormalize(out, mean[:, :1, :], std[:, :1, :])
        return out
