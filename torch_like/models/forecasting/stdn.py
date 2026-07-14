"""STDN-inspired dynamic similarity forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class FlowGating(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.key_proj = nn.Linear(dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)
        key = self.key_proj(x.mean(dim=1, keepdim=True))
        hidden = torch.tanh(query + key)
        return torch.sigmoid(self.gate_proj(hidden))


class PeriodicallyShiftedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rolled = [
            x,
            torch.roll(x, shifts=1, dims=1),
            torch.roll(x, shifts=max(1, x.size(1) // 2), dims=1),
        ]
        attended = []
        for candidate in rolled:
            out, _ = self.attn(x, candidate, candidate, need_weights=False)
            attended.append(out)
        mixed = torch.stack(attended, dim=0).mean(dim=0)
        return self.norm(x + self.dropout(mixed))


class STDNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.flow_gating = FlowGating(d_model, d_model)
        self.shifted_attention = PeriodicallyShiftedAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.flow_gating(x)
        x = x * gate
        x = self.shifted_attention(x)
        return self.norm(x + self.ffn(x))


class STDN(ForecastingModel):
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
    ):
        super().__init__()
        del dec_in, d_layers, embed, freq, activation, output_attention
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
            n_heads=n_heads,
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
        self.input_proj = nn.Linear(enc_in, d_model)
        self.blocks = nn.ModuleList([STDNBlock(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
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
            raise ValueError("STDN expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        x = self.input_norm(x_enc)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.temporal_head(x.transpose(1, 2)).transpose(1, 2)
        return self.output_proj(x)
