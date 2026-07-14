"""TiDE: Time-series Dense Encoder for long-horizon forecasting.

This is a framework-oriented TiDE integration. It keeps the dense
encoder-decoder structure, residual target projection, and historical/future
covariate handling while following torch_like's ForecastingBatch contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingBatch, ForecastingConfig, ForecastingModel


@dataclass
class TiDEConfig(ForecastingConfig):
    hidden_dim: int = 128
    temporal_decoder_dim: int = 64
    use_norm: bool = True


class TiDEResidualBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + self.skip(x))


class LazyTiDEResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )
        self.skip = nn.LazyLinear(out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + self.skip(x))


class TiDE(ForecastingModel):
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
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        hidden_dim: int | None = None,
        temporal_decoder_dim: int = 64,
        use_norm: bool = True,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, embed, freq, output_attention
        hidden = hidden_dim if hidden_dim is not None else d_model
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self._config = TiDEConfig(
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
            activation=activation,
            hidden_dim=hidden,
            temporal_decoder_dim=temporal_decoder_dim,
            use_norm=use_norm,
        )

        self.encoder_in = LazyTiDEResidualBlock(hidden_dim=d_ff, out_dim=hidden, dropout=dropout, activation=activation)
        self.encoder = nn.Sequential(
            *[TiDEResidualBlock(hidden, d_ff, hidden, dropout=dropout, activation=activation) for _ in range(e_layers)]
        )
        self.decoder = nn.Sequential(
            *[
                TiDEResidualBlock(hidden, d_ff, hidden, dropout=dropout, activation=activation)
                for _ in range(max(1, e_layers // 2))
            ]
        )
        self.horizon_context = nn.Linear(hidden, pred_len * temporal_decoder_dim)
        self.temporal_decoder = TiDEResidualBlock(
            temporal_decoder_dim + c_out,
            d_ff,
            c_out,
            dropout=dropout,
            activation=activation,
        )
        self.residual_head = nn.Linear(seq_len, pred_len)

    def _semantic_inputs(self, batch: ForecastingBatch):
        target_enc = batch.x_target_enc
        if target_enc is None or target_enc.numel() == 0:
            target_enc = batch.x_enc[..., -self.c_out :]

        exog_enc = batch.x_exog_enc
        if exog_enc is None or exog_enc.numel() == 0:
            exog_enc = batch.x_enc[..., : max(batch.x_enc.size(-1) - self.c_out, 0)]
        if batch.x_mark_enc is not None and batch.x_mark_enc.numel() > 0:
            exog_enc = torch.cat([exog_enc, batch.x_mark_enc], dim=-1)

        exog_dec = batch.x_exog_dec
        if exog_dec is None or exog_dec.numel() == 0:
            if batch.x_dec is not None and batch.x_dec.numel() > 0 and exog_enc.size(-1) > 0:
                exog_dec = batch.x_dec[..., : min(batch.x_dec.size(-1), exog_enc.size(-1))]
            else:
                exog_dec = torch.zeros(
                    batch.x_enc.size(0),
                    self.label_len + self.pred_len,
                    0,
                    device=batch.x_enc.device,
                    dtype=batch.x_enc.dtype,
                )
        future_exog = exog_dec[:, -self.pred_len :, :]
        if batch.x_mark_dec is not None and batch.x_mark_dec.numel() > 0:
            future_exog = torch.cat([future_exog, batch.x_mark_dec[:, -self.pred_len :, :]], dim=-1)
        return target_enc, exog_enc, future_exog

    def predict(self, batch: ForecastingBatch) -> torch.Tensor:
        target_enc, exog_enc, future_exog = self._semantic_inputs(batch)

        means = stdev = None
        if self.use_norm:
            means = target_enc.mean(dim=1, keepdim=True).detach()
            target_enc = target_enc - means
            stdev = torch.sqrt(torch.var(target_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            target_enc = target_enc / stdev

        residual = self.residual_head(target_enc.transpose(1, 2)).transpose(1, 2)
        flat_parts = [target_enc.flatten(start_dim=1)]
        if exog_enc.numel() > 0:
            flat_parts.append(exog_enc.flatten(start_dim=1))
        if future_exog.numel() > 0:
            flat_parts.append(future_exog.flatten(start_dim=1))
        h = torch.cat(flat_parts, dim=-1)
        h = self.encoder(self.encoder_in(h))
        h = self.decoder(h)
        horizon_context = self.horizon_context(h).view(target_enc.size(0), self.pred_len, -1)
        decoder_in = torch.cat([horizon_context, residual], dim=-1)
        out = self.temporal_decoder(decoder_in) + residual

        if self.use_norm and means is not None and stdev is not None:
            out = out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return out

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
        del enc_self_mask, dec_self_mask, dec_enc_mask
        if x_dec is None:
            x_dec = torch.empty(0, device=x_enc.device, dtype=x_enc.dtype)
        return self.predict(ForecastingBatch(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec))
