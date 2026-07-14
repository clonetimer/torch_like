"""DAG-style dual-correlation forecasting with exogenous variables."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingBatch, ForecastingConfig, ForecastingModel


class DAG(ForecastingModel):
    """Framework-oriented dual-path exogenous forecasting model.

    The temporal path models historical/future exogenous context. The channel
    path gates target dynamics with historical target-exogenous correlations.
    """

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
        use_norm: bool = True,
    ):
        super().__init__()
        del dec_in, d_layers, embed, freq, output_attention
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_norm = use_norm
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

        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.target_proj = nn.Linear(c_out, d_model)
        self.hist_exog_proj = nn.LazyLinear(d_model)
        self.future_exog_proj = nn.LazyLinear(d_model)
        self.target_encoder = nn.GRU(
            d_model, d_model, num_layers=e_layers, batch_first=True, dropout=dropout if e_layers > 1 else 0.0
        )
        self.exog_encoder = nn.GRU(
            d_model, d_model, num_layers=e_layers, batch_first=True, dropout=dropout if e_layers > 1 else 0.0
        )
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.channel_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            act,
        )
        self.head = nn.Linear(d_model, c_out)

    def _semantic_inputs(self, batch: ForecastingBatch):
        target_enc = batch.x_target_enc
        if target_enc is None or target_enc.numel() == 0:
            target_enc = batch.x_enc[..., -self.c_out :]

        exog_enc = batch.x_exog_enc
        if exog_enc is None or exog_enc.numel() == 0:
            exog_enc = batch.x_enc[..., : max(batch.x_enc.size(-1) - self.c_out, 0)]

        exog_dec = batch.x_exog_dec
        if exog_dec is None or exog_dec.numel() == 0:
            if batch.x_dec is not None and batch.x_dec.numel() > 0:
                exog_dec = batch.x_dec[..., : exog_enc.size(-1)] if exog_enc.size(-1) > 0 else batch.x_dec[..., :0]
            else:
                exog_dec = exog_enc[:, -self.label_len :, :]

        if batch.x_mark_enc is not None and batch.x_mark_enc.numel() > 0:
            exog_enc = torch.cat([exog_enc, batch.x_mark_enc], dim=-1)
        if batch.x_mark_dec is not None and batch.x_mark_dec.numel() > 0:
            mark_future = batch.x_mark_dec[:, -self.pred_len :, :]
        else:
            mark_future = torch.zeros(
                batch.x_enc.size(0), self.pred_len, 0, device=batch.x_enc.device, dtype=batch.x_enc.dtype
            )

        exog_future = exog_dec[:, -self.pred_len :, :]
        if mark_future.numel() > 0:
            exog_future = torch.cat([exog_future, mark_future], dim=-1)
        return target_enc, exog_enc, exog_future

    def predict(self, batch: ForecastingBatch) -> torch.Tensor:
        target_enc, exog_enc, exog_future = self._semantic_inputs(batch)

        means = None
        stdev = None
        if self.use_norm:
            means = target_enc.mean(1, keepdim=True).detach()
            target_enc = target_enc - means
            stdev = torch.sqrt(torch.var(target_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            target_enc = target_enc / stdev

        target_tokens = self.target_proj(target_enc)
        target_encoded, _ = self.target_encoder(target_tokens)
        if exog_enc.size(-1) == 0:
            exog_enc = torch.zeros(
                target_enc.size(0), target_enc.size(1), 1, device=target_enc.device, dtype=target_enc.dtype
            )
        if exog_future.size(-1) == 0:
            exog_future = torch.zeros(
                target_enc.size(0), self.pred_len, 1, device=target_enc.device, dtype=target_enc.dtype
            )
        exog_encoded, _ = self.exog_encoder(self.hist_exog_proj(exog_enc))
        future_query = self.future_exog_proj(exog_future)
        temporal_context, _ = self.temporal_attn(future_query, exog_encoded, exog_encoded, need_weights=False)

        target_summary = target_encoded[:, -1:, :].expand(-1, self.pred_len, -1)
        exog_summary = exog_encoded.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        gate = self.channel_gate(torch.cat([target_summary, exog_summary], dim=-1))
        channel_context = target_summary * gate
        fused = self.fusion(torch.cat([future_query, temporal_context, channel_context], dim=-1))
        out = self.head(fused)

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
