"""Temporal Fusion Transformer-style multi-horizon forecaster.

This is a framework-oriented TFT integration. It keeps the core TFT design
ideas: variable-selection-style feature gating, local recurrent processing,
static context injection, interpretable temporal attention, and gated residual
fusion. It follows torch_like's ForecastingBatch contract instead of requiring
the original dataset-specific preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingBatch, ForecastingConfig, ForecastingModel


@dataclass
class TFTConfig(ForecastingConfig):
    static_dim: int = 0
    quantiles: tuple[float, ...] | None = None
    use_norm: bool = True


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        activation: str = "elu",
        context_dim: int | None = None,
    ):
        super().__init__()
        if activation == "gelu":
            act: nn.Module = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.ELU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim is not None else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Sigmoid())
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = act

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.fc1(x)
        if context is not None and self.context is not None:
            while context.dim() < hidden.dim():
                context = context.unsqueeze(1)
            hidden = hidden + self.context(context)
        hidden = self.activation(hidden)
        hidden = self.dropout(self.fc2(hidden))
        gated = hidden * self.gate(hidden)
        return self.norm(gated + self.skip(x))


class VariableSelection(nn.Module):
    """Feature-wise soft selection over continuous inputs."""

    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.value_proj = nn.Linear(1, d_model)
        self.weight_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.weight_proj(x), dim=-1)
        values = self.value_proj(x.unsqueeze(-1))
        selected = torch.sum(values * weights.unsqueeze(-1), dim=-2)
        return self.dropout(selected), weights


class TFT(ForecastingModel):
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
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "elu",
        output_attention: bool = False,
        static_dim: int = 0,
        quantiles: tuple[float, ...] | None = None,
        use_norm: bool = True,
    ):
        super().__init__()
        del d_layers, embed, freq
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.static_dim = static_dim
        self.output_attention = output_attention
        self.quantiles = quantiles
        self.use_norm = use_norm
        self._last_attention: torch.Tensor | None = None
        self._config = TFTConfig(
            enc_in=enc_in,
            dec_in=dec_in,
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
            activation=activation,
            output_attention=output_attention,
            static_dim=static_dim,
            quantiles=quantiles,
            use_norm=use_norm,
        )

        self.history_selector = VariableSelection(enc_in, d_model, dropout)
        self.future_selector = VariableSelection(dec_in, d_model, dropout)
        self.static_context = nn.Linear(static_dim, d_model) if static_dim > 0 else None
        self.static_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout, activation)
        self.history_lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=e_layers,
            batch_first=True,
            dropout=dropout if e_layers > 1 else 0.0,
        )
        self.future_lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=e_layers,
            batch_first=True,
            dropout=dropout if e_layers > 1 else 0.0,
        )
        self.post_lstm_gate = GatedResidualNetwork(d_model, d_ff, d_model, dropout, activation)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post_attention_gate = GatedResidualNetwork(d_model, d_ff, d_model, dropout, activation)
        self.positionwise = GatedResidualNetwork(d_model, d_ff, d_model, dropout, activation)
        out_dim = c_out * len(quantiles) if quantiles else c_out
        self.head = nn.Linear(d_model, out_dim)

    def _build_inputs(self, batch: ForecastingBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        history = batch.x_enc
        future = batch.x_dec[:, -self.pred_len :, :] if batch.x_dec is not None and batch.x_dec.numel() > 0 else None
        if future is None:
            future = torch.zeros(
                history.size(0), self.pred_len, self.dec_in, device=history.device, dtype=history.dtype
            )
        if future.size(-1) != self.dec_in:
            if future.size(-1) > self.dec_in:
                future = future[..., : self.dec_in]
            else:
                pad = torch.zeros(
                    *future.shape[:-1], self.dec_in - future.size(-1), device=future.device, dtype=future.dtype
                )
                future = torch.cat([future, pad], dim=-1)

        static = None
        if self.static_dim > 0:
            static = torch.zeros(history.size(0), self.static_dim, device=history.device, dtype=history.dtype)
        return history, future, static

    def _semantic_inputs(self, batch: ForecastingBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target_enc = batch.x_target_enc
        if target_enc is None or target_enc.numel() == 0:
            target_enc = batch.x_enc[..., -self.c_out :]

        exog_enc = batch.x_exog_enc
        if exog_enc is None or exog_enc.numel() == 0:
            exog_enc = batch.x_enc[..., : max(batch.x_enc.size(-1) - self.c_out, 0)]
        if batch.x_mark_enc is not None and batch.x_mark_enc.numel() > 0:
            exog_enc = torch.cat([exog_enc, batch.x_mark_enc], dim=-1)
        history = torch.cat([exog_enc, target_enc], dim=-1) if exog_enc.numel() > 0 else target_enc

        future_exog = batch.x_exog_dec
        if future_exog is None or future_exog.numel() == 0:
            if batch.x_dec is not None and batch.x_dec.numel() > 0:
                future_exog = batch.x_dec[:, -self.pred_len :, :]
            else:
                future_exog = torch.zeros(
                    batch.x_enc.size(0),
                    self.pred_len,
                    self.dec_in,
                    device=batch.x_enc.device,
                    dtype=batch.x_enc.dtype,
                )
        else:
            future_exog = future_exog[:, -self.pred_len :, :]
        if batch.x_mark_dec is not None and batch.x_mark_dec.numel() > 0:
            future_exog = torch.cat([future_exog, batch.x_mark_dec[:, -self.pred_len :, :]], dim=-1)
        if future_exog.size(-1) != self.dec_in:
            if future_exog.size(-1) > self.dec_in:
                future_exog = future_exog[..., : self.dec_in]
            else:
                pad = torch.zeros(
                    *future_exog.shape[:-1],
                    self.dec_in - future_exog.size(-1),
                    device=future_exog.device,
                    dtype=future_exog.dtype,
                )
                future_exog = torch.cat([future_exog, pad], dim=-1)
        return history, future_exog, None

    def predict(self, batch: ForecastingBatch) -> torch.Tensor:
        if batch.x_target_enc is not None or batch.x_exog_enc is not None or batch.x_exog_dec is not None:
            history, future, static = self._semantic_inputs(batch)
            target_for_norm = batch.x_target_enc if batch.x_target_enc is not None else history[..., -self.c_out :]
        else:
            history, future, static = self._build_inputs(batch)
            target_for_norm = history[..., -self.c_out :]

        means = stdev = None
        if self.use_norm:
            means = target_for_norm.mean(dim=1, keepdim=True).detach()
            centered = target_for_norm - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            history = history.clone()
            history[..., -self.c_out :] = centered / stdev

        if history.size(-1) != self.enc_in:
            raise ValueError(f"TFT expected history feature dim {self.enc_in}, got {history.size(-1)}")
        hist_tokens, _ = self.history_selector(history)
        future_tokens, _ = self.future_selector(future)

        context = None
        if self.static_context is not None and static is not None:
            context = self.static_grn(self.static_context(static))
            hist_tokens = hist_tokens + context.unsqueeze(1)
            future_tokens = future_tokens + context.unsqueeze(1)

        hist_encoded, state = self.history_lstm(hist_tokens)
        future_encoded, _ = self.future_lstm(future_tokens, state)
        temporal = self.post_lstm_gate(torch.cat([hist_encoded, future_encoded], dim=1))
        query = temporal[:, -self.pred_len :, :]
        attended, attn_weights = self.attention(
            query, temporal, temporal, need_weights=True, average_attn_weights=False
        )
        self._last_attention = attn_weights.detach()
        decoded = self.post_attention_gate(attended + query)
        decoded = self.positionwise(decoded, context=context)
        out = self.head(decoded)
        if self.quantiles:
            out = out.view(out.size(0), out.size(1), self.c_out, len(self.quantiles))
            out = out[..., len(self.quantiles) // 2]

        if self.use_norm and means is not None and stdev is not None:
            out = out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return out

    def attention_weights(self) -> torch.Tensor | None:
        """Return the latest decoder attention weights if predict/forward ran."""
        return self._last_attention

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
