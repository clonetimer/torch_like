"""Portable Non-stationary Transformer forecasting components.

NeurIPS 2022: https://arxiv.org/abs/2205.14415
Pinned official MIT source: https://github.com/thuml/Nonstationary_Transformers/tree/c4ec40675d11d50b3d9923657f408d0db6f90f56

The model uses sample-wise de-stationary attention factors learned from the
un-normalized observed window; it is not an official-checkpoint adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.models.forecasting.nonstationary_attention import _DeStationaryMultiHeadAttention


class _DeStationaryProjector(nn.Module):
    """Learn a sample factor from a detached raw window and window statistics.

    As in the reference projector, the convolution's *channels* correspond
    to observed time positions. Its length axis is the feature dimension,
    with circular padding across features, not across time.
    """

    def __init__(self, seq_len: int, channels: int, out_features: int, hidden: int):
        super().__init__()
        for name, value in (
            ("seq_len", seq_len),
            ("channels", channels),
            ("out_features", out_features),
            ("hidden", hidden),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.seq_len = seq_len
        self.channels = channels
        self.conv = nn.Conv1d(seq_len, 1, kernel_size=3, padding=1, padding_mode="circular")
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, raw: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 3 or raw.shape[1:] != (self.seq_len, self.channels) or raw.shape[0] == 0:
            raise ValueError(f"raw must have shape (nonempty_batch, {self.seq_len}, {self.channels})")
        if stats.shape != (raw.shape[0], 1, self.channels):
            raise ValueError(f"stats must have shape (batch, 1, {self.channels})")
        raw_flat = self.conv(raw.detach()).squeeze(1)
        return self.mlp(torch.cat((raw_flat, stats.detach().squeeze(1)), dim=-1))


class _TemporalEmbedding(nn.Module):
    """Portable value, four-feature calendar and sinusoidal time position map."""

    def __init__(self, channels: int, d_model: int, max_length: int, dropout: float):
        super().__init__()
        self.value_projection = nn.Linear(channels, d_model)
        self.marker_projection = nn.Linear(4, d_model, bias=False)
        positions = torch.arange(max_length, dtype=torch.float64)[:, None]
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float64) * (-math.log(10_000.0) / d_model))
        positional = torch.zeros(max_length, d_model, dtype=torch.float64)
        positional[:, 0::2] = torch.sin(positions * frequencies)
        positional[:, 1::2] = torch.cos(positions * frequencies[: d_model // 2])
        self.register_buffer("positional", positional, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
        value = self.value_projection(values)
        time = self.marker_projection(markers)
        position = self.positional[: values.shape[1], :].to(dtype=value.dtype)
        return self.dropout(value + time + position.unsqueeze(0))


def _feedforward(d_model: int, d_ff: int, dropout: float, activation: str) -> nn.Module:
    function = nn.GELU() if activation == "gelu" else nn.ReLU()
    return nn.Sequential(nn.Linear(d_model, d_ff), function, nn.Dropout(dropout), nn.Linear(d_ff, d_model))


class _DeStationaryEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.attention = _DeStationaryMultiHeadAttention(d_model, n_heads, dropout)
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = _feedforward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, tau: torch.Tensor, delta: torch.Tensor, keep_mask: torch.Tensor | None
    ) -> torch.Tensor:
        x = self.norm_attention(x + self.dropout(self.attention(x, x, x, tau, delta, keep_mask)))
        return self.norm_feedforward(x + self.dropout(self.feedforward(x)))


class _DeStationaryDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.self_attention = _DeStationaryMultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = _DeStationaryMultiHeadAttention(d_model, n_heads, dropout)
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = _feedforward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # A decoder's causal self attention uses tau but not encoder-key delta.
        attention = self.self_attention(x, x, x, tau, keep_mask=self_mask, causal=True)
        x = self.norm_self(x + self.dropout(attention))
        attention = self.cross_attention(x, memory, memory, tau, delta, keep_mask=cross_mask)
        x = self.norm_cross(x + self.dropout(attention))
        return self.norm_feedforward(x + self.dropout(self.feedforward(x)))


@dataclass
class NonstationaryTransformerConfig(ForecastingConfig):
    """Standard forecasting sizes plus hidden projector width and normalization epsilon."""

    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    factor_hidden: int = 128
    eps: float = 1e-5


class NonstationaryTransformer(ForecastingModel):
    """Non-stationary Transformer point forecaster, portable paper-aligned subset.

    Learned factors from detached raw windows restore temporal distinctions
    after observed-only per-instance stationarization. Decoder values from the
    caller are ignored (including future targets); only optional known time
    marks enter the decoder. The model returns a forecast-only Tensor even
    with ``output_attention=True``. Attention remains dense and quadratic.
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
        dropout: float = 0.05,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        factor_hidden: int = 128,
        eps: float = 1e-5,
    ):
        super().__init__()
        for name, value in (
            ("enc_in", enc_in),
            ("dec_in", dec_in),
            ("c_out", c_out),
            ("pred_len", pred_len),
            ("d_model", d_model),
            ("n_heads", n_heads),
            ("e_layers", e_layers),
            ("d_layers", d_layers),
            ("d_ff", d_ff),
            ("factor_hidden", factor_hidden),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seq_len, bool) or not isinstance(seq_len, Integral) or seq_len < 2:
            raise ValueError("seq_len must be an integer greater than or equal to 2")
        if isinstance(label_len, bool) or not isinstance(label_len, Integral) or not 0 <= label_len <= seq_len:
            raise ValueError("label_len must be an integer in [0, seq_len]")
        if enc_in != dec_in or enc_in != c_out:
            raise ValueError("NonstationaryTransformer requires enc_in == dec_in == c_out")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive real number")
        eps32 = torch.tensor(float(eps), dtype=torch.float32)
        if not torch.isfinite(eps32) or eps32 == 0:
            raise ValueError("eps must remain finite and nonzero in float32")
        if activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        if not isinstance(output_attention, bool):
            raise ValueError("output_attention must be a bool")
        if not isinstance(embed, str) or not isinstance(freq, str):
            raise ValueError("embed and freq must be strings")

        # Public dimensions accept numbers.Integral, including NumPy integer
        # scalars; internal PyTorch modules should see canonical Python ints.
        enc_in = int(enc_in)
        dec_in = int(dec_in)
        c_out = int(c_out)
        seq_len = int(seq_len)
        label_len = int(label_len)
        pred_len = int(pred_len)
        d_model = int(d_model)
        n_heads = int(n_heads)
        e_layers = int(e_layers)
        d_layers = int(d_layers)
        d_ff = int(d_ff)
        factor_hidden = int(factor_hidden)

        self.enc_in = int(enc_in)
        self.dec_in = int(dec_in)
        self.c_out = int(c_out)
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.eps = float(eps)
        decoder_length = label_len + pred_len

        self.tau_projector = _DeStationaryProjector(seq_len, enc_in, 1, factor_hidden)
        self.delta_projector = _DeStationaryProjector(seq_len, enc_in, seq_len, factor_hidden)
        self.encoder_embedding = _TemporalEmbedding(enc_in, d_model, seq_len, float(dropout))
        self.decoder_embedding = _TemporalEmbedding(dec_in, d_model, decoder_length, float(dropout))
        self.encoder = nn.ModuleList(
            _DeStationaryEncoderLayer(d_model, n_heads, d_ff, float(dropout), activation) for _ in range(e_layers)
        )
        self.decoder = nn.ModuleList(
            _DeStationaryDecoderLayer(d_model, n_heads, d_ff, float(dropout), activation) for _ in range(d_layers)
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, c_out)

    def _validate_observations(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input x_enc must have shape (batch, seq_len, enc_in)")
        if x.shape[0] == 0 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"input must have (nonempty_batch, {self.seq_len}, {self.enc_in}) shape")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("input dtype must be float32 or float64")
        parameter = self.output_projection.weight
        if x.dtype != parameter.dtype:
            raise TypeError("input dtype must match model dtype")
        if x.device != parameter.device:
            raise ValueError("input and model must be on the same device")
        if not torch.isfinite(x).all():
            raise ValueError("input values must all be finite")

    @staticmethod
    def _markers(
        value: torch.Tensor | None, batch: int, length: int, reference: torch.Tensor, name: str
    ) -> torch.Tensor:
        if value is None:
            return reference.new_zeros((batch, length, 4))
        if not isinstance(value, torch.Tensor) or value.shape != (batch, length, 4):
            raise ValueError(f"{name} marker must have (batch, {length}, 4) shape")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} marker must match input dtype/device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} marker must contain only finite values")
        return value

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_dec  # No caller-supplied future values may enter a prediction.
        self._validate_observations(x_enc)
        batch = x_enc.shape[0]
        total_decoder_length = self.label_len + self.pred_len
        encoder_markers = self._markers(x_mark_enc, batch, self.seq_len, x_enc, "encoder")
        decoder_markers = self._markers(x_mark_dec, batch, total_decoder_length, x_enc, "decoder")

        mean = x_enc.mean(dim=1, keepdim=True)
        scale = torch.sqrt(x_enc.var(dim=1, unbiased=False, keepdim=True) + self.eps)
        stationary = (x_enc - mean) / scale
        log_tau = self.tau_projector(x_enc, scale).squeeze(-1)
        tau = torch.exp(torch.clamp(log_tau, -15.0, 15.0))
        delta = self.delta_projector(x_enc, mean)

        known = stationary[:, -self.label_len :, :] if self.label_len else stationary[:, :0, :]
        future = stationary.new_zeros((batch, self.pred_len, self.dec_in))
        decoder_input = torch.cat((known, future), dim=1)
        memory = self.encoder_embedding(stationary, encoder_markers)
        for block in self.encoder:
            memory = block(memory, tau, delta, enc_self_mask)
        memory = self.encoder_norm(memory)

        hidden = self.decoder_embedding(decoder_input, decoder_markers)
        for block in self.decoder:
            hidden = block(hidden, memory, tau, delta, dec_self_mask, dec_enc_mask)
        normalized_forecast = self.output_projection(self.decoder_norm(hidden))[:, -self.pred_len :, :]
        return normalized_forecast * scale + mean
