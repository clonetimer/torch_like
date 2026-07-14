"""FreTS-style frequency-domain MLP forecaster.

This framework-oriented implementation follows the FreTS idea that frequency
domain MLPs can model temporal patterns and channel interactions efficiently.
It combines temporal-frequency and channel-frequency transformations under the
standard torch_like forecasting contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class FreTSConfig(ForecastingConfig):
    sparsity_threshold: float = 0.01
    use_channel_frequency: bool = True
    use_revin: bool = True


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


class ComplexFrequencyMLP(nn.Module):
    def __init__(self, num_freq: int, sparsity_threshold: float, dropout: float):
        super().__init__()
        self.real_weight = nn.Parameter(torch.empty(num_freq, num_freq))
        self.imag_weight = nn.Parameter(torch.empty(num_freq, num_freq))
        self.real_bias = nn.Parameter(torch.zeros(num_freq))
        self.imag_bias = nn.Parameter(torch.zeros(num_freq))
        self.sparsity_threshold = sparsity_threshold
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.real_weight)
        nn.init.xavier_uniform_(self.imag_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., F), complex tensor.
        real = torch.einsum("...f,fg->...g", x.real, self.real_weight) - torch.einsum(
            "...f,fg->...g", x.imag, self.imag_weight
        )
        imag = torch.einsum("...f,fg->...g", x.real, self.imag_weight) + torch.einsum(
            "...f,fg->...g", x.imag, self.real_weight
        )
        real = F.gelu(real + self.real_bias)
        imag = F.gelu(imag + self.imag_bias)
        real = F.softshrink(real, lambd=self.sparsity_threshold)
        imag = F.softshrink(imag, lambd=self.sparsity_threshold)
        real = self.dropout(real)
        imag = self.dropout(imag)
        return torch.complex(real, imag)


class FreTS(ForecastingModel):
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
        sparsity_threshold: float = 0.01,
        use_channel_frequency: bool = True,
        use_revin: bool = True,
    ):
        super().__init__()
        del dec_in, d_model, n_heads, e_layers, d_layers, d_ff, embed, freq, activation, output_attention
        if seq_len <= 1:
            raise ValueError("seq_len must be greater than 1 for FreTS")
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_channel_frequency = use_channel_frequency
        self.use_revin = use_revin
        self._config = FreTSConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            dropout=dropout,
            sparsity_threshold=sparsity_threshold,
            use_channel_frequency=use_channel_frequency,
            use_revin=use_revin,
        )

        self.revin = RevIN(enc_in) if use_revin else None
        self.temporal_mlp = ComplexFrequencyMLP(seq_len // 2 + 1, sparsity_threshold, dropout)
        self.channel_mlp = (
            ComplexFrequencyMLP(enc_in // 2 + 1, sparsity_threshold, dropout) if use_channel_frequency else None
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * enc_in, enc_in),
            nn.Sigmoid(),
        )
        self.horizon_head = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _temporal_frequency_mix(self, x: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.rfft(x, dim=1).transpose(1, 2)
        mixed = self.temporal_mlp(freq).transpose(1, 2)
        return torch.fft.irfft(mixed, n=self.seq_len, dim=1)

    def _channel_frequency_mix(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_mlp is None:
            return torch.zeros_like(x)
        freq = torch.fft.rfft(x, dim=-1)
        mixed = self.channel_mlp(freq)
        return torch.fft.irfft(mixed, n=self.enc_in, dim=-1)

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
            raise ValueError("FreTS expects x_enc with shape (batch, seq_len, enc_in)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input features, got {x_enc.size(-1)}")

        mean = std = None
        if self.revin is not None:
            x_enc, mean, std = self.revin.normalize(x_enc)

        temporal = self._temporal_frequency_mix(x_enc)
        channel = self._channel_frequency_mix(x_enc)
        summary = torch.cat([temporal.mean(dim=1), channel.mean(dim=1)], dim=-1)
        gate = self.fusion_gate(summary).unsqueeze(1)
        mixed = gate * temporal + (1.0 - gate) * channel
        out = self.horizon_head(mixed.transpose(1, 2)).transpose(1, 2)
        out = self.output_proj(out)

        if self.revin is not None and mean is not None and std is not None:
            out = self.revin.denormalize(out, mean[:, :1, :], std[:, :1, :])
        return out
