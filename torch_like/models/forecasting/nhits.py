"""N-HiTS: Neural Hierarchical Interpolation for time series forecasting.

This framework-oriented implementation keeps the core N-HiTS mechanics:
multi-rate temporal pooling, residual backcast refinement, forecast knots, and
hierarchical interpolation to the final horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class NHITSConfig(ForecastingConfig):
    stack_layers: int = 2
    pooling_kernels: tuple[int, ...] = (1, 2, 4)
    downsample_frequencies: tuple[int, ...] = (1, 2, 4)
    interpolation_mode: str = "linear"
    use_norm: bool = False


class NHITSBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        pred_len: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pooling_kernel: int,
        downsample_frequency: int,
        num_knots: int,
        interpolation_mode: str,
    ):
        super().__init__()
        self.input_size = input_size
        self.pred_len = pred_len
        self.pooling_kernel = max(1, pooling_kernel)
        self.downsample_frequency = max(1, downsample_frequency)
        self.num_knots = max(1, num_knots)
        self.interpolation_mode = interpolation_mode
        self.pooled_size = max(1, math.ceil(input_size / self.downsample_frequency))

        layers: list[nn.Module] = []
        in_features = self.pooled_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_size, input_size)
        self.forecast_head = nn.Linear(hidden_size, self.num_knots)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        pooled = F.avg_pool1d(
            x.reshape(batch * channels, 1, length),
            kernel_size=self.pooling_kernel,
            stride=self.downsample_frequency,
            padding=self.pooling_kernel // 2,
            count_include_pad=False,
        )
        pooled = F.adaptive_avg_pool1d(pooled, self.pooled_size)
        return pooled.reshape(batch, channels, self.pooled_size)

    def _interpolate(self, knots: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = knots.shape
        flat = knots.reshape(batch * channels, 1, self.num_knots)
        if self.num_knots == self.pred_len:
            out = flat
        elif self.interpolation_mode == "nearest":
            out = F.interpolate(flat, size=self.pred_len, mode="nearest")
        else:
            out = F.interpolate(flat, size=self.pred_len, mode="linear", align_corners=False)
        return out.reshape(batch, channels, self.pred_len)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(self._pool(residual))
        backcast = self.backcast_head(features)
        forecast = self._interpolate(self.forecast_head(features))
        return backcast, forecast


class NHITS(ForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 1,
        d_ff: int = 512,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "relu",
        output_attention: bool = False,
        stack_layers: int = 2,
        pooling_kernels: tuple[int, ...] = (1, 2, 4),
        downsample_frequencies: tuple[int, ...] = (1, 2, 4),
        interpolation_mode: str = "linear",
        use_norm: bool = False,
    ):
        super().__init__()
        del dec_in, label_len, n_heads, d_layers, embed, freq, activation, output_attention
        if interpolation_mode not in {"linear", "nearest"}:
            raise ValueError("interpolation_mode must be 'linear' or 'nearest'")
        if not pooling_kernels or not downsample_frequencies:
            raise ValueError("pooling_kernels and downsample_frequencies must be non-empty")

        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = pred_len
        self.use_norm = use_norm
        self._config = NHITSConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=0,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=1,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=dropout,
            activation="relu",
            stack_layers=stack_layers,
            pooling_kernels=pooling_kernels,
            downsample_frequencies=downsample_frequencies,
            interpolation_mode=interpolation_mode,
            use_norm=use_norm,
        )

        blocks = []
        for idx in range(e_layers):
            pool = pooling_kernels[idx % len(pooling_kernels)]
            downsample = downsample_frequencies[idx % len(downsample_frequencies)]
            knots = max(1, math.ceil(pred_len / downsample))
            blocks.append(
                NHITSBlock(
                    input_size=seq_len,
                    pred_len=pred_len,
                    hidden_size=d_ff,
                    num_layers=stack_layers,
                    dropout=dropout,
                    pooling_kernel=pool,
                    downsample_frequency=downsample,
                    num_knots=knots,
                    interpolation_mode=interpolation_mode,
                )
            )
        self.blocks = nn.ModuleList(blocks)
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
            raise ValueError("NHITS expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        means = stdev = None
        x = x_enc
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        residual = x.transpose(1, 2)
        forecast = x.new_zeros(x.size(0), self.enc_in, self.pred_len)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        out = self.output_proj(forecast.transpose(1, 2))
        if self.use_norm and means is not None and stdev is not None:
            out = out * stdev[:, 0, : self.c_out].unsqueeze(1) + means[:, 0, : self.c_out].unsqueeze(1)
        return out
