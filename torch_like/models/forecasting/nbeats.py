"""N-BEATS: deep residual basis expansion for univariate or multivariate forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class NBEATSBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, theta_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_size, theta_size)
        self.backcast_proj = nn.Linear(theta_size, input_size)
        self.forecast_proj = nn.Linear(theta_size, theta_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        theta = self.theta(features)
        backcast = self.backcast_proj(theta)
        forecast = self.forecast_proj(theta)
        return backcast, forecast


class NBEATS(ForecastingModel):
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
        e_layers: int = 4,
        d_layers: int = 1,
        d_ff: int = 512,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "relu",
        output_attention: bool = False,
        stack_layers: int = 4,
    ):
        super().__init__()
        del dec_in, label_len, n_heads, d_layers, embed, freq, activation, output_attention
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.stack_layers = stack_layers
        self.hidden_size = d_model
        self.theta_size = pred_len
        self._config = ForecastingConfig(
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
            embed="fixed",
            freq="h",
            activation="relu",
            output_attention=False,
        )

        input_size = seq_len
        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(
                    input_size=input_size,
                    hidden_size=d_ff,
                    theta_size=pred_len,
                    num_layers=stack_layers,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
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
            raise ValueError("NBEATS expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        residual = x_enc.transpose(1, 2)
        forecast = x_enc.new_zeros(x_enc.size(0), self.enc_in, self.pred_len)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        forecast = forecast.transpose(1, 2)
        return self.output_proj(forecast)
