"""TimesNet-inspired temporal 2D variation modeling for forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


def _select_top_periods(x: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    freq = torch.fft.rfft(x, dim=1)
    amplitude = freq.abs().mean(dim=(0, 2))
    amplitude[0] = 0
    top_values, top_indices = torch.topk(amplitude, k=min(top_k, amplitude.size(0)))
    periods = torch.div(x.size(1), top_indices.clamp_min(1), rounding_mode="floor").clamp_min(1)
    weights = torch.softmax(top_values, dim=0)
    return periods, weights


class InceptionBlock1D(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, num_kernels: int):
        super().__init__()
        kernels = [2 * i + 1 for i in range(num_kernels)]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden_channels, kernel_size=(1, kernel), padding=(0, kernel // 2)),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, channels, kernel_size=(1, kernel), padding=(0, kernel // 2)),
                )
                for kernel in kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([branch(x) for branch in self.branches], dim=0)
        return stacked.mean(dim=0)


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, top_k: int, num_kernels: int, dropout: float):
        super().__init__()
        self.top_k = top_k
        self.conv = InceptionBlock1D(d_model, d_ff, num_kernels)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, total_len, channels = x.shape
        periods, weights = _select_top_periods(x, self.top_k)
        outputs = []
        for idx, period in enumerate(periods.tolist()):
            pad_len = (period - total_len % period) % period
            x_pad = (
                x
                if pad_len == 0
                else torch.cat([x, torch.zeros(batch, pad_len, channels, device=x.device, dtype=x.dtype)], dim=1)
            )
            steps = x_pad.size(1) // period
            x_2d = x_pad.view(batch, steps, period, channels).permute(0, 3, 1, 2)
            y = self.conv(x_2d)
            y = y.permute(0, 2, 3, 1).reshape(batch, -1, channels)[:, :total_len, :]
            outputs.append(y * weights[idx])
        mixed = torch.stack(outputs, dim=0).sum(dim=0)
        return self.norm(x + self.dropout(mixed))


class TimesNet(ForecastingModel):
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
        top_k: int = 3,
        num_kernels: int = 4,
    ):
        super().__init__()
        del dec_in, label_len, n_heads, d_layers, embed, freq, activation, output_attention
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
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
            activation="gelu",
            output_attention=False,
        )
        self.value_proj = nn.Linear(enc_in, d_model)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(d_model=d_model, d_ff=d_ff, top_k=top_k, num_kernels=num_kernels, dropout=dropout)
                for _ in range(e_layers)
            ]
        )
        self.pred_proj = nn.Linear(seq_len, pred_len)
        self.feature_proj = nn.Linear(d_model, c_out)

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
            raise ValueError("TimesNet expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        x = self.value_proj(x_enc)
        for block in self.blocks:
            x = block(x)
        x = self.pred_proj(x.transpose(1, 2)).transpose(1, 2)
        x = self.feature_proj(x)
        return x
