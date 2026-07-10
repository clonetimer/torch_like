"""SCINet: sample convolution and interaction network for forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class SCIInteractor(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.phi = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.psi = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.rho = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.eta = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, even: torch.Tensor, odd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        even_t = even.transpose(1, 2)
        odd_t = odd.transpose(1, 2)
        d = odd_t * torch.exp(self.phi(even_t))
        c = even_t * torch.exp(self.psi(odd_t))
        even_out = c + self.eta(d)
        odd_out = d - self.rho(c)
        return even_out.transpose(1, 2), odd_out.transpose(1, 2)


class SCITree(nn.Module):
    def __init__(self, levels: int, channels: int, hidden_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.levels = levels
        self.interactor = SCIInteractor(channels, hidden_channels, kernel_size, dropout)
        self.even_branch = None
        self.odd_branch = None
        if levels > 0:
            self.even_branch = SCITree(levels - 1, channels, hidden_channels, kernel_size, dropout)
            self.odd_branch = SCITree(levels - 1, channels, hidden_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        even = x[:, ::2, :]
        odd = x[:, 1::2, :]
        min_len = min(even.size(1), odd.size(1))
        even = even[:, :min_len, :]
        odd = odd[:, :min_len, :]
        even, odd = self.interactor(even, odd)
        if self.levels == 0:
            return self._zip(even, odd)
        even = self.even_branch(even)
        odd = self.odd_branch(odd)
        return self._zip(even, odd)

    @staticmethod
    def _zip(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = even.shape
        out = torch.zeros(batch, steps * 2, channels, device=even.device, dtype=even.dtype)
        out[:, 0::2, :] = even
        out[:, 1::2, :] = odd
        return out


class SCINet(ForecastingModel):
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
        levels: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, embed, freq, activation, output_attention
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
        self.input_proj = nn.Linear(enc_in, d_model)
        self.blocks = nn.ModuleList([SCITree(levels, d_model, d_ff, kernel_size, dropout) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.time_proj = nn.Linear(seq_len, pred_len)
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
            raise ValueError("SCINet expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        x = self.input_proj(x_enc)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = self.norm(x[:, : residual.size(1), :] + residual[:, : x.size(1), :])
        x = self.time_proj(x.transpose(1, 2)).transpose(1, 2)
        return self.output_proj(x)
