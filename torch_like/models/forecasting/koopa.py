"""Koopa-style Koopman predictor for non-stationary time series forecasting.

This implementation is framework-oriented: it preserves the main Koopa idea
of splitting dynamics with a Fourier filter and advancing latent states with
learned Koopman operators, while keeping the standard torch_like forecasting
contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class KoopaConfig(ForecastingConfig):
    modes: int = 8
    koopman_dim: int = 64
    num_predictors: int = 2
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


class FourierDynamicsSplitter(nn.Module):
    def __init__(self, modes: int):
        super().__init__()
        if modes <= 0:
            raise ValueError("modes must be positive")
        self.modes = modes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C). Keep the strongest global frequencies as invariant dynamics.
        freq = torch.fft.rfft(x, dim=1)
        num_freq = freq.size(1)
        keep = min(self.modes, num_freq)
        energy = freq.abs().mean(dim=(0, 2))
        indices = torch.topk(energy, k=keep).indices
        mask = torch.zeros(num_freq, device=x.device, dtype=torch.bool)
        mask[indices] = True
        invariant_freq = torch.zeros_like(freq)
        invariant_freq[:, mask, :] = freq[:, mask, :]
        invariant = torch.fft.irfft(invariant_freq, n=x.size(1), dim=1)
        variant = x - invariant
        return invariant, variant


class KoopmanPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int, pred_len: int, dropout: float):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=d_model, batch_first=True)
        self.operator = nn.Parameter(torch.empty(d_model, d_model))
        self.local_update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        nn.init.orthogonal_(self.operator)

    def _stable_operator(self) -> torch.Tensor:
        scale = self.operator.norm(dim=1, keepdim=True).clamp_min(1.0)
        return 0.98 * self.operator / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, state = self.encoder(x)
        context = encoded.mean(dim=1)
        z = state[-1]
        operator = self._stable_operator()
        outputs = []
        for _ in range(self.pred_len):
            global_step = z @ operator.T
            local_step = self.local_update(torch.cat([z, context], dim=-1))
            z = global_step + local_step
            outputs.append(self.decoder(z))
        return torch.stack(outputs, dim=1)


class KoopaBlock(nn.Module):
    def __init__(self, enc_in: int, d_model: int, pred_len: int, dropout: float, modes: int):
        super().__init__()
        self.splitter = FourierDynamicsSplitter(modes=modes)
        self.invariant_predictor = KoopmanPredictor(enc_in, enc_in, d_model, pred_len, dropout)
        self.variant_predictor = KoopmanPredictor(enc_in, enc_in, d_model, pred_len, dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * enc_in, enc_in),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        invariant, variant = self.splitter(x)
        invariant_pred = self.invariant_predictor(invariant)
        variant_pred = self.variant_predictor(variant)
        summary = torch.cat([invariant[:, -1, :], variant[:, -1, :]], dim=-1)
        gate = self.gate(summary).unsqueeze(1)
        return gate * invariant_pred + (1.0 - gate) * variant_pred


class Koopa(ForecastingModel):
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
        modes: int = 8,
        koopman_dim: int | None = None,
        num_predictors: int | None = None,
        use_revin: bool = True,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, d_ff, embed, freq, activation, output_attention
        hidden_dim = koopman_dim if koopman_dim is not None else d_model
        blocks = num_predictors if num_predictors is not None else e_layers
        if blocks <= 0:
            raise ValueError("num_predictors/e_layers must be positive")
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_revin = use_revin
        self._config = KoopaConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            e_layers=blocks,
            dropout=dropout,
            modes=modes,
            koopman_dim=hidden_dim,
            num_predictors=blocks,
            use_revin=use_revin,
        )

        self.revin = RevIN(enc_in) if use_revin else None
        self.blocks = nn.ModuleList(
            [
                KoopaBlock(enc_in=enc_in, d_model=hidden_dim, pred_len=pred_len, dropout=dropout, modes=modes)
                for _ in range(blocks)
            ]
        )
        self.block_weights = nn.Parameter(torch.zeros(blocks))
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
            raise ValueError("Koopa expects x_enc with shape (batch, seq_len, enc_in)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input features, got {x_enc.size(-1)}")

        mean = std = None
        if self.revin is not None:
            x_enc, mean, std = self.revin.normalize(x_enc)

        forecasts = torch.stack([block(x_enc) for block in self.blocks], dim=0)
        weights = torch.softmax(self.block_weights, dim=0).view(-1, 1, 1, 1)
        out = (weights * forecasts).sum(dim=0)
        out = self.output_proj(out)

        if self.revin is not None and mean is not None and std is not None:
            out = self.revin.denormalize(out, mean[:, :1, :], std[:, :1, :])
        return out
