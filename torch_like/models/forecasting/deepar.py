"""DeepAR-style probabilistic forecasting model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import (
    ProbabilisticForecastingBatch,
    ProbabilisticForecastingConfig,
    ProbabilisticForecastingModel,
)


class DeepAR(ProbabilisticForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        distribution: str = "gaussian",
        num_samples: int = 100,
    ):
        super().__init__()
        del n_heads, d_layers, d_ff
        if distribution != "gaussian":
            raise ValueError("DeepAR currently supports distribution='gaussian' only")

        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.distribution = distribution
        self._config = ProbabilisticForecastingConfig(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=8,
            e_layers=e_layers,
            d_layers=1,
            d_ff=128,
            dropout=dropout,
            distribution=distribution,
            num_samples=num_samples,
        )

        self.encoder_input = nn.Linear(enc_in, d_model)
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=e_layers,
            dropout=dropout if e_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=dec_in,
            hidden_size=d_model,
            num_layers=e_layers,
            dropout=dropout if e_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mean_head = nn.Linear(d_model, c_out)
        self.scale_head = nn.Linear(d_model, c_out)

    def _encode(self, x_enc: torch.Tensor):
        if x_enc.dim() != 3:
            raise ValueError("DeepAR expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} encoder features, got {x_enc.size(-1)}")
        enc_input = self.encoder_input(x_enc)
        _, state = self.encoder(enc_input)
        return state

    def _decode_params(self, x_dec: torch.Tensor, state):
        if x_dec.dim() != 3:
            raise ValueError("DeepAR expects x_dec with shape (batch, label_len + pred_len, num_features)")
        if x_dec.size(-1) != self.dec_in:
            raise ValueError(f"Expected {self.dec_in} decoder features, got {x_dec.size(-1)}")
        dec_out, _ = self.decoder(x_dec, state)
        future_hidden = dec_out[:, -self.pred_len :, :]
        mean = self.mean_head(future_hidden)
        scale = F.softplus(self.scale_head(future_hidden)) + 1e-4
        return mean, scale

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor | None,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        state = self._encode(x_enc)
        mean, _ = self._decode_params(x_dec, state)
        return mean

    def predict_distribution(self, batch: ProbabilisticForecastingBatch):
        state = self._encode(batch.x_enc)
        mean, scale = self._decode_params(batch.x_dec, state)
        return {"mean": mean, "scale": scale, "distribution": self.distribution}

    def sample(self, batch: ProbabilisticForecastingBatch, num_samples: int | None = None) -> torch.Tensor:
        dist = self.predict_distribution(batch)
        mean = dist["mean"]
        scale = dist["scale"]
        sample_count = num_samples or self.num_samples
        noise = torch.randn(
            sample_count,
            mean.size(0),
            mean.size(1),
            mean.size(2),
            device=mean.device,
            dtype=mean.dtype,
        )
        return mean.unsqueeze(0) + noise * scale.unsqueeze(0)

    def fit_step(self, batch: ProbabilisticForecastingBatch) -> torch.Tensor:
        if batch.y_true is None:
            raise ValueError("ProbabilisticForecastingBatch.y_true is required for fit_step")
        dist = self.predict_distribution(batch)
        mean = dist["mean"]
        scale = dist["scale"]
        target = batch.y_true
        if target.shape != mean.shape:
            raise ValueError(
                f"DeepAR fit_step expects y_true with shape {tuple(mean.shape)}, got {tuple(target.shape)}"
            )
        variance = scale.square()
        log_term = torch.log(2 * torch.tensor(math.pi, device=mean.device, dtype=mean.dtype) * variance)
        nll = 0.5 * log_term + 0.5 * (target - mean).square() / variance
        return nll.mean()
