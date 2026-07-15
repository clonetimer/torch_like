"""PPM-style parametric-prior probabilistic forecasting model."""

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


class PPM(ProbabilisticForecastingModel):
    """Framework-oriented Parametric Prior Mapping forecaster.

    The implementation keeps the paper's main interface idea: encode historical
    context, map it into horizon-wise Gaussian prior parameters, then produce a
    full predictive distribution in one forward pass.
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
        distribution: str = "gaussian",
        num_samples: int = 100,
        prior_rank: int | None = None,
        mse_weight: float = 0.1,
    ):
        super().__init__()
        del d_layers
        if distribution != "gaussian":
            raise ValueError("PPM currently supports distribution='gaussian' only")
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.distribution = distribution
        self.prior_rank = prior_rank or d_model
        self.mse_weight = mse_weight
        self._config = ProbabilisticForecastingConfig(
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
            distribution=distribution,
            num_samples=num_samples,
        )

        self.enc_proj = nn.Linear(enc_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers, enable_nested_tensor=False)
        self.dec_proj = nn.Linear(dec_in, d_model)
        self.horizon_embed = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.prior_mapper = nn.Sequential(
            nn.Linear(d_model * 3, self.prior_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.prior_rank, d_model),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(d_model, c_out)
        self.scale_head = nn.Linear(d_model, c_out)

    def _encode_context(self, x_enc: torch.Tensor) -> torch.Tensor:
        if x_enc.dim() != 3:
            raise ValueError("PPM expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} encoder features, got {x_enc.size(-1)}")
        encoded = self.encoder(self.enc_proj(x_enc))
        return encoded

    def _decoder_context(self, x_dec: torch.Tensor) -> torch.Tensor:
        if x_dec.dim() != 3:
            raise ValueError("PPM expects x_dec with shape (batch, label_len + pred_len, num_features)")
        if x_dec.size(-1) != self.dec_in:
            raise ValueError(f"Expected {self.dec_in} decoder features, got {x_dec.size(-1)}")
        return self.dec_proj(x_dec[:, -self.pred_len :, :])

    def _distribution_params(self, x_enc: torch.Tensor, x_dec: torch.Tensor):
        encoded = self._encode_context(x_enc)
        context_mean = encoded.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        context_last = encoded[:, -1:, :].expand(-1, self.pred_len, -1)
        dec_context = self._decoder_context(x_dec)
        horizon = self.horizon_embed.expand(x_enc.size(0), -1, -1)
        prior_hidden = self.prior_mapper(torch.cat([context_mean, context_last, dec_context + horizon], dim=-1))
        mean = self.mean_head(prior_hidden)
        scale = F.softplus(self.scale_head(prior_hidden)) + 1e-4
        return mean, scale, prior_hidden

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
        mean, _, _ = self._distribution_params(x_enc, x_dec)
        return mean

    def predict_distribution(self, batch: ProbabilisticForecastingBatch):
        mean, scale, prior_hidden = self._distribution_params(batch.x_enc, batch.x_dec)
        return {
            "mean": mean,
            "scale": scale,
            "prior_hidden": prior_hidden,
            "distribution": self.distribution,
        }

    def sample(self, batch: ProbabilisticForecastingBatch, num_samples: int | None = None) -> torch.Tensor:
        dist = self.predict_distribution(batch)
        mean = dist["mean"]
        scale = dist["scale"]
        count = num_samples or self.num_samples
        noise = torch.randn(count, *mean.shape, device=mean.device, dtype=mean.dtype)
        return mean.unsqueeze(0) + noise * scale.unsqueeze(0)

    def fit_step(self, batch: ProbabilisticForecastingBatch) -> torch.Tensor:
        if batch.y_true is None:
            raise ValueError("ProbabilisticForecastingBatch.y_true is required for fit_step")
        dist = self.predict_distribution(batch)
        mean = dist["mean"]
        scale = dist["scale"]
        if batch.y_true.shape != mean.shape:
            raise ValueError(
                f"PPM fit_step expects y_true with shape {tuple(mean.shape)}, got {tuple(batch.y_true.shape)}"
            )
        variance = scale.square()
        log_term = torch.log(2 * torch.tensor(math.pi, device=mean.device, dtype=mean.dtype) * variance)
        nll = 0.5 * log_term + 0.5 * (batch.y_true - mean).square() / variance
        return nll.mean() + self.mse_weight * F.mse_loss(mean, batch.y_true)
