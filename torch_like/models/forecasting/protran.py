"""ProTran-style latent probabilistic Transformer forecasting model."""

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


class ProTran(ProbabilisticForecastingModel):
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
        latent_dim: int | None = None,
        kl_weight: float = 0.01,
    ):
        super().__init__()
        del d_layers
        if distribution != "gaussian":
            raise ValueError("ProTran currently supports distribution='gaussian' only")
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.latent_dim = latent_dim or d_model
        self.kl_weight = kl_weight
        self.distribution = distribution
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
        self.dec_proj = nn.Linear(dec_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.posterior_mean = nn.Linear(d_model * 2, self.latent_dim)
        self.posterior_logvar = nn.Linear(d_model * 2, self.latent_dim)
        self.prior_mean = nn.Linear(d_model, self.latent_dim)
        self.prior_logvar = nn.Linear(d_model, self.latent_dim)
        self.latent_to_hidden = nn.Linear(self.latent_dim, d_model)
        self.hidden_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.mean_head = nn.Linear(d_model, c_out)
        self.scale_head = nn.Linear(d_model, c_out)

    def _encode_history(self, x_enc: torch.Tensor) -> torch.Tensor:
        if x_enc.dim() != 3:
            raise ValueError("ProTran expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} encoder features, got {x_enc.size(-1)}")
        hidden = self.enc_proj(x_enc)
        encoded = self.encoder(hidden)
        return encoded

    def _future_context(self, x_dec: torch.Tensor) -> torch.Tensor:
        if x_dec.dim() != 3:
            raise ValueError("ProTran expects x_dec with shape (batch, label_len + pred_len, num_features)")
        if x_dec.size(-1) != self.dec_in:
            raise ValueError(f"Expected {self.dec_in} decoder features, got {x_dec.size(-1)}")
        return self.dec_proj(x_dec[:, -self.pred_len :, :])

    def _posterior_params(self, history_state: torch.Tensor, future_context: torch.Tensor):
        history_summary = history_state.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        features = torch.cat([history_summary, future_context], dim=-1)
        return self.posterior_mean(features), self.posterior_logvar(features)

    def _prior_params(self, history_state: torch.Tensor):
        history_summary = history_state.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        return self.prior_mean(history_summary), self.prior_logvar(history_summary)

    def _fuse_hidden(
        self, history_state: torch.Tensor, future_context: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        history_summary = history_state.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        latent_hidden = self.latent_to_hidden(latent)
        return self.hidden_fusion(torch.cat([history_summary, future_context, latent_hidden], dim=-1))

    def _distribution_from_hidden(self, hidden: torch.Tensor):
        mean = self.mean_head(hidden)
        scale = F.softplus(self.scale_head(hidden)) + 1e-4
        return mean, scale

    def _sample_latent(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(0.5 * logvar)

    def _forward_distribution(self, batch: ProbabilisticForecastingBatch, use_posterior: bool):
        history_state = self._encode_history(batch.x_enc)
        future_context = self._future_context(batch.x_dec)
        prior_mean, prior_logvar = self._prior_params(history_state)
        if use_posterior and batch.y_true is not None:
            posterior_context = future_context + self.dec_proj(
                torch.cat(
                    [
                        batch.x_dec[:, -self.pred_len :, : -self.c_out],
                        batch.y_true,
                    ],
                    dim=-1,
                )
                if self.dec_in > self.c_out
                else batch.y_true
            )
            post_mean, post_logvar = self._posterior_params(history_state, posterior_context)
            latent = self._sample_latent(post_mean, post_logvar)
        else:
            post_mean, post_logvar = prior_mean, prior_logvar
            latent = self._sample_latent(prior_mean, prior_logvar)
        hidden = self._fuse_hidden(history_state, future_context, latent)
        mean, scale = self._distribution_from_hidden(hidden)
        return mean, scale, prior_mean, prior_logvar, post_mean, post_logvar

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
        batch = ProbabilisticForecastingBatch(x_enc=x_enc, x_mark_enc=None, x_dec=x_dec, x_mark_dec=None)
        mean, _, _, _, _, _ = self._forward_distribution(batch, use_posterior=False)
        return mean

    def predict_distribution(self, batch: ProbabilisticForecastingBatch):
        mean, scale, _, _, _, _ = self._forward_distribution(batch, use_posterior=False)
        return {"mean": mean, "scale": scale, "distribution": self.distribution}

    def sample(self, batch: ProbabilisticForecastingBatch, num_samples: int | None = None) -> torch.Tensor:
        count = num_samples or self.num_samples
        samples = []
        for _ in range(count):
            mean, scale, _, _, _, _ = self._forward_distribution(batch, use_posterior=False)
            samples.append(mean + torch.randn_like(mean) * scale)
        return torch.stack(samples, dim=0)

    def fit_step(self, batch: ProbabilisticForecastingBatch) -> torch.Tensor:
        if batch.y_true is None:
            raise ValueError("ProbabilisticForecastingBatch.y_true is required for fit_step")
        mean, scale, prior_mean, prior_logvar, post_mean, post_logvar = self._forward_distribution(
            batch, use_posterior=True
        )
        if batch.y_true.shape != mean.shape:
            raise ValueError(
                f"ProTran fit_step expects y_true with shape {tuple(mean.shape)}, got {tuple(batch.y_true.shape)}"
            )
        variance = scale.square()
        recon = 0.5 * torch.log(2 * torch.tensor(math.pi, device=mean.device, dtype=mean.dtype) * variance)
        recon = recon + 0.5 * (batch.y_true - mean).square() / variance
        kl = 0.5 * (
            prior_logvar
            - post_logvar
            + (torch.exp(post_logvar) + (post_mean - prior_mean).square()) / torch.exp(prior_logvar)
            - 1.0
        )
        return recon.mean() + self.kl_weight * kl.mean()
