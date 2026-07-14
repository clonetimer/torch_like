"""OmniAnomaly-style variational recurrent anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


@dataclass
class OmniAnomalyConfig(DetectionConfig):
    latent_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 1
    beta: float = 0.1


class OmniAnomaly(DetectionModel):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        dropout: float = 0.1,
        threshold: float = 0.5,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 1,
        beta: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.beta = beta
        self._config = OmniAnomalyConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            threshold=threshold,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            beta=beta,
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.encoder = nn.GRU(
            d_model,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.GRU(
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def latent_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(f"OmniAnomaly expects x with shape (batch, {self.seq_len}, {self.input_dim})")
        hidden, _ = self.encoder(self.input_projection(x))
        return self.mu(hidden), self.logvar(hidden).clamp(min=-8.0, max=8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.latent_distribution(x)
        z = self.reparameterize(mu, logvar)
        hidden, _ = self.decoder(z)
        reconstruction = self.output_projection(hidden)
        return reconstruction, mu, logvar

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        reconstruction, _, _ = self.reconstruct(x)
        return reconstruction[:, -1]

    def fit_step(self, batch: DetectionBatch) -> torch.Tensor:
        reconstruction, mu, logvar = self.reconstruct(batch.x)
        reconstruction_loss = F.mse_loss(reconstruction, batch.x)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + self.beta * kl

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        reconstruction, mu, logvar = self.reconstruct(batch.x)
        score = (reconstruction[:, -1] - batch.x[:, -1]).pow(2)
        uncertainty = torch.exp(logvar[:, -1]).mean(dim=-1, keepdim=True)
        score = score + self.beta * uncertainty
        if reduction == "none":
            return score
        if reduction == "sum":
            return score.sum(dim=-1)
        return score.mean(dim=-1)
