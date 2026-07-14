"""USAD dual-autoencoder anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


@dataclass
class USADConfig(DetectionConfig):
    latent_dim: int = 64
    hidden_dim: int = 128
    alpha: float = 0.5


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class USAD(DetectionModel):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        dropout: float = 0.1,
        threshold: float = 0.5,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        alpha: float = 0.5,
    ):
        del d_model
        super().__init__()
        if min(input_dim, seq_len, latent_dim, hidden_dim) <= 0:
            raise ValueError("input_dim, seq_len, latent_dim, and hidden_dim must be positive")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.flat_dim = input_dim * seq_len
        self.alpha = alpha
        self._config = USADConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=hidden_dim,
            dropout=dropout,
            threshold=threshold,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            alpha=alpha,
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder_1 = MLPDecoder(latent_dim, hidden_dim, self.flat_dim, dropout)
        self.decoder_2 = MLPDecoder(latent_dim, hidden_dim, self.flat_dim, dropout)

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(f"USAD expects x with shape (batch, {self.seq_len}, {self.input_dim})")
        flat = x.reshape(x.size(0), -1)
        z = self.encoder(flat)
        w1 = self.decoder_1(z).reshape_as(x)
        w2 = self.decoder_2(z).reshape_as(x)
        w3 = self.decoder_2(self.encoder(w1.reshape(w1.size(0), -1))).reshape_as(x)
        return w1, w2, w3

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        w1, _, w3 = self.reconstruct(x)
        recon = self.alpha * w1 + (1.0 - self.alpha) * w3
        return recon[:, -1]

    def fit_step(self, batch: DetectionBatch) -> torch.Tensor:
        w1, w2, w3 = self.reconstruct(batch.x)
        loss_1 = (w1 - batch.x).pow(2).mean()
        loss_2 = (w2 - batch.x).pow(2).mean()
        loss_3 = (w3 - batch.x).pow(2).mean()
        return loss_1 + loss_2 + 0.5 * loss_3

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        w1, _, w3 = self.reconstruct(batch.x)
        score = self.alpha * (w1[:, -1] - batch.x[:, -1]).pow(2) + (1.0 - self.alpha) * (
            w3[:, -1] - batch.x[:, -1]
        ).pow(2)
        if reduction == "none":
            return score
        if reduction == "sum":
            return score.sum(dim=-1)
        return score.mean(dim=-1)
