"""Variational autoencoder image model.

Framework-oriented implementation of "Auto-Encoding Variational Bayes".
The model exposes encoder statistics, reparameterized latent samples,
reconstruction output, ELBO-style loss, and latent sampling helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel


@dataclass
class VAEConfig(GenerativeConfig):
    latent_dim: int = 128
    hidden_channels: tuple[int, ...] = (32, 64)
    reconstruction_loss: str = "mse"
    kl_weight: float = 1.0


class VAE(GenerativeModel):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        d_model: int = 64,
        dropout: float = 0.0,
        latent_dim: int = 128,
        hidden_channels: tuple[int, ...] = (32, 64),
        reconstruction_loss: str = "mse",
        kl_weight: float = 1.0,
        num_timesteps: int = 1,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        del num_timesteps, beta_start, beta_end
        if reconstruction_loss not in {"mse", "bce"}:
            raise ValueError("reconstruction_loss must be 'mse' or 'bce'")
        if min(in_channels, image_size, latent_dim) <= 0:
            raise ValueError("in_channels, image_size, and latent_dim must be positive")
        if not hidden_channels or any(channel <= 0 for channel in hidden_channels):
            raise ValueError("hidden_channels must contain positive values")
        if image_size % (2 ** len(hidden_channels)) != 0:
            raise ValueError("image_size must be divisible by 2 ** len(hidden_channels)")
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss
        self.kl_weight = kl_weight
        self._config = VAEConfig(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
        )

        encoder_layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels in hidden_channels:
            encoder_layers.extend(
                [
                    nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)
        self.spatial_size = image_size // (2 ** len(hidden_channels))
        self.flat_dim = current_channels * self.spatial_size * self.spatial_size
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)
        decoder_layers: list[nn.Module] = []
        reversed_channels = list(reversed(hidden_channels))
        current_channels = reversed_channels[0]
        for out_channels in reversed_channels[1:]:
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_channels = out_channels
        decoder_layers.append(nn.ConvTranspose2d(current_channels, in_channels, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)
        self._init_weights()

    @classmethod
    def vae_tiny(cls, **kwargs) -> "VAE":
        return cls(latent_dim=16, hidden_channels=(8, 16), **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError("VAE expects image tensors with shape (batch, channels, height, width)")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
        if x.size(-2) != self.image_size or x.size(-1) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}, got {x.size(-2)}x{x.size(-1)}")
        hidden = self.encoder(x).flatten(1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError("VAE expects z with shape (batch, latent_dim)")
        if z.size(1) != self.latent_dim:
            raise ValueError(f"Expected latent_dim {self.latent_dim}, got {z.size(1)}")
        hidden = self.decoder_input(z).view(z.size(0), -1, self.spatial_size, self.spatial_size)
        reconstruction = self.decoder(hidden)
        return torch.sigmoid(reconstruction) if self.reconstruction_loss == "bce" else reconstruction

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return {"reconstruction": reconstruction, "mu": mu, "logvar": logvar, "z": z}

    def predict(self, batch: GenerativeBatch) -> dict[str, torch.Tensor]:
        return self(batch.x_start)

    def loss_step(self, batch: GenerativeBatch) -> torch.Tensor:
        outputs = self.predict(batch)
        reconstruction = outputs["reconstruction"]
        if self.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(reconstruction, batch.x_start, reduction="mean")
        else:
            recon_loss = F.mse_loss(reconstruction, batch.x_start)
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.kl_weight * kl_loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device | str | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z is None:
            if device is None:
                device = next(self.parameters()).device
            z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
