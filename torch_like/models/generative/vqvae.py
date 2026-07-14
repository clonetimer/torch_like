"""VQ-VAE discrete latent image model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel


@dataclass
class VQVAEConfig(GenerativeConfig):
    latent_dim: int = 64
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    hidden_channels: tuple[int, ...] = (64, 128)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.reshape(-1, self.embedding_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view_as(z_perm).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = z + (quantized - z).detach()
        perplexity = self._perplexity(indices, z.device)
        return quantized, loss, indices.view(z.size(0), z.size(2), z.size(3)), perplexity

    def _perplexity(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        encodings = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = encodings.mean(dim=0)
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        return torch.exp(entropy).to(device)


class VQVAE(GenerativeModel):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        latent_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        hidden_channels: tuple[int, ...] = (64, 128),
        d_model: int = 64,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % (2 ** len(hidden_channels)) != 0:
            raise ValueError("image_size must be divisible by 2 ** len(hidden_channels)")
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self._config = VQVAEConfig(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dropout=dropout,
            latent_dim=latent_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            hidden_channels=hidden_channels,
        )
        encoder: list[nn.Module] = []
        prev = in_channels
        for channels in hidden_channels:
            encoder.extend([nn.Conv2d(prev, channels, 4, stride=2, padding=1), nn.GELU()])
            prev = channels
        encoder.extend([nn.Conv2d(prev, latent_dim, 3, padding=1), nn.GELU(), nn.Conv2d(latent_dim, latent_dim, 1)])
        self.encoder = nn.Sequential(*encoder)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        decoder: list[nn.Module] = [nn.Conv2d(latent_dim, hidden_channels[-1], 3, padding=1), nn.GELU()]
        channels = list(reversed(hidden_channels))
        for idx, channels_out in enumerate(channels[1:] + [in_channels]):
            channels_in = channels[idx]
            decoder.append(nn.ConvTranspose2d(channels_in, channels_out, 4, stride=2, padding=1))
            if channels_out != in_channels:
                decoder.append(nn.GELU())
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantizer(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del timesteps
        z = self.encode(x)
        z_q, quantization_loss, indices, perplexity = self.quantize(z)
        reconstruction = self.decode(z_q)
        return {
            "reconstruction": reconstruction,
            "z": z,
            "z_q": z_q,
            "indices": indices,
            "quantization_loss": quantization_loss,
            "perplexity": perplexity,
        }

    def predict(self, batch: GenerativeBatch) -> dict[str, torch.Tensor]:
        return self(batch.x_start, batch.timesteps)

    def loss_step(self, batch: GenerativeBatch) -> torch.Tensor:
        out = self.predict(batch)
        reconstruction_loss = F.mse_loss(out["reconstruction"], batch.x_start)
        return reconstruction_loss + out["quantization_loss"]

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = torch.device(device) if device is not None else next(self.parameters()).device
        scale = 2 ** len(self._config.hidden_channels)
        latent_size = self.image_size // scale
        indices = torch.randint(
            0,
            self.quantizer.num_embeddings,
            (num_samples, latent_size, latent_size),
            device=device,
        )
        z_q = self.quantizer.embedding(indices.reshape(-1)).view(num_samples, latent_size, latent_size, self.latent_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)
