"""DCGAN image generator/discriminator pair.

Framework-oriented implementation of "Unsupervised Representation Learning
with Deep Convolutional Generative Adversarial Networks". The model exposes
generator, discriminator, sampling, and adversarial loss helpers without owning
the two-optimizer training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel


@dataclass
class DCGANConfig(GenerativeConfig):
    latent_dim: int = 100
    feature_maps: int = 64


def _num_upsample_blocks(image_size: int) -> int:
    if image_size < 8 or image_size & (image_size - 1) != 0:
        raise ValueError("image_size must be a power of two and at least 8")
    return image_size.bit_length() - 3


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, image_size: int, feature_maps: int):
        super().__init__()
        blocks = _num_upsample_blocks(image_size)
        start_channels = feature_maps * (2 ** max(blocks - 1, 0))
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_dim, start_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(start_channels),
            nn.ReLU(inplace=True),
        ]
        current_channels = start_channels
        for _ in range(blocks):
            next_channels = max(feature_maps, current_channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = next_channels
        layers.extend(
            [
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        if z.dim() != 4:
            raise ValueError("DCGANGenerator expects z with shape (batch, latent_dim) or (batch, latent_dim, 1, 1)")
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int, image_size: int, feature_maps: int):
        super().__init__()
        blocks = _num_upsample_blocks(image_size)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        current_channels = feature_maps
        for _ in range(max(blocks - 1, 0)):
            next_channels = current_channels * 2
            layers.extend(
                [
                    nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_channels = next_channels
        layers.append(nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=0))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.flatten(1).mean(dim=1)


class DCGAN(GenerativeModel):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 64,
        d_model: int = 64,
        dropout: float = 0.0,
        latent_dim: int = 100,
        feature_maps: int = 64,
        num_timesteps: int = 1,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        del dropout, num_timesteps, beta_start, beta_end
        if min(in_channels, image_size, latent_dim, feature_maps, d_model) <= 0:
            raise ValueError("in_channels, image_size, latent_dim, feature_maps, and d_model must be positive")
        _num_upsample_blocks(image_size)
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self._config = DCGANConfig(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            dropout=0.0,
            latent_dim=latent_dim,
            feature_maps=feature_maps,
        )
        self.generator = DCGANGenerator(latent_dim, in_channels, image_size, feature_maps)
        self.discriminator = DCGANDiscriminator(in_channels, image_size, feature_maps)
        self._init_weights()

    @classmethod
    def dcgan_tiny(cls, **kwargs) -> "DCGAN":
        return cls(latent_dim=16, feature_maps=8, image_size=16, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def sample_latent(
        self,
        num_samples: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(num_samples, self.latent_dim, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def sample(
        self,
        num_samples: int,
        device: torch.device | str | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z is None:
            z = self.sample_latent(num_samples, device=device)
        return self.generator(z)

    def predict(self, batch: GenerativeBatch) -> torch.Tensor:
        return self.sample(batch.x_start.size(0), device=batch.x_start.device)

    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError("DCGAN expects image tensors with shape (batch, channels, height, width)")
        return self.discriminator(images)

    def generator_loss(self, fake_images: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        if fake_images is None:
            if z is None:
                raise ValueError("generator_loss requires fake_images or z")
            fake_images = self.generator(z)
        fake_logits = self.discriminator(fake_images)
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

    def discriminator_loss(self, real_images: torch.Tensor, fake_images: torch.Tensor | None = None) -> torch.Tensor:
        if fake_images is None:
            fake_images = self.sample(real_images.size(0), device=real_images.device)
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (real_loss + fake_loss)

    def loss_step(self, batch: GenerativeBatch) -> dict[str, torch.Tensor]:
        fake_images = self.sample(batch.x_start.size(0), device=batch.x_start.device)
        return {
            "generator_loss": self.generator_loss(fake_images),
            "discriminator_loss": self.discriminator_loss(batch.x_start, fake_images),
        }
