"""Latent diffusion image generator.

This is a framework-oriented implementation of the core idea behind Latent
Diffusion Models and Stable Diffusion: encode images into a lower-resolution
latent grid, train a denoiser in that latent space, and decode sampled latents
back to image space. Tokenizers, text encoders, pretrained weights, guidance
policy, and scheduler libraries intentionally stay outside this model class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel
from torch_like.models.generative.ddpm import DenoiseBlock, SinusoidalTimeEmbedding, _extract


@dataclass
class LatentDiffusionConfig(GenerativeConfig):
    latent_channels: int = 4
    downsample_factor: int = 4
    num_blocks: int = 4
    time_dim: int = 128
    condition_dim: int = 0
    latent_scaling: float = 0.18215


class LatentDiffusion(GenerativeModel):
    """Compact latent diffusion model with optional vector conditioning."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        d_model: int = 64,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.0,
        latent_channels: int = 4,
        downsample_factor: int = 4,
        num_blocks: int = 4,
        time_dim: int = 128,
        condition_dim: int = 0,
        latent_scaling: float = 0.18215,
    ):
        super().__init__()
        if (
            min(
                in_channels,
                image_size,
                d_model,
                num_timesteps,
                latent_channels,
                downsample_factor,
                num_blocks,
                time_dim,
            )
            <= 0
        ):
            raise ValueError("LatentDiffusion dimensions, layer counts, and timesteps must be positive")
        if condition_dim < 0:
            raise ValueError("condition_dim must be non-negative")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("expected 0 < beta_start < beta_end < 1")
        if image_size % downsample_factor != 0:
            raise ValueError("image_size must be divisible by downsample_factor")
        if downsample_factor & (downsample_factor - 1) != 0:
            raise ValueError("downsample_factor must be a power of two")
        if latent_scaling <= 0:
            raise ValueError("latent_scaling must be positive")

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.latent_size = image_size // downsample_factor
        self.num_timesteps = num_timesteps
        self.condition_dim = condition_dim
        self.latent_scaling = latent_scaling
        self._config = LatentDiffusionConfig(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dropout=dropout,
            latent_channels=latent_channels,
            downsample_factor=downsample_factor,
            num_blocks=num_blocks,
            time_dim=time_dim,
            condition_dim=condition_dim,
            latent_scaling=latent_scaling,
        )

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

        downsample_steps = int(math.log2(downsample_factor))
        encoder_layers: list[nn.Module] = [
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(downsample_steps):
            encoder_layers.extend(
                [
                    nn.Conv2d(d_model, d_model, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(1, d_model),
                    nn.SiLU(),
                ]
            )
        encoder_layers.append(nn.Conv2d(d_model, latent_channels, kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = [
            nn.Conv2d(latent_channels, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(downsample_steps):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(1, d_model),
                    nn.SiLU(),
                ]
            )
        decoder_layers.append(nn.Conv2d(d_model, in_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.condition_proj = nn.Linear(condition_dim, time_dim) if condition_dim > 0 else None
        self.input_proj = nn.Conv2d(latent_channels, d_model, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([DenoiseBlock(d_model, time_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, latent_channels, kernel_size=3, padding=1),
        )
        self._init_weights()

    @classmethod
    def latent_diffusion_tiny(cls, **kwargs) -> "LatentDiffusion":
        return cls(d_model=32, num_blocks=2, time_dim=64, latent_channels=4, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_image(self, image: torch.Tensor) -> None:
        if image.dim() != 4:
            raise ValueError("LatentDiffusion expects image tensors with shape (batch, channels, height, width)")
        if image.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} image channels, got {image.size(1)}")
        if image.size(2) != self.image_size or image.size(3) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}")

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if latents.dim() != 4:
            raise ValueError("LatentDiffusion expects latent tensors with shape (batch, channels, height, width)")
        if latents.size(1) != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {latents.size(1)}")
        if latents.size(2) != self.latent_size or latents.size(3) != self.latent_size:
            raise ValueError(f"Expected latent size {self.latent_size}x{self.latent_size}")

    def _condition_embedding(
        self, condition: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor | None:
        if self.condition_proj is None:
            if condition is not None:
                raise ValueError("condition was provided but condition_dim is 0")
            return None
        if condition is None:
            return torch.zeros(batch_size, self.condition_proj.in_features, device=device)
        if (
            condition.dim() != 2
            or condition.size(0) != batch_size
            or condition.size(1) != self.condition_proj.in_features
        ):
            raise ValueError(f"condition must have shape (batch, {self.condition_proj.in_features})")
        return condition.to(device)

    def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_image(image)
        return self.encoder(image) * self.latent_scaling

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._validate_latents(latents)
        return self.decoder(latents / self.latent_scaling)

    def q_sample(
        self, z_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._validate_latents(z_start)
        if noise is None:
            noise = torch.randn_like(z_start)
        if noise.shape != z_start.shape:
            raise ValueError("noise must match latent tensor shape")
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, z_start.shape).to(z_start.device)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, z_start.shape).to(z_start.device)
        return sqrt_alpha * z_start + sqrt_one_minus * noise

    def forward(
        self,
        z_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_latents(z_noisy)
        if timesteps.dim() != 1 or timesteps.size(0) != z_noisy.size(0):
            raise ValueError("timesteps must have shape (batch,)")
        time_emb = self.time_mlp(timesteps)
        condition_emb = self._condition_embedding(condition, z_noisy.size(0), z_noisy.device)
        if condition_emb is not None:
            time_emb = time_emb + self.condition_proj(condition_emb)
        x = self.input_proj(z_noisy)
        for block in self.blocks:
            x = block(x, time_emb)
        return self.output(x)

    def predict_noise(
        self,
        z_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self(z_noisy, timesteps, condition=condition)

    def loss_step(self, batch: GenerativeBatch, condition: torch.Tensor | None = None) -> torch.Tensor:
        z_start = self.encode_to_latent(batch.x_start)
        noise = batch.noise if batch.noise is not None else torch.randn_like(z_start)
        if noise.shape != z_start.shape:
            raise ValueError("LatentDiffusion batch.noise must match encoded latent shape")
        z_noisy = self.q_sample(z_start, batch.timesteps, noise)
        predicted_noise = self.predict_noise(z_noisy, batch.timesteps, condition=condition)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def denoise_step(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        betas_t = _extract(self.betas, timesteps, z_t.shape).to(z_t.device)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, z_t.shape).to(z_t.device)
        sqrt_recip_alpha = _extract(self.sqrt_recip_alphas, timesteps, z_t.shape).to(z_t.device)
        predicted_noise = self.predict_noise(z_t, timesteps, condition=condition)
        mean = sqrt_recip_alpha * (z_t - betas_t * predicted_noise / sqrt_one_minus.clamp(min=1e-8))
        variance = _extract(self.posterior_variance, timesteps, z_t.shape).to(z_t.device)
        noise = torch.randn_like(z_t)
        nonzero_mask = (timesteps != 0).float().reshape(z_t.size(0), *((1,) * (z_t.dim() - 1)))
        return mean + nonzero_mask * variance.sqrt() * noise

    @torch.no_grad()
    def sample_latents(
        self,
        num_samples: int,
        condition: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if device is None:
            device = self.betas.device
        if num_steps is None:
            num_steps = self.num_timesteps
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError(f"num_steps must be in [1, {self.num_timesteps}]")

        z = torch.randn(num_samples, self.latent_channels, self.latent_size, self.latent_size, device=device)
        if condition is not None:
            condition = condition.to(device)
        step_indices = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
        for index, step in enumerate(step_indices):
            timesteps = torch.full((num_samples,), int(step.item()), device=device, dtype=torch.long)
            if num_steps == self.num_timesteps:
                z = self.denoise_step(z, timesteps, condition=condition)
                continue

            predicted_noise = self.predict_noise(z, timesteps, condition=condition)
            alpha_t = _extract(self.alphas_cumprod, timesteps, z.shape).to(device=z.device, dtype=z.dtype)
            predicted_z_start = (z - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            if index + 1 < len(step_indices):
                next_step = int(step_indices[index + 1].item())
                next_timesteps = torch.full((num_samples,), next_step, device=device, dtype=torch.long)
                alpha_next = _extract(self.alphas_cumprod, next_timesteps, z.shape).to(device=z.device, dtype=z.dtype)
                z = torch.sqrt(alpha_next) * predicted_z_start + torch.sqrt(1.0 - alpha_next) * predicted_noise
            else:
                z = predicted_z_start
        return z

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        condition: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        latents = self.sample_latents(num_samples, condition=condition, device=device, num_steps=num_steps)
        return self.decode_latent(latents)
