"""ControlNet-style conditional latent diffusion model.

The implementation captures the reusable model-side idea from ControlNet:
encode an external control signal and inject zero-initialized residuals into a
latent diffusion denoiser. It intentionally does not include Stable Diffusion
checkpoint conversion, annotators, ControlNet preprocessors, tokenizer logic,
or classifier-free guidance policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch
from torch_like.models.generative.latent_diffusion import LatentDiffusion, LatentDiffusionConfig


@dataclass
class ControlNetConfig(LatentDiffusionConfig):
    control_channels: int = 3
    control_strength: float = 1.0


class ControlNet(LatentDiffusion):
    """Latent diffusion denoiser with ControlNet-style control residuals."""

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
        control_channels: int = 3,
        control_strength: float = 1.0,
    ):
        super().__init__(
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
        if control_channels <= 0:
            raise ValueError("control_channels must be positive")
        if control_strength < 0:
            raise ValueError("control_strength must be non-negative")

        self.control_channels = control_channels
        self.control_strength = control_strength
        self._config = ControlNetConfig(
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
            control_channels=control_channels,
            control_strength=control_strength,
        )

        downsample_steps = int(math.log2(downsample_factor))
        control_layers: list[nn.Module] = [
            nn.Conv2d(control_channels, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(downsample_steps):
            control_layers.extend(
                [
                    nn.Conv2d(d_model, d_model, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(1, d_model),
                    nn.SiLU(),
                ]
            )
        self.control_encoder = nn.Sequential(*control_layers)
        self.control_projections = nn.ModuleList(
            [nn.Conv2d(d_model, d_model, kernel_size=1) for _ in range(num_blocks)]
        )
        self._init_control_weights()

    @classmethod
    def controlnet_tiny(cls, **kwargs) -> "ControlNet":
        return cls(d_model=32, num_blocks=2, time_dim=64, latent_channels=4, **kwargs)

    def _init_control_weights(self) -> None:
        for module in self.control_encoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for projection in self.control_projections:
            nn.init.zeros_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def _validate_control(self, control: torch.Tensor, batch_size: int) -> None:
        if control.dim() != 4:
            raise ValueError("ControlNet expects control tensors with shape (batch, channels, height, width)")
        if control.size(0) != batch_size:
            raise ValueError("control batch size must match latent batch size")
        if control.size(1) != self.control_channels:
            raise ValueError(f"Expected {self.control_channels} control channels, got {control.size(1)}")
        if control.size(2) != self.image_size or control.size(3) != self.image_size:
            raise ValueError(f"Expected control size {self.image_size}x{self.image_size}")

    def encode_control(self, control: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = control.size(0)
        self._validate_control(control, batch_size)
        return self.control_encoder(control)

    def control_residuals(self, control: torch.Tensor, batch_size: int | None = None) -> list[torch.Tensor]:
        encoded = self.encode_control(control, batch_size=batch_size)
        return [projection(encoded) * self.control_strength for projection in self.control_projections]

    def forward(
        self,
        z_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        control: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_latents(z_noisy)
        self._validate_control(control, z_noisy.size(0))
        if timesteps.dim() != 1 or timesteps.size(0) != z_noisy.size(0):
            raise ValueError("timesteps must have shape (batch,)")

        time_emb = self.time_mlp(timesteps)
        condition_emb = self._condition_embedding(condition, z_noisy.size(0), z_noisy.device)
        if condition_emb is not None:
            time_emb = time_emb + self.condition_proj(condition_emb)

        residuals = self.control_residuals(control.to(z_noisy.device), batch_size=z_noisy.size(0))
        x = self.input_proj(z_noisy)
        for block, residual in zip(self.blocks, residuals, strict=True):
            x = block(x, time_emb) + residual.to(x.device, dtype=x.dtype)
        return self.output(x)

    def predict_noise(
        self,
        z_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        control: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self(z_noisy, timesteps, control=control, condition=condition)

    def loss_step(
        self,
        batch: GenerativeBatch,
        control: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_start = self.encode_to_latent(batch.x_start)
        noise = batch.noise if batch.noise is not None else torch.randn_like(z_start)
        if noise.shape != z_start.shape:
            raise ValueError("ControlNet batch.noise must match encoded latent shape")
        z_noisy = self.q_sample(z_start, batch.timesteps, noise)
        predicted_noise = self.predict_noise(z_noisy, batch.timesteps, control=control, condition=condition)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def denoise_step(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        control: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        betas_t = self.betas.gather(0, timesteps.to(self.betas.device)).reshape(z_t.size(0), 1, 1, 1).to(z_t.device)
        sqrt_one_minus = (
            self.sqrt_one_minus_alphas_cumprod.gather(0, timesteps.to(self.betas.device))
            .reshape(z_t.size(0), 1, 1, 1)
            .to(z_t.device)
        )
        sqrt_recip_alpha = (
            self.sqrt_recip_alphas.gather(0, timesteps.to(self.betas.device))
            .reshape(z_t.size(0), 1, 1, 1)
            .to(z_t.device)
        )
        predicted_noise = self.predict_noise(z_t, timesteps, control=control, condition=condition)
        mean = sqrt_recip_alpha * (z_t - betas_t * predicted_noise / sqrt_one_minus.clamp(min=1e-8))
        variance = (
            self.posterior_variance.gather(0, timesteps.to(self.betas.device))
            .reshape(z_t.size(0), 1, 1, 1)
            .to(z_t.device)
        )
        noise = torch.randn_like(z_t)
        nonzero_mask = (timesteps != 0).float().reshape(z_t.size(0), *((1,) * (z_t.dim() - 1)))
        return mean + nonzero_mask * variance.sqrt() * noise

    @torch.no_grad()
    def sample_latents(
        self,
        control: torch.Tensor,
        condition: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = self.betas.device
        control = control.to(device)
        num_samples = control.size(0)
        self._validate_control(control, num_samples)
        if condition is not None:
            condition = condition.to(device)
        if num_steps is None:
            num_steps = self.num_timesteps
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError(f"num_steps must be in [1, {self.num_timesteps}]")

        z = torch.randn(num_samples, self.latent_channels, self.latent_size, self.latent_size, device=device)
        step_indices = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
        for index, step in enumerate(step_indices):
            timesteps = torch.full((num_samples,), int(step.item()), device=device, dtype=torch.long)
            if num_steps == self.num_timesteps:
                z = self.denoise_step(z, timesteps, control=control, condition=condition)
                continue

            predicted_noise = self.predict_noise(z, timesteps, control=control, condition=condition)
            alpha_t = (
                self.alphas_cumprod.gather(0, timesteps.to(self.betas.device))
                .reshape(num_samples, 1, 1, 1)
                .to(
                    device=z.device,
                    dtype=z.dtype,
                )
            )
            predicted_z_start = (z - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            if index + 1 < len(step_indices):
                next_step = int(step_indices[index + 1].item())
                next_timesteps = torch.full((num_samples,), next_step, device=device, dtype=torch.long)
                alpha_next = (
                    self.alphas_cumprod.gather(0, next_timesteps.to(self.betas.device))
                    .reshape(num_samples, 1, 1, 1)
                    .to(device=z.device, dtype=z.dtype)
                )
                z = torch.sqrt(alpha_next) * predicted_z_start + torch.sqrt(1.0 - alpha_next) * predicted_noise
            else:
                z = predicted_z_start
        return z

    @torch.no_grad()
    def sample(
        self,
        control: torch.Tensor,
        condition: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        latents = self.sample_latents(control=control, condition=condition, device=device, num_steps=num_steps)
        return self.decode_latent(latents)
