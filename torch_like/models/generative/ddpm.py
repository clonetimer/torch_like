"""DDPM image diffusion model.

Framework-oriented implementation of "Denoising Diffusion Probabilistic
Models". The model exposes the core diffusion-model API: forward noising,
epsilon prediction, denoising loss, and iterative sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel


@dataclass
class DDPMConfig(GenerativeConfig):
    num_blocks: int = 4
    time_dim: int = 128


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps.to(values.device))
    return out.reshape(timesteps.size(0), *((1,) * (len(shape) - 1))).to(dtype=torch.float32)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 0:
            return timesteps.float().unsqueeze(-1)
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=timesteps.device))
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DenoiseBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        return x + self.net(x)


class DDPM(GenerativeModel):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        d_model: int = 64,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.0,
        num_blocks: int = 4,
        time_dim: int = 128,
    ):
        super().__init__()
        if min(in_channels, image_size, d_model, num_timesteps, num_blocks, time_dim) <= 0:
            raise ValueError(
                "in_channels, image_size, d_model, num_timesteps, num_blocks, and time_dim must be positive"
            )
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("expected 0 < beta_start < beta_end < 1")

        self.in_channels = in_channels
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        self._config = DDPMConfig(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dropout=dropout,
            num_blocks=num_blocks,
            time_dim=time_dim,
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

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([DenoiseBlock(d_model, time_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, in_channels, kernel_size=3, padding=1),
        )
        self._init_weights()

    @classmethod
    def ddpm_tiny(cls, **kwargs) -> "DDPM":
        return cls(d_model=32, num_blocks=2, time_dim=64, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def q_sample(
        self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape).to(x_start.device)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape).to(x_start.device)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def forward(self, x_noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x_noisy.dim() != 4:
            raise ValueError("DDPM expects image tensors with shape (batch, channels, height, width)")
        if x_noisy.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x_noisy.size(1)}")
        if timesteps.dim() != 1 or timesteps.size(0) != x_noisy.size(0):
            raise ValueError("timesteps must have shape (batch,)")
        time_emb = self.time_mlp(timesteps)
        x = self.input_proj(x_noisy)
        for block in self.blocks:
            x = block(x, time_emb)
        return self.output(x)

    def predict_noise(self, x_noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self(x_noisy, timesteps)

    def loss_step(self, batch: GenerativeBatch) -> torch.Tensor:
        noise = batch.noise if batch.noise is not None else torch.randn_like(batch.x_start)
        x_noisy = self.q_sample(batch.x_start, batch.timesteps, noise)
        predicted_noise = self.predict_noise(x_noisy, batch.timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def denoise_step(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        betas_t = _extract(self.betas, timesteps, x_t.shape).to(x_t.device)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape).to(x_t.device)
        sqrt_recip_alpha = _extract(self.sqrt_recip_alphas, timesteps, x_t.shape).to(x_t.device)
        predicted_noise = self.predict_noise(x_t, timesteps)
        mean = sqrt_recip_alpha * (x_t - betas_t * predicted_noise / sqrt_one_minus.clamp(min=1e-8))
        variance = _extract(self.posterior_variance, timesteps, x_t.shape).to(x_t.device)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().reshape(x_t.size(0), *((1,) * (x_t.dim() - 1)))
        return mean + nonzero_mask * variance.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int, int],
        device: torch.device | str | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if len(shape) != 4:
            raise ValueError("shape must be (batch, channels, height, width)")
        if shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {shape[1]}")
        if device is None:
            device = self.betas.device
        if num_steps is None:
            num_steps = self.num_timesteps
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError(f"num_steps must be in [1, {self.num_timesteps}]")
        x = torch.randn(shape, device=device)
        step_indices = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
        for index, step in enumerate(step_indices):
            timesteps = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            if num_steps == self.num_timesteps:
                x = self.denoise_step(x, timesteps)
                continue

            predicted_noise = self.predict_noise(x, timesteps)
            alpha_t = _extract(self.alphas_cumprod, timesteps, x.shape).to(device=x.device, dtype=x.dtype)
            predicted_x_start = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            if index + 1 < len(step_indices):
                next_step = int(step_indices[index + 1].item())
                next_timesteps = torch.full((shape[0],), next_step, device=device, dtype=torch.long)
                alpha_next = _extract(self.alphas_cumprod, next_timesteps, x.shape).to(device=x.device, dtype=x.dtype)
                x = torch.sqrt(alpha_next) * predicted_x_start + torch.sqrt(1.0 - alpha_next) * predicted_noise
            else:
                x = predicted_x_start
        return x
