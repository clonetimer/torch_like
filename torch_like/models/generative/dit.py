"""DiT-style diffusion Transformer for image generation.

This is a framework-oriented implementation of "Scalable Diffusion Models
with Transformers". It keeps the torch_like diffusion API while using patch
tokens, timestep conditioning, and Transformer blocks instead of a U-Net.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel
from torch_like.models.generative.ddpm import SinusoidalTimeEmbedding, _extract


@dataclass
class DiTConfig(GenerativeConfig):
    patch_size: int = 4
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    time_dim: int = 128


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ada = nn.Linear(time_dim, 4 * d_model)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.ada(time_emb).chunk(4, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        return x + self.ffn(h)


class DiT(GenerativeModel):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        d_model: int = 64,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.0,
        patch_size: int = 4,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        time_dim: int = 128,
    ):
        super().__init__()
        if min(in_channels, image_size, d_model, num_timesteps, patch_size, n_heads, n_layers, d_ff, time_dim) <= 0:
            raise ValueError("all size parameters must be positive")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("expected 0 < beta_start < beta_end < 1")

        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps
        self._config = DiTConfig(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dropout=dropout,
            patch_size=patch_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
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

        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        grid = image_size // patch_size
        self.position = nn.Parameter(torch.zeros(1, grid * grid, d_model))
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, d_ff, time_dim, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, patch_size * patch_size * in_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, num_patches, patch_dim = tokens.shape
        grid = int(num_patches**0.5)
        expected = self.patch_size * self.patch_size * self.in_channels
        if grid * grid != num_patches or patch_dim != expected:
            raise ValueError("invalid patch token shape")
        x = tokens.reshape(batch, grid, grid, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(batch, self.in_channels, grid * self.patch_size, grid * self.patch_size)

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
            raise ValueError("DiT expects image tensors with shape (batch, channels, height, width)")
        if x_noisy.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(
                f"expected image shape (*, {self.in_channels}, {self.image_size}, {self.image_size}), "
                f"got {tuple(x_noisy.shape)}"
            )
        if timesteps.dim() != 1 or timesteps.size(0) != x_noisy.size(0):
            raise ValueError("timesteps must have shape (batch,)")

        x = self.patch_embed(x_noisy).flatten(2).transpose(1, 2) + self.position
        time_emb = self.time_mlp(timesteps)
        for block in self.blocks:
            x = block(x, time_emb)
        return self._unpatchify(self.out(self.norm(x)))

    def predict_noise(self, x_noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self(x_noisy, timesteps)

    def loss_step(self, batch: GenerativeBatch) -> torch.Tensor:
        noise = batch.noise if batch.noise is not None else torch.randn_like(batch.x_start)
        x_noisy = self.q_sample(batch.x_start, batch.timesteps, noise)
        return F.mse_loss(self.predict_noise(x_noisy, batch.timesteps), noise)

    @torch.no_grad()
    def denoise_step(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        betas_t = _extract(self.betas, timesteps, x_t.shape).to(x_t.device)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape).to(x_t.device)
        sqrt_recip_alpha = _extract(self.sqrt_recip_alphas, timesteps, x_t.shape).to(x_t.device)
        mean = sqrt_recip_alpha * (x_t - betas_t * self.predict_noise(x_t, timesteps) / sqrt_one_minus.clamp(min=1e-8))
        variance = _extract(self.posterior_variance, timesteps, x_t.shape).to(x_t.device)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().reshape(x_t.size(0), *((1,) * (x_t.dim() - 1)))
        return mean + nonzero_mask * variance.sqrt() * noise

    @torch.no_grad()
    def sample(
        self, batch_size: int, device: torch.device | str | None = None, num_steps: int | None = None
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = self.betas.device
        if num_steps is None:
            num_steps = self.num_timesteps
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError(f"num_steps must be in [1, {self.num_timesteps}]")
        x = torch.randn(batch_size, self.in_channels, self.image_size, self.image_size, device=device)
        step_indices = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
        for step in step_indices:
            timesteps = torch.full((batch_size,), int(step.item()), device=device, dtype=torch.long)
            x = self.denoise_step(x, timesteps)
        return x
