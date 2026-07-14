"""RealNVP normalizing flow for image-density modeling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel


@dataclass
class RealNVPConfig(GenerativeConfig):
    hidden_dim: int = 256
    num_coupling_layers: int = 6
    scale_clip: float = 2.0


class CouplingNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, scale_clip: float):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_shift = CouplingNet(dim, hidden_dim, 2 * dim)
        self.scale_clip = scale_clip

    def _params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.scale_shift(x * self.mask).chunk(2, dim=-1)
        scale = torch.tanh(scale) * self.scale_clip
        return scale * (1 - self.mask), shift * (1 - self.mask)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._params(x)
        z = x * self.mask + (1 - self.mask) * (x * torch.exp(scale) + shift)
        return z, scale.sum(dim=-1)

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._params(z)
        x = z * self.mask + (1 - self.mask) * ((z - shift) * torch.exp(-scale))
        return x, -scale.sum(dim=-1)


class RealNVP(GenerativeModel):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        d_model: int = 64,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.0,
        hidden_dim: int = 256,
        num_coupling_layers: int = 6,
        scale_clip: float = 2.0,
    ):
        del d_model, num_timesteps, beta_start, beta_end, dropout
        super().__init__()
        if min(in_channels, image_size, hidden_dim, num_coupling_layers) <= 0:
            raise ValueError("in_channels, image_size, hidden_dim, and num_coupling_layers must be positive")
        if scale_clip <= 0:
            raise ValueError("scale_clip must be positive")
        self.in_channels = in_channels
        self.image_size = image_size
        self.dim = in_channels * image_size * image_size
        self._config = RealNVPConfig(
            in_channels=in_channels,
            image_size=image_size,
            hidden_dim=hidden_dim,
            num_coupling_layers=num_coupling_layers,
            scale_clip=scale_clip,
        )

        base = torch.arange(self.dim) % 2
        layers = []
        for idx in range(num_coupling_layers):
            mask = (base if idx % 2 == 0 else 1 - base).float()
            layers.append(AffineCoupling(self.dim, hidden_dim, mask, scale_clip))
        self.layers = nn.ModuleList(layers)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("RealNVP expects image tensors with shape (batch, channels, height, width)")
        if x.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(
                f"expected image shape (*, {self.in_channels}, {self.image_size}, {self.image_size}), "
                f"got {tuple(x.shape)}"
            )
        return x.reshape(x.size(0), -1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del timesteps
        z = self._flatten(x)
        log_det = z.new_zeros(z.size(0))
        for layer in self.layers:
            z, delta = layer(z)
            log_det = log_det + delta
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.dim:
            raise ValueError(f"z must have shape (batch, {self.dim})")
        x = z
        for layer in reversed(self.layers):
            x, _ = layer.inverse(x)
        return x.reshape(z.size(0), self.in_channels, self.image_size, self.image_size)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self(x)
        log_base = -0.5 * (z.pow(2) + torch.log(torch.tensor(2.0 * torch.pi, device=z.device))).sum(dim=-1)
        return log_base + log_det

    def loss_step(self, batch: GenerativeBatch) -> torch.Tensor:
        return -self.log_prob(batch.x_start).mean()

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(batch_size, self.dim, device=device)
        return self.inverse(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self(x)
        return self.inverse(z)
