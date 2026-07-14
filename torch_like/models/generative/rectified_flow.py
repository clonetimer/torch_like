"""Rectified Flow / Flow Matching image generator.

Rectified Flow learns a velocity field that transports samples from a base
noise distribution to the data distribution through a straight-line
interpolation path. This model exposes the framework-facing core only:
interpolation, velocity prediction, flow-matching loss, and Euler sampling.
External ODE solvers, large pretrained checkpoints, guidance policies, and
dataset-specific preprocessing stay outside the model class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel
from torch_like.models.generative.ddpm import DenoiseBlock, SinusoidalTimeEmbedding


@dataclass
class RectifiedFlowConfig(GenerativeConfig):
    num_blocks: int = 4
    time_dim: int = 128
    condition_dim: int = 0


class RectifiedFlow(GenerativeModel):
    """Compact flow-matching generator for image tensors."""

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
        condition_dim: int = 0,
    ):
        super().__init__()
        if min(in_channels, image_size, d_model, num_timesteps, num_blocks, time_dim) <= 0:
            raise ValueError("RectifiedFlow dimensions, layer counts, and timesteps must be positive")
        if condition_dim < 0:
            raise ValueError("condition_dim must be non-negative")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("expected 0 < beta_start < beta_end < 1")

        self.in_channels = in_channels
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        self.condition_dim = condition_dim
        self._config = RectifiedFlowConfig(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dropout=dropout,
            num_blocks=num_blocks,
            time_dim=time_dim,
            condition_dim=condition_dim,
        )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.condition_proj = nn.Linear(condition_dim, time_dim) if condition_dim > 0 else None
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([DenoiseBlock(d_model, time_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, in_channels, kernel_size=3, padding=1),
        )
        self._init_weights()

    @classmethod
    def rectified_flow_tiny(cls, **kwargs) -> "RectifiedFlow":
        return cls(d_model=32, num_blocks=2, time_dim=64, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_image(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("RectifiedFlow expects image tensors with shape (batch, channels, height, width)")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}")

    def _time_values(self, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        if timesteps.dim() != 1 or timesteps.size(0) != shape[0]:
            raise ValueError("timesteps must have shape (batch,)")
        t = timesteps.to(dtype=torch.float32)
        if torch.any(t < 0):
            raise ValueError("timesteps must be non-negative")
        if torch.any(t > 1):
            t = t / max(self.num_timesteps - 1, 1)
        if torch.any(t > 1):
            raise ValueError("timesteps must be in [0, 1] or integer indices below num_timesteps")
        return t.reshape(shape[0], *((1,) * (len(shape) - 1)))

    def _time_embedding_input(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = timesteps.to(dtype=torch.float32)
        if torch.any(t > 1):
            return t
        return t * max(self.num_timesteps - 1, 1)

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

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_image(x0)
        self._validate_image(x1)
        if x0.shape != x1.shape:
            raise ValueError("x0 and x1 must have the same shape")
        t = self._time_values(timesteps, x1.shape).to(x1.device)
        return (1.0 - t) * x0 + t * x1

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        self._validate_image(x0)
        self._validate_image(x1)
        if x0.shape != x1.shape:
            raise ValueError("x0 and x1 must have the same shape")
        return x1 - x0

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_image(x_t)
        if timesteps.dim() != 1 or timesteps.size(0) != x_t.size(0):
            raise ValueError("timesteps must have shape (batch,)")
        time_emb = self.time_mlp(self._time_embedding_input(timesteps).to(x_t.device))
        condition_emb = self._condition_embedding(condition, x_t.size(0), x_t.device)
        if condition_emb is not None:
            time_emb = time_emb + self.condition_proj(condition_emb)
        x = self.input_proj(x_t)
        for block in self.blocks:
            x = block(x, time_emb)
        return self.output(x)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self(x_t, timesteps, condition=condition)

    def loss_step(self, batch: GenerativeBatch, condition: torch.Tensor | None = None) -> torch.Tensor:
        x1 = batch.x_start
        self._validate_image(x1)
        x0 = batch.noise if batch.noise is not None else torch.randn_like(x1)
        if x0.shape != x1.shape:
            raise ValueError("RectifiedFlow batch.noise must match x_start shape")
        x_t = self.interpolate(x0, x1, batch.timesteps)
        target = self.target_velocity(x0, x1)
        predicted = self.predict_velocity(x_t, batch.timesteps, condition=condition)
        return F.mse_loss(predicted, target)

    @torch.no_grad()
    def euler_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        velocity = self.predict_velocity(x_t, t, condition=condition)
        return x_t + dt * velocity

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int, int],
        condition: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        num_steps: int = 16,
    ) -> torch.Tensor:
        if len(shape) != 4:
            raise ValueError("shape must be (batch, channels, height, width)")
        if shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {shape[1]}")
        if shape[2] != self.image_size or shape[3] != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if device is None:
            device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        if condition is not None:
            condition = condition.to(device)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((shape[0],), step / num_steps, device=device, dtype=torch.float32)
            x = self.euler_step(x, t, dt, condition=condition)
        return x
