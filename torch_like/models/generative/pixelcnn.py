"""PixelCNN-style autoregressive image density model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel


@dataclass
class PixelCNNConfig(GenerativeConfig):
    num_values: int = 256
    hidden_channels: int = 64
    num_layers: int = 7
    kernel_size: int = 7


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        _, _, height, width = self.weight.shape
        mask = torch.ones_like(self.weight)
        center_h, center_w = height // 2, width // 2
        mask[:, :, center_h + 1 :, :] = 0
        mask[:, :, center_h, center_w + (mask_type == "B") :] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNN(GenerativeModel):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        d_model: int = 64,
        dropout: float = 0.0,
        num_values: int = 256,
        hidden_channels: int = 64,
        num_layers: int = 7,
        kernel_size: int = 7,
        num_timesteps: int = 1,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        del d_model, num_timesteps, beta_start, beta_end
        if min(in_channels, image_size, num_values, hidden_channels, num_layers, kernel_size) <= 0:
            raise ValueError("PixelCNN dimensions and counts must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_values = num_values
        self._config = PixelCNNConfig(
            in_channels=in_channels,
            image_size=image_size,
            d_model=hidden_channels,
            dropout=dropout,
            num_values=num_values,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            MaskedConv2d("A", in_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    MaskedConv2d("B", hidden_channels, hidden_channels, kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, in_channels * num_values, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        del timesteps
        if x.dim() != 4:
            raise ValueError("PixelCNN expects x with shape (batch, channels, height, width)")
        logits = self.net(x.float())
        batch, _, height, width = logits.shape
        return logits.view(batch, self.in_channels, self.num_values, height, width)

    def predict(self, batch: GenerativeBatch) -> torch.Tensor:
        return self(batch.x_start, batch.timesteps)

    def _targets(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_floating_point():
            x = x.clamp(0, self.num_values - 1).round().long()
        return x.long().clamp(0, self.num_values - 1)

    def loss_step(self, batch: GenerativeBatch) -> torch.Tensor:
        logits = self.predict(batch)
        targets = self._targets(batch.x_start)
        logits = logits.permute(0, 1, 3, 4, 2).reshape(-1, self.num_values)
        targets = targets.reshape(-1)
        return F.cross_entropy(logits, targets)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        targets = self._targets(x)
        log_probs = F.log_softmax(logits, dim=2)
        gathered = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
        return gathered.flatten(1).sum(dim=1)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = torch.device(device) if device is not None else next(self.parameters()).device
        x = torch.zeros(num_samples, self.in_channels, self.image_size, self.image_size, device=device)
        for row in range(self.image_size):
            for col in range(self.image_size):
                logits = self(x)[:, :, :, row, col]
                probs = torch.softmax(logits, dim=-1)
                sample = torch.multinomial(probs.reshape(-1, self.num_values), 1).view(num_samples, self.in_channels)
                x[:, :, row, col] = sample.float()
        return x
