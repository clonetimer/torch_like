"""One-dimensional convolutional block attention for sequence features."""

from __future__ import annotations

import torch
import torch.nn as nn


class CBAM1d(nn.Module):
    """Apply channel attention followed by temporal attention to ``(B, C, L)`` tensors."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )
        self.temporal_conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("CBAM1d expects x with shape (batch, channels, length)")
        average = torch.mean(x, dim=-1, keepdim=True)
        maximum = torch.amax(x, dim=-1, keepdim=True)
        x = x * self.activation(self.channel_mlp(average) + self.channel_mlp(maximum))
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        return x * self.activation(self.temporal_conv(torch.cat((average, maximum), dim=1)))


__all__ = ["CBAM1d"]
