"""MLP-Mixer image classification backbone.

Framework-oriented implementation of "MLP-Mixer: An all-MLP Architecture for
Vision". Images are split into patches, then Mixer blocks alternate token-mixing
MLPs across patches and channel-mixing MLPs across hidden channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class MLPMixerConfig(VisionConfig):
    patch_size: int = 16
    n_layers: int = 8
    token_dim: int = 256
    channel_dim: int = 2048


class MixerMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, num_patches: int, d_model: int, token_dim: int, channel_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mlp = MixerMLP(num_patches, token_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mlp = MixerMLP(d_model, channel_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        return x + self.channel_mlp(self.norm2(x))


class MLPMixer(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 512,
        dropout: float = 0.0,
        patch_size: int = 16,
        n_layers: int = 8,
        token_dim: int = 256,
        channel_dim: int = 2048,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model <= 0 or patch_size <= 0 or n_layers <= 0:
            raise ValueError("d_model, patch_size, and n_layers must be positive")
        if token_dim <= 0 or channel_dim <= 0:
            raise ValueError("token_dim and channel_dim must be positive")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.feature_dim = d_model
        self._config = MLPMixerConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            patch_size=patch_size,
            n_layers=n_layers,
            token_dim=token_dim,
            channel_dim=channel_dim,
        )

        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.Sequential(
            *[MixerBlock(self.num_patches, d_model, token_dim, channel_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    @classmethod
    def mixer_s16(cls, **kwargs) -> "MLPMixer":
        return cls(patch_size=16, d_model=512, n_layers=8, token_dim=256, channel_dim=2048, **kwargs)

    @classmethod
    def mixer_b16(cls, **kwargs) -> "MLPMixer":
        return cls(patch_size=16, d_model=768, n_layers=12, token_dim=384, channel_dim=3072, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("MLPMixer expects x with shape (batch, channels, height, width)")
        if x.size(-2) != self.image_size or x.size(-1) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}, got {x.size(-2)}x{x.size(-1)}")
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
