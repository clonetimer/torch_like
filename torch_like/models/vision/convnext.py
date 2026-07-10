"""ConvNeXt image classification backbone.

Framework-oriented implementation of "A ConvNet for the 2020s". ConvNeXt
modernizes a ResNet-style convolutional backbone with large-kernel depthwise
convolutions, channels-last MLP blocks, LayerNorm, and optional LayerScale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class ConvNeXtConfig(VisionConfig):
    depths: tuple[int, ...] = (3, 3, 9, 3)
    dims: tuple[int, ...] = (96, 192, 384, 768)
    layer_scale_init_value: float = 1e-6


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6, dropout: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()
        return shortcut + self.drop(x)


class ConvNeXt(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 96,
        dropout: float = 0.0,
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (96, 192, 384, 768),
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        if len(depths) != len(dims):
            raise ValueError("depths and dims must have the same length")
        if not depths:
            raise ValueError("depths and dims must not be empty")
        if any(depth <= 0 for depth in depths):
            raise ValueError("depths must contain positive values")
        if any(dim <= 0 for dim in dims):
            raise ValueError("dims must contain positive values")
        if layer_scale_init_value < 0:
            raise ValueError("layer_scale_init_value must be non-negative")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.depths = depths
        self.dims = dims
        self.feature_dim = dims[-1]
        self._config = ConvNeXtConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            depths=depths,
            dims=dims,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.downsample_layers = nn.ModuleList()
        stem_dim = dims[0]
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, stem_dim, kernel_size=4, stride=4),
                nn.GroupNorm(1, stem_dim),
            )
        )
        for idx in range(len(dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, dims[idx]),
                    nn.Conv2d(dims[idx], dims[idx + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ConvNeXtBlock(dim, layer_scale_init_value=layer_scale_init_value, dropout=dropout)
                        for _ in range(depth)
                    ]
                )
                for dim, depth in zip(dims, depths)
            ]
        )
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self._init_weights()

    @classmethod
    def convnext_tiny(cls, **kwargs) -> "ConvNeXt":
        return cls(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)

    @classmethod
    def convnext_small(cls, **kwargs) -> "ConvNeXt":
        return cls(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768), **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("ConvNeXt expects x with shape (batch, channels, height, width)")
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
        x = x.mean(dim=(-2, -1))
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
