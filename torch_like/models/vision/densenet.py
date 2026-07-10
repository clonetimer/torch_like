"""DenseNet image classification backbone.

Framework-oriented implementation of "Densely Connected Convolutional
Networks". Each dense block concatenates all previous feature maps, encouraging
feature reuse and short gradient paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class DenseNetConfig(VisionConfig):
    growth_rate: int = 32
    block_layers: tuple[int, ...] = (6, 12, 24, 16)
    num_init_features: int = 64
    bn_size: int = 4
    compression: float = 0.5


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int, dropout: float = 0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        return self.dropout(out)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(in_channels + idx * growth_rate, growth_rate, bn_size, dropout=dropout)
                for idx in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNet(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.0,
        growth_rate: int = 32,
        block_layers: tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
    ):
        super().__init__()
        if growth_rate <= 0:
            raise ValueError("growth_rate must be positive")
        if not block_layers:
            raise ValueError("block_layers must not be empty")
        if any(num_layers <= 0 for num_layers in block_layers):
            raise ValueError("block_layers must contain positive layer counts")
        if num_init_features <= 0:
            raise ValueError("num_init_features must be positive")
        if bn_size <= 0:
            raise ValueError("bn_size must be positive")
        if not 0 < compression <= 1:
            raise ValueError("compression must be in the range (0, 1]")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.growth_rate = growth_rate
        self.block_layers = block_layers
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.compression = compression
        self._config = DenseNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            growth_rate=growth_rate,
            block_layers=block_layers,
            num_init_features=num_init_features,
            bn_size=bn_size,
            compression=compression,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        blocks: list[nn.Module] = []
        num_features = num_init_features
        for idx, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size, dropout=dropout)
            blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if idx != len(block_layers) - 1:
                out_features = max(1, int(num_features * compression))
                blocks.append(Transition(num_features, out_features))
                num_features = out_features
        self.features = nn.Sequential(*blocks)
        self.norm = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self.feature_dim = num_features
        self._init_weights()

    @classmethod
    def densenet121(cls, **kwargs) -> "DenseNet":
        return cls(growth_rate=32, block_layers=(6, 12, 24, 16), num_init_features=64, **kwargs)

    @classmethod
    def densenet169(cls, **kwargs) -> "DenseNet":
        return cls(growth_rate=32, block_layers=(6, 12, 32, 32), num_init_features=64, **kwargs)

    @classmethod
    def densenet201(cls, **kwargs) -> "DenseNet":
        return cls(growth_rate=32, block_layers=(6, 12, 48, 32), num_init_features=64, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("DenseNet expects x with shape (batch, channels, height, width)")
        x = self.stem(x)
        x = self.features(x)
        x = F.relu(self.norm(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
