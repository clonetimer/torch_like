"""ResNet image classification backbone.

Framework-oriented implementation of residual networks from "Deep Residual
Learning for Image Recognition". It supports common ResNet-18/34 basic-block
and ResNet-50 bottleneck-style depths through a single local implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class ResNetConfig(VisionConfig):
    block: str = "basic"
    layers: tuple[int, int, int, int] = (2, 2, 2, 2)
    zero_init_residual: bool = False


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.0,
        block: str = "basic",
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        zero_init_residual: bool = False,
    ):
        super().__init__()
        if block not in {"basic", "bottleneck"}:
            raise ValueError("block must be 'basic' or 'bottleneck'")
        if len(layers) != 4:
            raise ValueError("layers must contain four stage depths")
        block_cls = BasicBlock if block == "basic" else Bottleneck
        self.inplanes = d_model
        self.num_classes = num_classes
        self._config = ResNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            block=block,
            layers=layers,
            zero_init_residual=zero_init_residual,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block_cls, d_model, layers[0])
        self.layer2 = self._make_layer(block_cls, d_model * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block_cls, d_model * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block_cls, d_model * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(d_model * 8 * block_cls.expansion, num_classes)

        self._init_weights(zero_init_residual)

    @classmethod
    def resnet18(cls, **kwargs) -> "ResNet":
        return cls(block="basic", layers=(2, 2, 2, 2), **kwargs)

    @classmethod
    def resnet34(cls, **kwargs) -> "ResNet":
        return cls(block="basic", layers=(3, 4, 6, 3), **kwargs)

    @classmethod
    def resnet50(cls, **kwargs) -> "ResNet":
        return cls(block="bottleneck", layers=(3, 4, 6, 3), **kwargs)

    def _make_layer(self, block_cls, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block_cls.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block_cls.expansion, stride),
                nn.BatchNorm2d(planes * block_cls.expansion),
            )
        layers = [block_cls(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block_cls.expansion
        for _ in range(1, blocks):
            layers.append(block_cls(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.zeros_(module.bn3.weight)
                elif isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("ResNet expects x with shape (batch, channels, height, width)")
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.fc(self.dropout(features))
