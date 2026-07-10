"""ResNeXt image classification backbone.

Framework-oriented implementation of "Aggregated Residual Transformations for
Deep Neural Networks". ResNeXt generalizes residual bottlenecks with grouped
convolutions, exposing cardinality as a core model capacity dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class ResNeXtConfig(VisionConfig):
    layers: tuple[int, int, int, int] = (3, 4, 6, 3)
    groups: int = 32
    width_per_group: int = 4
    zero_init_residual: bool = False


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def _conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 32,
        width_per_group: int = 4,
    ):
        super().__init__()
        width = int(planes * (width_per_group / 64.0)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
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


class ResNeXt(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.0,
        layers: tuple[int, int, int, int] = (3, 4, 6, 3),
        groups: int = 32,
        width_per_group: int = 4,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must contain four stage depths")
        if any(depth <= 0 for depth in layers):
            raise ValueError("layers must contain positive stage depths")
        if groups <= 0 or width_per_group <= 0:
            raise ValueError("groups and width_per_group must be positive")

        self.inplanes = d_model
        self.num_classes = num_classes
        self.groups = groups
        self.width_per_group = width_per_group
        self._config = ResNeXtConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            layers=layers,
            groups=groups,
            width_per_group=width_per_group,
            zero_init_residual=zero_init_residual,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(d_model, layers[0])
        self.layer2 = self._make_layer(d_model * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(d_model * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(d_model * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.feature_dim = d_model * 8 * ResNeXtBottleneck.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self._init_weights(zero_init_residual)

    @classmethod
    def resnext50_32x4d(cls, **kwargs) -> "ResNeXt":
        return cls(layers=(3, 4, 6, 3), groups=32, width_per_group=4, **kwargs)

    @classmethod
    def resnext101_32x8d(cls, **kwargs) -> "ResNeXt":
        return cls(layers=(3, 4, 23, 3), groups=32, width_per_group=8, **kwargs)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        outplanes = planes * ResNeXtBottleneck.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, outplanes, stride),
                nn.BatchNorm2d(outplanes),
            )
        layers = [
            ResNeXtBottleneck(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                width_per_group=self.width_per_group,
            )
        ]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(
                ResNeXtBottleneck(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )
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
                if isinstance(module, ResNeXtBottleneck):
                    nn.init.zeros_(module.bn3.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("ResNeXt expects x with shape (batch, channels, height, width)")
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))
