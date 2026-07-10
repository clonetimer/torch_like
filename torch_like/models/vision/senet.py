"""SENet image classification backbone.

Framework-oriented implementation of "Squeeze-and-Excitation Networks".
SENet augments residual bottlenecks with channel attention that adaptively
recalibrates feature responses after global context aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class SENetConfig(VisionConfig):
    layers: tuple[int, int, int, int] = (3, 4, 6, 3)
    reduction: int = 16
    zero_init_residual: bool = False


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class SELayer(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.excitation(self.pool(x))
        return x * scale


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        reduction: int = 16,
    ):
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SENet(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.0,
        layers: tuple[int, int, int, int] = (3, 4, 6, 3),
        reduction: int = 16,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must contain four stage depths")
        if any(depth <= 0 for depth in layers):
            raise ValueError("layers must contain positive stage depths")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        self.inplanes = d_model
        self.num_classes = num_classes
        self.reduction = reduction
        self._config = SENetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            layers=layers,
            reduction=reduction,
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
        self.feature_dim = d_model * 8 * SEBottleneck.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self._init_weights(zero_init_residual)

    @classmethod
    def senet50(cls, **kwargs) -> "SENet":
        return cls(layers=(3, 4, 6, 3), **kwargs)

    @classmethod
    def senet101(cls, **kwargs) -> "SENet":
        return cls(layers=(3, 4, 23, 3), **kwargs)

    @classmethod
    def senet152(cls, **kwargs) -> "SENet":
        return cls(layers=(3, 8, 36, 3), **kwargs)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        outplanes = planes * SEBottleneck.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, outplanes, stride),
                nn.BatchNorm2d(outplanes),
            )
        layers = [
            SEBottleneck(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                reduction=self.reduction,
            )
        ]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(SEBottleneck(self.inplanes, planes, reduction=self.reduction))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, SEBottleneck):
                    nn.init.zeros_(module.bn3.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("SENet expects x with shape (batch, channels, height, width)")
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))
