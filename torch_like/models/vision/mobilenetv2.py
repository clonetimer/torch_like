"""MobileNetV2 lightweight image classification backbone.

Framework-oriented implementation of "MobileNetV2: Inverted Residuals and
Linear Bottlenecks". It uses depthwise separable convolutions, inverted
residual blocks, and a width multiplier for efficient vision backbones.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class MobileNetV2Config(VisionConfig):
    width_mult: float = 1.0
    inverted_residual_setting: tuple[tuple[int, int, int, int], ...] = (
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )
    last_channel: int = 1280


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("stride must be 1 or 2")
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 32,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        inverted_residual_setting: tuple[tuple[int, int, int, int], ...] = (
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ),
        last_channel: int = 1280,
    ):
        super().__init__()
        if width_mult <= 0:
            raise ValueError("width_mult must be positive")
        if not inverted_residual_setting:
            raise ValueError("inverted_residual_setting must not be empty")
        for setting in inverted_residual_setting:
            if len(setting) != 4:
                raise ValueError("each inverted residual setting must be (expand_ratio, channels, repeats, stride)")
            expand_ratio, channels, repeats, stride = setting
            if expand_ratio <= 0 or channels <= 0 or repeats <= 0:
                raise ValueError("expand_ratio, channels, and repeats must be positive")
            if stride not in {1, 2}:
                raise ValueError("stage stride must be 1 or 2")

        input_channel = _make_divisible(d_model * width_mult, 8)
        final_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.width_mult = width_mult
        self.feature_dim = final_channel
        self._config = MobileNetV2Config(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            width_mult=width_mult,
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
        )

        features: list[nn.Module] = [ConvBNReLU(in_channels, input_channel, stride=2)]
        for expand_ratio, channels, repeats, stride in inverted_residual_setting:
            output_channel = _make_divisible(channels * width_mult, 8)
            for idx in range(repeats):
                block_stride = stride if idx == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, block_stride, expand_ratio))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, final_channel, stride=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(final_channel, num_classes),
        )
        self._init_weights()

    @classmethod
    def mobilenet_v2(cls, **kwargs) -> "MobileNetV2":
        return cls(**kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("MobileNetV2 expects x with shape (batch, channels, height, width)")
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
