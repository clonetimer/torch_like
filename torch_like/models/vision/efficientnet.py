"""EfficientNet image classification backbone.

Framework-oriented implementation of EfficientNet-style compound scaling. It
uses MBConv blocks with squeeze-and-excitation and stochastic depth/drop
connect, while keeping the API compact for local model-library usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class EfficientNetConfig(VisionConfig):
    width_mult: float = 1.0
    depth_mult: float = 1.0
    dropout: float = 0.2
    drop_connect_rate: float = 0.2
    se_ratio: float = 0.25
    block_settings: tuple[tuple[int, int, int, int, int], ...] = (
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    )
    last_channel: int = 1280


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _scaled_depth(repeats: int, depth_mult: float) -> int:
    return int(ceil(repeats * depth_mult))


def _drop_connect(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    mask = torch.empty((x.size(0), 1, 1, 1), dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        kernel_size: int,
        se_ratio: float,
        drop_connect_rate: float,
    ):
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("stride must be 1 or 2")
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.drop_connect_rate = drop_connect_rate

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim))
        squeeze_channels = max(1, int(in_channels * se_ratio))
        layers.append(SqueezeExcitation(hidden_dim, squeeze_channels))
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res_connect:
            out = _drop_connect(out, self.drop_connect_rate, self.training)
            return x + out
        return out


class EfficientNet(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 32,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        drop_connect_rate: float = 0.2,
        se_ratio: float = 0.25,
        block_settings: tuple[tuple[int, int, int, int, int], ...] = (
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ),
        last_channel: int = 1280,
    ):
        super().__init__()
        if width_mult <= 0 or depth_mult <= 0:
            raise ValueError("width_mult and depth_mult must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if not 0 <= drop_connect_rate < 1:
            raise ValueError("drop_connect_rate must be in the range [0, 1)")
        if not 0 < se_ratio <= 1:
            raise ValueError("se_ratio must be in the range (0, 1]")
        if not block_settings:
            raise ValueError("block_settings must not be empty")
        for setting in block_settings:
            if len(setting) != 5:
                raise ValueError("each block setting must be (expand_ratio, channels, repeats, stride, kernel_size)")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.width_mult = width_mult
        self.depth_mult = depth_mult
        self.drop_connect_rate = drop_connect_rate
        self._config = EfficientNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            width_mult=width_mult,
            depth_mult=depth_mult,
            drop_connect_rate=drop_connect_rate,
            se_ratio=se_ratio,
            block_settings=block_settings,
            last_channel=last_channel,
        )

        stem_channels = _make_divisible(d_model * width_mult, 8)
        layers: list[nn.Module] = [ConvBNAct(in_channels, stem_channels, stride=2)]
        current_channels = stem_channels
        total_blocks = sum(_scaled_depth(repeats, depth_mult) for _, _, repeats, _, _ in block_settings)
        block_idx = 0
        for expand_ratio, channels, repeats, stride, kernel_size in block_settings:
            out_channels = _make_divisible(channels * width_mult, 8)
            scaled_repeats = _scaled_depth(repeats, depth_mult)
            for repeat_idx in range(scaled_repeats):
                block_stride = stride if repeat_idx == 0 else 1
                block_drop = drop_connect_rate * block_idx / max(1, total_blocks)
                layers.append(
                    MBConv(
                        current_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=block_stride,
                        kernel_size=kernel_size,
                        se_ratio=se_ratio,
                        drop_connect_rate=block_drop,
                    )
                )
                current_channels = out_channels
                block_idx += 1
        final_channels = _make_divisible(last_channel * width_mult, 8)
        layers.append(ConvBNAct(current_channels, final_channels, kernel_size=1))
        self.features = nn.Sequential(*layers)
        self.feature_dim = final_channels
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(final_channels, num_classes),
        )
        self._init_weights()

    @classmethod
    def efficientnet_b0(cls, **kwargs) -> "EfficientNet":
        return cls(width_mult=1.0, depth_mult=1.0, dropout=0.2, **kwargs)

    @classmethod
    def efficientnet_b1(cls, **kwargs) -> "EfficientNet":
        return cls(width_mult=1.0, depth_mult=1.1, dropout=0.2, **kwargs)

    @classmethod
    def efficientnet_b2(cls, **kwargs) -> "EfficientNet":
        return cls(width_mult=1.1, depth_mult=1.2, dropout=0.3, **kwargs)

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
            raise ValueError("EfficientNet expects x with shape (batch, channels, height, width)")
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
