"""GoogLeNet/Inception image classification backbone.

Framework-oriented implementation of the Inception v1 idea from "Going Deeper
with Convolutions": multi-branch convolutional blocks aggregate 1x1, 3x3, 5x5,
and pooling paths to provide multi-scale features under a controlled compute
budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class GoogLeNetConfig(VisionConfig):
    stem_channels: int = 64
    inception_channels: tuple[tuple[int, int, int, int, int, int], ...] = (
        (64, 96, 128, 16, 32, 32),
        (128, 128, 192, 32, 96, 64),
        (192, 96, 208, 16, 48, 64),
        (160, 112, 224, 24, 64, 64),
    )


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
    ):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_reduce, kernel_size=1),
            BasicConv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5_reduce, kernel_size=1),
            BasicConv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
            ],
            dim=1,
        )


class GoogLeNet(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.2,
        stem_channels: int = 64,
        inception_channels: tuple[tuple[int, int, int, int, int, int], ...] = (
            (64, 96, 128, 16, 32, 32),
            (128, 128, 192, 32, 96, 64),
            (192, 96, 208, 16, 48, 64),
            (160, 112, 224, 24, 64, 64),
        ),
    ):
        super().__init__()
        if stem_channels <= 0:
            raise ValueError("stem_channels must be positive")
        if not inception_channels:
            raise ValueError("inception_channels must not be empty")
        for setting in inception_channels:
            if len(setting) != 6:
                raise ValueError("each inception setting must contain six channel values")
            if any(channel <= 0 for channel in setting):
                raise ValueError("inception channel values must be positive")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.stem_channels = stem_channels
        self.inception_channels = inception_channels
        self._config = GoogLeNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            stem_channels=stem_channels,
            inception_channels=inception_channels,
        )

        self.stem = nn.Sequential(
            BasicConv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(stem_channels, stem_channels, kernel_size=1),
            BasicConv2d(stem_channels, stem_channels * 3, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        blocks: list[nn.Module] = []
        current_channels = stem_channels * 3
        for idx, setting in enumerate(inception_channels):
            block = InceptionBlock(current_channels, *setting)
            blocks.append(block)
            current_channels = block.out_channels
            if idx != len(inception_channels) - 1 and idx % 2 == 1:
                blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception = nn.Sequential(*blocks)
        self.feature_dim = current_channels
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(current_channels, num_classes)
        self._init_weights()

    @classmethod
    def googlenet(cls, **kwargs) -> "GoogLeNet":
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
            raise ValueError("GoogLeNet expects x with shape (batch, channels, height, width)")
        x = self.stem(x)
        x = self.inception(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))
