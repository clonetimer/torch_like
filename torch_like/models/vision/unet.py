"""U-Net dense prediction model.

Framework-oriented implementation of "U-Net: Convolutional Networks for
Biomedical Image Segmentation". It provides an encoder-decoder CNN with
skip connections for segmentation-style pixel predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class UNetConfig(VisionConfig):
    base_channels: int = 64
    depth: int = 4
    bilinear: bool = True


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels - out_channels, in_channels - out_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.0,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear
        self._config = UNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            base_channels=base_channels,
            depth=depth,
            bilinear=bilinear,
        )

        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.inc = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList([Down(channels[idx - 1], channels[idx]) for idx in range(1, depth)])
        self.ups = nn.ModuleList(
            [
                Up(channels[idx + 1] + channels[idx], channels[idx], bilinear=bilinear)
                for idx in range(depth - 2, -1, -1)
            ]
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_conv = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("UNet expects x with shape (batch, channels, height, width)")
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.forward_features(x))
