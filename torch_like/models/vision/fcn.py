"""Fully Convolutional Network for dense prediction.

Framework-oriented implementation of the FCN segmentation pattern from
"Fully Convolutional Networks for Semantic Segmentation": an all-convolutional
encoder maps images to dense feature maps, then a 1x1 classifier and upsampling
produce pixel-level logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class FCNConfig(VisionConfig):
    channels: tuple[int, ...] = (64, 128, 256)
    strides: tuple[int, ...] = (2, 2, 2)


class FCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCN(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.0,
        channels: tuple[int, ...] = (64, 128, 256),
        strides: tuple[int, ...] = (2, 2, 2),
    ):
        super().__init__()
        if not channels:
            raise ValueError("channels must not be empty")
        if len(channels) != len(strides):
            raise ValueError("channels and strides must have the same length")
        if any(channel <= 0 for channel in channels):
            raise ValueError("channels must be positive")
        if any(stride <= 0 for stride in strides):
            raise ValueError("strides must be positive")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.strides = strides
        self._config = FCNConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            channels=channels,
            strides=strides,
        )

        blocks: list[nn.Module] = []
        current_channels = in_channels
        for out_channels, stride in zip(channels, strides, strict=True):
            blocks.append(FCNBlock(current_channels, out_channels, stride=stride))
            current_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(current_channels, num_classes, kernel_size=1)
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
            raise ValueError("FCN expects x with shape (batch, channels, height, width)")
        return self.dropout(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        logits = self.classifier(self.forward_features(x))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
