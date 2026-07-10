"""DeepLabV3 semantic segmentation model.

Framework-oriented implementation of DeepLabV3-style atrous spatial pyramid
pooling. The local backbone is intentionally compact, while the public API
exposes the core segmentation pattern: dilated feature extraction, ASPP, and
full-resolution dense logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class DeepLabV3Config(VisionConfig):
    backbone_channels: tuple[int, ...] = (32, 64, 128)
    aspp_channels: int = 128
    atrous_rates: tuple[int, ...] = (6, 12, 18)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = ConvBNReLU(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.proj(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: tuple[int, ...]):
        super().__init__()
        branches: list[nn.Module] = [ConvBNReLU(in_channels, out_channels, kernel_size=1)]
        branches.extend(ConvBNReLU(in_channels, out_channels, dilation=rate) for rate in atrous_rates)
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)
        self.project = ConvBNReLU(out_channels * len(branches), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class DeepLabV3(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.1,
        backbone_channels: tuple[int, ...] = (32, 64, 128),
        aspp_channels: int = 128,
        atrous_rates: tuple[int, ...] = (6, 12, 18),
    ):
        super().__init__()
        if len(backbone_channels) < 2:
            raise ValueError("backbone_channels must contain at least two stages")
        if aspp_channels <= 0:
            raise ValueError("aspp_channels must be positive")
        if not atrous_rates:
            raise ValueError("atrous_rates must not be empty")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.backbone_channels = backbone_channels
        self.aspp_channels = aspp_channels
        self.atrous_rates = atrous_rates
        self._config = DeepLabV3Config(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            backbone_channels=backbone_channels,
            aspp_channels=aspp_channels,
            atrous_rates=atrous_rates,
        )

        stages: list[nn.Module] = []
        current_channels = in_channels
        for idx, out_channels in enumerate(backbone_channels):
            stride = 2 if idx < 2 else 1
            dilation = 1 if idx < 2 else 2 ** (idx - 1)
            stages.append(
                nn.Sequential(
                    ConvBNReLU(current_channels, out_channels, stride=stride, dilation=dilation),
                    ConvBNReLU(out_channels, out_channels, dilation=dilation),
                )
            )
            current_channels = out_channels
        self.backbone = nn.Sequential(*stages)
        self.aspp = ASPP(current_channels, aspp_channels, atrous_rates)
        self.classifier = nn.Sequential(
            ConvBNReLU(aspp_channels, aspp_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(aspp_channels, num_classes, kernel_size=1),
        )
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
            raise ValueError("DeepLabV3 expects x with shape (batch, channels, height, width)")
        return self.aspp(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        logits = self.classifier(self.forward_features(x))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
