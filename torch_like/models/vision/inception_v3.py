"""Inception-v3 image classification backbone.

Framework-oriented implementation of "Rethinking the Inception Architecture
for Computer Vision". The model exposes factorized Inception blocks and grid
reduction stages while keeping the public API aligned with other vision models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class InceptionV3Config(VisionConfig):
    stem_channels: int = 32
    a_blocks: int = 3
    b_blocks: int = 4
    c_blocks: int = 2


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        **kwargs,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InceptionA(nn.Module):
    """Inception-v3 block with factorized 5x5 as stacked 3x3 convolutions."""

    def __init__(self, in_channels: int, branch_channels: int):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, branch_channels, kernel_size=1)
        self.branch5 = nn.Sequential(
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
        )
        self.branch3dbl = nn.Sequential(
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
        )
        self.out_channels = branch_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch5(x), self.branch3dbl(x), self.branch_pool(x)],
            dim=1,
        )


class ReductionA(nn.Module):
    """Grid-size reduction block from the Inception-v3 family."""

    def __init__(self, in_channels: int, branch_channels: int):
        super().__init__()
        out_channels = branch_channels * 2
        self.branch3 = BasicConv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.branch3dbl = nn.Sequential(
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            BasicConv2d(branch_channels, out_channels, kernel_size=3, stride=2),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.out_channels = out_channels * 2 + in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3(x), self.branch3dbl(x), self.branch_pool(x)], dim=1)


class InceptionB(nn.Module):
    """Inception-v3 block using factorized 7x7 convolutions."""

    def __init__(self, in_channels: int, branch_channels: int):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, branch_channels, kernel_size=1)
        self.branch7 = nn.Sequential(
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch7dbl = nn.Sequential(
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
        )
        self.out_channels = branch_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch7(x), self.branch7dbl(x), self.branch_pool(x)],
            dim=1,
        )


class ReductionB(nn.Module):
    """Second grid-size reduction block with factorized 7x7 preprocessing."""

    def __init__(self, in_channels: int, branch_channels: int):
        super().__init__()
        out_channels = branch_channels * 2
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
            BasicConv2d(branch_channels, out_channels, kernel_size=3, stride=2),
        )
        self.branch7x3 = nn.Sequential(
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(branch_channels, out_channels, kernel_size=3, stride=2),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.out_channels = out_channels * 2 + in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3(x), self.branch7x3(x), self.branch_pool(x)], dim=1)


class InceptionC(nn.Module):
    """Inception-v3 high-level block with split asymmetric branches."""

    def __init__(self, in_channels: int, branch_channels: int):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, branch_channels, kernel_size=1)

        self.branch3_base = BasicConv2d(in_channels, branch_channels, kernel_size=1)
        self.branch3_1 = BasicConv2d(branch_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1))
        self.branch3_2 = BasicConv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0))

        self.branch3dbl_base = nn.Sequential(
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(branch_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch3dbl_1 = BasicConv2d(branch_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1))
        self.branch3dbl_2 = BasicConv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, branch_channels, kernel_size=1),
        )
        self.out_channels = branch_channels * 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch3 = self.branch3_base(x)
        branch3 = torch.cat([self.branch3_1(branch3), self.branch3_2(branch3)], dim=1)

        branch3dbl = self.branch3dbl_base(x)
        branch3dbl = torch.cat([self.branch3dbl_1(branch3dbl), self.branch3dbl_2(branch3dbl)], dim=1)

        return torch.cat([self.branch1(x), branch3, branch3dbl, self.branch_pool(x)], dim=1)


class InceptionV3(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 299,
        d_model: int = 64,
        dropout: float = 0.2,
        stem_channels: int = 32,
        a_blocks: int = 3,
        b_blocks: int = 4,
        c_blocks: int = 2,
    ):
        super().__init__()
        if d_model <= 0 or stem_channels <= 0:
            raise ValueError("d_model and stem_channels must be positive")
        if min(a_blocks, b_blocks, c_blocks) <= 0:
            raise ValueError("a_blocks, b_blocks, and c_blocks must be positive")

        self.num_classes = num_classes
        self._config = InceptionV3Config(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            stem_channels=stem_channels,
            a_blocks=a_blocks,
            b_blocks=b_blocks,
            c_blocks=c_blocks,
        )

        self.stem = nn.Sequential(
            BasicConv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            BasicConv2d(stem_channels, stem_channels, kernel_size=3, padding=1),
            BasicConv2d(stem_channels, stem_channels * 2, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        blocks: list[nn.Module] = []
        current_channels = stem_channels * 2
        for _ in range(a_blocks):
            block = InceptionA(current_channels, d_model)
            blocks.append(block)
            current_channels = block.out_channels

        reduction_a = ReductionA(current_channels, d_model)
        blocks.append(reduction_a)
        current_channels = reduction_a.out_channels

        for _ in range(b_blocks):
            block = InceptionB(current_channels, d_model)
            blocks.append(block)
            current_channels = block.out_channels

        reduction_b = ReductionB(current_channels, d_model)
        blocks.append(reduction_b)
        current_channels = reduction_b.out_channels

        for _ in range(c_blocks):
            block = InceptionC(current_channels, d_model)
            blocks.append(block)
            current_channels = block.out_channels

        self.inception = nn.Sequential(*blocks)
        self.feature_dim = current_channels
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self._init_weights()

    @classmethod
    def inception_v3(cls, **kwargs) -> "InceptionV3":
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
            raise ValueError("InceptionV3 expects x with shape (batch, channels, height, width)")
        x = self.stem(x)
        x = self.inception(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))
