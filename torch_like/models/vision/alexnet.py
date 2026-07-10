"""AlexNet image classification backbone.

Framework-oriented implementation of "ImageNet Classification with Deep
Convolutional Neural Networks". AlexNet is kept as a foundational CNN baseline
with convolutional feature extraction, ReLU activations, dropout, and an MLP
classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class AlexNetConfig(VisionConfig):
    channels: tuple[int, int, int, int, int] = (64, 192, 384, 256, 256)
    classifier_dim: int = 4096


class AlexNet(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.5,
        channels: tuple[int, int, int, int, int] = (64, 192, 384, 256, 256),
        classifier_dim: int = 4096,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("channels must contain five stage widths")
        if any(channel <= 0 for channel in channels):
            raise ValueError("channels must be positive")
        if classifier_dim <= 0:
            raise ValueError("classifier_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        c1, c2, c3, c4, c5 = channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.classifier_dim = classifier_dim
        self._config = AlexNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            channels=channels,
            classifier_dim=classifier_dim,
        )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.feature_dim = c5 * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_dim, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Linear(classifier_dim, num_classes),
        )
        self._init_weights()

    @classmethod
    def alexnet(cls, **kwargs) -> "AlexNet":
        return cls(**kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("AlexNet expects x with shape (batch, channels, height, width)")
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
