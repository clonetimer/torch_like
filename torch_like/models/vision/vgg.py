"""VGG image classification backbone.

Framework-oriented implementation of "Very Deep Convolutional Networks for
Large-Scale Image Recognition". VGG uses repeated small 3x3 convolutions and
max-pooling stages to scale CNN depth in a simple, reusable backbone pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class VGGConfig(VisionConfig):
    cfg: tuple[int | str, ...] = (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M")
    classifier_dim: int = 4096
    batch_norm: bool = False


_VGG_CONFIGS: dict[str, tuple[int | str, ...]] = {
    "A": (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    "D": (64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"),
    "E": (64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"),
}


def _make_layers(cfg: tuple[int | str, ...], in_channels: int, batch_norm: bool) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    current_channels = in_channels
    for item in cfg:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        if not isinstance(item, int) or item <= 0:
            raise ValueError("cfg entries must be positive integers or 'M'")
        conv = nn.Conv2d(current_channels, item, kernel_size=3, padding=1)
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(item), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv, nn.ReLU(inplace=True)])
        current_channels = item
    return nn.Sequential(*layers), current_channels


class VGG(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 64,
        dropout: float = 0.5,
        cfg: tuple[int | str, ...] = _VGG_CONFIGS["D"],
        classifier_dim: int = 4096,
        batch_norm: bool = False,
    ):
        super().__init__()
        if not cfg:
            raise ValueError("cfg must not be empty")
        if classifier_dim <= 0:
            raise ValueError("classifier_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.cfg = cfg
        self.classifier_dim = classifier_dim
        self.batch_norm = batch_norm
        self._config = VGGConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            cfg=cfg,
            classifier_dim=classifier_dim,
            batch_norm=batch_norm,
        )

        self.features, feature_channels = _make_layers(cfg, in_channels, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_dim = feature_channels * 7 * 7
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_dim, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_dim, num_classes),
        )
        self._init_weights()

    @classmethod
    def vgg11(cls, **kwargs) -> "VGG":
        return cls(cfg=_VGG_CONFIGS["A"], **kwargs)

    @classmethod
    def vgg16(cls, **kwargs) -> "VGG":
        return cls(cfg=_VGG_CONFIGS["D"], **kwargs)

    @classmethod
    def vgg19(cls, **kwargs) -> "VGG":
        return cls(cfg=_VGG_CONFIGS["E"], **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("VGG expects x with shape (batch, channels, height, width)")
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
