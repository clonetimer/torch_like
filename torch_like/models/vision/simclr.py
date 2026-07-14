"""SimCLR-style visual representation model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionBatch, VisionConfig, VisionModel


@dataclass
class SimCLRConfig(VisionConfig):
    projection_dim: int = 128
    temperature: float = 0.5
    encoder_channels: tuple[int, ...] = (32, 64, 128)


class SimCLR(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 128,
        dropout: float = 0.0,
        projection_dim: int = 128,
        temperature: float = 0.5,
        encoder_channels: tuple[int, ...] = (32, 64, 128),
    ):
        del num_classes
        super().__init__()
        if min(in_channels, image_size, d_model, projection_dim) <= 0:
            raise ValueError("in_channels, image_size, d_model, and projection_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not encoder_channels or any(ch <= 0 for ch in encoder_channels):
            raise ValueError("encoder_channels must contain positive channel widths")
        self.in_channels = in_channels
        self.image_size = image_size
        self.temperature = temperature
        self._config = SimCLRConfig(
            in_channels=in_channels,
            num_classes=projection_dim,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            projection_dim=projection_dim,
            temperature=temperature,
            encoder_channels=tuple(encoder_channels),
        )
        layers: list[nn.Module] = []
        current = in_channels
        for channels in encoder_channels:
            layers.extend(
                [
                    nn.Conv2d(current, channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current = channels
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)))
        self.feature_proj = nn.Linear(current, d_model)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, projection_dim),
        )

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("SimCLR expects x with shape (batch, channels, height, width)")
        if x.size(1) != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.size(1)}")
        return self.feature_proj(torch.flatten(self.encoder(x), 1))

    def project(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(features), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.encode_image(x))

    def contrastive_loss(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        if view1.shape != view2.shape:
            raise ValueError("view1 and view2 must have the same shape")
        z1 = self(view1)
        z2 = self(view2)
        z = torch.cat([z1, z2], dim=0)
        logits = z @ z.t() / self.temperature
        batch_size = view1.size(0)
        mask = torch.eye(2 * batch_size, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=logits.device),
                torch.arange(0, batch_size, device=logits.device),
            ]
        )
        return F.cross_entropy(logits, labels)

    def loss_step(self, batch: VisionBatch | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(batch, tuple):
            return self.contrastive_loss(batch[0], batch[1])
        if batch.y is None:
            raise ValueError("SimCLR loss_step requires batch.y to contain the second augmented view")
        return self.contrastive_loss(batch.x, batch.y)
