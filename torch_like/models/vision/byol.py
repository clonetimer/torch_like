"""BYOL-style visual representation learner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionBatch, VisionConfig, VisionModel
from torch_like.models.vision.simclr import SimCLR


@dataclass
class BYOLConfig(VisionConfig):
    projection_dim: int = 128
    hidden_dim: int = 256
    momentum: float = 0.99
    encoder_channels: tuple[int, ...] = (32, 64, 128)


class BYOL(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 128,
        dropout: float = 0.0,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        momentum: float = 0.99,
        encoder_channels: tuple[int, ...] = (32, 64, 128),
    ):
        del num_classes
        super().__init__()
        if min(in_channels, image_size, d_model, projection_dim, hidden_dim) <= 0:
            raise ValueError("in_channels, image_size, d_model, projection_dim, and hidden_dim must be positive")
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")
        self.momentum = momentum
        self._config = BYOLConfig(
            in_channels=in_channels,
            num_classes=projection_dim,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            momentum=momentum,
            encoder_channels=tuple(encoder_channels),
        )
        self.online = SimCLR(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            projection_dim=projection_dim,
            encoder_channels=encoder_channels,
        )
        self.target = deepcopy(self.online)
        for param in self.target.parameters():
            param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    @torch.no_grad()
    def update_target(self) -> None:
        for online_param, target_param in zip(self.online.parameters(), self.target.parameters(), strict=True):
            target_param.data.mul_(self.momentum).add_(online_param.data, alpha=1.0 - self.momentum)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.online.encode_image(x)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        return self.online.project(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.encode_image(x))

    def _predict_online(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.predictor(self.online(x)), dim=-1)

    @torch.no_grad()
    def _target_project(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x).detach()

    def byol_loss(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        if view1.shape != view2.shape:
            raise ValueError("view1 and view2 must have the same shape")
        pred1 = self._predict_online(view1)
        pred2 = self._predict_online(view2)
        target1 = self._target_project(view1)
        target2 = self._target_project(view2)
        return 2 - 2 * (pred1 * target2).sum(dim=-1).mean() + 2 - 2 * (pred2 * target1).sum(dim=-1).mean()

    def loss_step(self, batch: VisionBatch | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(batch, tuple):
            return self.byol_loss(batch[0], batch[1])
        if batch.y is None:
            raise ValueError("BYOL loss_step requires batch.y to contain the second augmented view")
        return self.byol_loss(batch.x, batch.y)
