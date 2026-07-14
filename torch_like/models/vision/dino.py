"""DINO-style teacher-student visual self-supervision."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionBatch, VisionConfig, VisionModel
from torch_like.models.vision.vit import ViT


@dataclass
class DINOConfig(VisionConfig):
    patch_size: int = 8
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    projection_dim: int = 128
    hidden_dim: int = 256
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    momentum: float = 0.996


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINO(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 128,
        dropout: float = 0.0,
        patch_size: int = 8,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        momentum: float = 0.996,
    ):
        del num_classes
        super().__init__()
        if min(in_channels, image_size, d_model, patch_size, n_heads, e_layers, d_ff, projection_dim, hidden_dim) <= 0:
            raise ValueError("all size parameters must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 < teacher_temp and 0 < student_temp:
            raise ValueError("teacher_temp and student_temp must be positive")
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.momentum = momentum
        self._config = DINOConfig(
            in_channels=in_channels,
            num_classes=projection_dim,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            patch_size=patch_size,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            momentum=momentum,
        )
        self.student = ViT(
            in_channels=in_channels,
            num_classes=projection_dim,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            patch_size=patch_size,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            pool="cls",
        )
        self.teacher = deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.student_head = DINOHead(d_model, hidden_dim, projection_dim)
        self.teacher_head = deepcopy(self.student_head)
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        self.register_buffer("center", torch.zeros(1, projection_dim))

    @torch.no_grad()
    def update_teacher(self) -> None:
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters(), strict=True):
            teacher_param.data.mul_(self.momentum).add_(student_param.data, alpha=1.0 - self.momentum)
        for student_param, teacher_param in zip(
            self.student_head.parameters(), self.teacher_head.parameters(), strict=True
        ):
            teacher_param.data.mul_(self.momentum).add_(student_param.data, alpha=1.0 - self.momentum)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.student.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student_head(self.encode_image(x))

    @torch.no_grad()
    def teacher_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher_head(self.teacher.forward_features(x))

    def dino_loss(self, student_view: torch.Tensor, teacher_view: torch.Tensor) -> torch.Tensor:
        student_logits = self(student_view) / self.student_temp
        with torch.no_grad():
            teacher_probs = F.softmax((self.teacher_logits(teacher_view) - self.center) / self.teacher_temp, dim=-1)
        log_probs = F.log_softmax(student_logits, dim=-1)
        return -(teacher_probs * log_probs).sum(dim=-1).mean()

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor, momentum: float = 0.9) -> None:
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)

    def loss_step(self, batch: VisionBatch | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(batch, tuple):
            return self.dino_loss(batch[0], batch[1])
        if batch.y is None:
            raise ValueError("DINO loss_step requires batch.y to contain the teacher/global view")
        return self.dino_loss(batch.x, batch.y)
