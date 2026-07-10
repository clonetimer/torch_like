"""Common vision-domain configs and model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from torch_like.models.base import BaseModel


@dataclass
class VisionConfig:
    in_channels: int = 3
    num_classes: int = 1000
    image_size: int = 224
    d_model: int = 64
    dropout: float = 0.0


@dataclass
class VisionBatch:
    x: torch.Tensor
    y: Optional[torch.Tensor] = None


class VisionModel(BaseModel):
    """Base class for image classification, dense prediction, and detection models."""

    def predict(self, batch: VisionBatch) -> Any:
        return self(batch.x)

    def make_vision_batch(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> VisionBatch:
        return VisionBatch(x=x, y=y)
