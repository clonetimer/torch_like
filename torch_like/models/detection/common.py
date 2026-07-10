"""Common detection configs and model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torch_like.models.base import BaseModel


@dataclass
class DetectionConfig:
    input_dim: int
    seq_len: int
    d_model: int = 256
    dropout: float = 0.1
    threshold: float = 0.5


@dataclass
class DetectionBatch:
    x: torch.Tensor
    x_mark: Optional[torch.Tensor] = None


class DetectionModel(BaseModel):
    """Base class for detection models."""

    def predict(self, batch: DetectionBatch) -> torch.Tensor:
        return self(batch.x, batch.x_mark)

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        """Return anomaly scores for one batch.

        Convention:
        - input batch carries full windows `(B, L, C)`
        - model predicts reconstruction for the last timestamp `(B, C)` unless overridden
        - score is reconstruction error against the last timestamp
        """
        pred = self.predict(batch)
        target = batch.x[:, -1, :]
        score = (pred - target) ** 2
        if reduction == "none":
            return score
        if reduction == "sum":
            return score.sum(dim=-1)
        return score.mean(dim=-1)

    def fit_step(self, batch: DetectionBatch):
        """Compute the training loss for one batch.

        Convention:
        - reconstruction-based detectors train against the last timestamp
        - subclasses can override if they use forecasting-style or other
          sequence-level objectives
        """
        pred = self.predict(batch)
        target = batch.x[:, -1, :]
        return torch.mean((pred - target) ** 2)

    def make_detection_batch(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> DetectionBatch:
        if x_mark is None:
            x_mark = torch.zeros(x.size(0), x.size(1), 4, device=x.device, dtype=x.dtype)
        return DetectionBatch(x=x, x_mark=x_mark)
