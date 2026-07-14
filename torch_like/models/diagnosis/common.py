"""Common diagnosis configs and model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torch_like.models.base import BaseModel


@dataclass
class DiagnosisConfig:
    input_dim: int
    num_classes: int
    seq_len: int
    d_model: int = 256
    dropout: float = 0.1


@dataclass
class DiagnosisBatch:
    x: torch.Tensor
    x_mark: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class DiagnosisModel(BaseModel):
    """Base class for diagnosis models."""

    def predict(self, batch: DiagnosisBatch) -> torch.Tensor:
        return self(batch.x, batch.x_mark)

    def loss_step(self, batch: DiagnosisBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("DiagnosisBatch.labels is required for loss_step")
        return F.cross_entropy(self.predict(batch), batch.labels)

    def make_diagnosis_batch(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> DiagnosisBatch:
        if x_mark is None:
            x_mark = torch.zeros(x.size(0), x.size(1), 4, device=x.device, dtype=x.dtype)
        return DiagnosisBatch(x=x, x_mark=x_mark, labels=labels)
