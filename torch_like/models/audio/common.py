"""Common audio-domain configs and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torch_like.models.base import BaseModel


@dataclass
class AudioConfig:
    input_dim: int = 80
    output_dim: int = 1000
    d_model: int = 256
    dropout: float = 0.1


@dataclass
class AudioBatch:
    x: torch.Tensor
    lengths: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None


class AudioModel(BaseModel):
    """Base class for speech/audio representation and prediction models."""

    def predict(self, batch: AudioBatch) -> torch.Tensor:
        return self(batch.x, lengths=batch.lengths)

    def make_audio_batch(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> AudioBatch:
        return AudioBatch(x=x, lengths=lengths, y=y)

    def loss_step(self, batch: AudioBatch) -> torch.Tensor:
        if batch.y is None:
            raise ValueError("loss_step requires batch.y")
        logits = self.predict(batch)
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            target = batch.y.reshape(-1).long()
            return F.cross_entropy(logits, target)
        if logits.size(-1) > 1 and batch.y.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            return F.cross_entropy(logits, batch.y.long())
        target = batch.y.to(device=logits.device, dtype=logits.dtype).reshape_as(logits)
        return F.mse_loss(logits, target)
