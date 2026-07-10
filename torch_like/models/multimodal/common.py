"""Common multimodal-domain configs and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from torch_like.models.base import BaseModel


@dataclass
class MultimodalConfig:
    vocab_size: int = 49408
    image_size: int = 224
    patch_size: int = 16
    max_text_len: int = 77
    d_model: int = 512
    embed_dim: int = 512
    dropout: float = 0.0


@dataclass
class MultimodalBatch:
    image: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class MultimodalModel(BaseModel):
    """Base class for image-text and general multimodal models."""

    def predict(self, batch: MultimodalBatch) -> Any:
        return self(batch.image, batch.input_ids, attention_mask=batch.attention_mask)

    def make_multimodal_batch(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> MultimodalBatch:
        return MultimodalBatch(image=image, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
