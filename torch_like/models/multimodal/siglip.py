"""SigLIP image-text dual encoder with pairwise sigmoid loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.multimodal.clip import CLIP
from torch_like.models.multimodal.common import MultimodalConfig


@dataclass
class SigLIPConfig(MultimodalConfig):
    in_channels: int = 3
    vision_layers: int = 12
    vision_heads: int = 8
    text_layers: int = 12
    text_heads: int = 8
    d_ff: int = 2048
    pad_token_id: int = 0
    logit_bias_init: float = -10.0


class SigLIP(CLIP):
    def __init__(
        self,
        vocab_size: int = 49408,
        image_size: int = 224,
        patch_size: int = 16,
        max_text_len: int = 64,
        in_channels: int = 3,
        d_model: int = 768,
        embed_dim: int = 768,
        dropout: float = 0.0,
        vision_layers: int = 12,
        vision_heads: int = 12,
        text_layers: int = 12,
        text_heads: int = 12,
        d_ff: int = 3072,
        pad_token_id: int = 0,
        logit_bias_init: float = -10.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            image_size=image_size,
            patch_size=patch_size,
            max_text_len=max_text_len,
            in_channels=in_channels,
            d_model=d_model,
            embed_dim=embed_dim,
            dropout=dropout,
            vision_layers=vision_layers,
            vision_heads=vision_heads,
            text_layers=text_layers,
            text_heads=text_heads,
            d_ff=d_ff,
            pad_token_id=pad_token_id,
        )
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias_init)))
        self._config = SigLIPConfig(
            vocab_size=vocab_size,
            image_size=image_size,
            patch_size=patch_size,
            max_text_len=max_text_len,
            in_channels=in_channels,
            d_model=d_model,
            embed_dim=embed_dim,
            dropout=dropout,
            vision_layers=vision_layers,
            vision_heads=vision_heads,
            text_layers=text_layers,
            text_heads=text_heads,
            d_ff=d_ff,
            pad_token_id=pad_token_id,
            logit_bias_init=logit_bias_init,
        )

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        image_embeds = self.encode_image(image)
        text_embeds = self.encode_text(input_ids, attention_mask)
        logits = self.logit_scale.exp().clamp(max=100) * image_embeds @ text_embeds.t() + self.logit_bias
        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits_per_image": logits,
            "logits_per_text": logits.t(),
        }

    def loss_step(self, batch) -> torch.Tensor:
        logits = self.predict(batch)["logits_per_image"]
        labels = 2 * torch.eye(logits.size(0), device=logits.device, dtype=logits.dtype) - 1
        return -F.logsigmoid(labels * logits).mean()
