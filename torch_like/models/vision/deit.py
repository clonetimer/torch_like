"""DeiT image classification backbone.

Framework-oriented implementation of "Training data-efficient image
transformers & distillation through attention". DeiT extends ViT with a
distillation token and an optional distillation head while keeping inference
compatible with the standard vision model API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel
from torch_like.models.vision.vit import PatchEmbedding


@dataclass
class DeiTConfig(VisionConfig):
    patch_size: int = 16
    n_heads: int = 12
    e_layers: int = 12
    d_ff: int = 3072
    distillation: bool = True
    output_mode: str = "average"


class DeiT(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 768,
        dropout: float = 0.0,
        patch_size: int = 16,
        n_heads: int = 12,
        e_layers: int = 12,
        d_ff: int = 3072,
        distillation: bool = True,
        output_mode: str = "average",
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if output_mode not in {"average", "cls", "dist"}:
            raise ValueError("output_mode must be 'average', 'cls', or 'dist'")
        if output_mode == "dist" and not distillation:
            raise ValueError("output_mode='dist' requires distillation=True")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.distillation = distillation
        self.output_mode = output_mode
        self.num_patches = (image_size // patch_size) ** 2
        token_count = self.num_patches + 2 if distillation else self.num_patches + 1
        self._config = DeiTConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            patch_size=patch_size,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            distillation=distillation,
            output_mode=output_mode,
        )

        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model)) if distillation else None
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, d_model))
        self.pos_drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dist_head = nn.Linear(d_model, num_classes) if distillation else None
        self.feature_dim = d_model
        self._init_weights()

    @classmethod
    def deit_tiny(cls, **kwargs) -> "DeiT":
        return cls(d_model=192, n_heads=3, e_layers=12, d_ff=768, **kwargs)

    @classmethod
    def deit_small(cls, **kwargs) -> "DeiT":
        return cls(d_model=384, n_heads=6, e_layers=12, d_ff=1536, **kwargs)

    @classmethod
    def deit_base(cls, **kwargs) -> "DeiT":
        return cls(d_model=768, n_heads=12, e_layers=12, d_ff=3072, **kwargs)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.dist_head is not None:
            nn.init.trunc_normal_(self.dist_head.weight, std=0.02)
            nn.init.zeros_(self.dist_head.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("DeiT expects x with shape (batch, channels, height, width)")
        if x.size(-2) != self.image_size or x.size(-1) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}, got {x.size(-2)}x{x.size(-1)}")
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        if self.distillation:
            dist = self.dist_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, dist, tokens], dim=1)
        else:
            tokens = torch.cat([cls, tokens], dim=1)
        return self.norm(self.encoder(self.pos_drop(tokens + self.pos_embed)))

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = self._encode(x)
        cls_feature = tokens[:, 0, :]
        dist_feature = tokens[:, 1, :] if self.distillation else None
        return cls_feature, dist_feature

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        cls_feature, dist_feature = self.forward_tokens(x)
        if self.output_mode == "dist":
            if dist_feature is None:
                raise RuntimeError("distillation token is unavailable")
            return dist_feature
        if self.output_mode == "average" and dist_feature is not None:
            return 0.5 * (cls_feature + dist_feature)
        return cls_feature

    def forward_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        cls_feature, dist_feature = self.forward_tokens(x)
        cls_logits = self.head(cls_feature)
        dist_logits = self.dist_head(dist_feature) if self.dist_head is not None and dist_feature is not None else None
        return cls_logits, dist_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_logits, dist_logits = self.forward_logits(x)
        if self.output_mode == "dist":
            if dist_logits is None:
                raise RuntimeError("distillation logits are unavailable")
            return dist_logits
        if self.output_mode == "average" and dist_logits is not None:
            return 0.5 * (cls_logits + dist_logits)
        return cls_logits
