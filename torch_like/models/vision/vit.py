"""Vision Transformer image classifier.

Framework-oriented implementation of "An Image is Worth 16x16 Words":
image patches are embedded as tokens, a class token is prepended, and a
Transformer encoder produces the image-level representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class ViTConfig(VisionConfig):
    patch_size: int = 16
    n_heads: int = 8
    e_layers: int = 12
    d_ff: int = 3072
    pool: str = "cls"


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViT(VisionModel):
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
        pool: str = "cls",
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be 'cls' or 'mean'")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.pool = pool
        self.num_patches = (image_size // patch_size) ** 2
        self._config = ViTConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            patch_size=patch_size,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            pool=pool,
        )

        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
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

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def vit_tiny(cls, **kwargs) -> "ViT":
        return cls(d_model=192, n_heads=3, e_layers=12, d_ff=768, **kwargs)

    @classmethod
    def vit_small(cls, **kwargs) -> "ViT":
        return cls(d_model=384, n_heads=6, e_layers=12, d_ff=1536, **kwargs)

    @classmethod
    def vit_base(cls, **kwargs) -> "ViT":
        return cls(d_model=768, n_heads=12, e_layers=12, d_ff=3072, **kwargs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("ViT expects x with shape (batch, channels, height, width)")
        if x.size(-2) != self.image_size or x.size(-1) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}, got {x.size(-2)}x{x.size(-1)}")
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.norm(self.encoder(tokens))
        if self.pool == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        return tokens[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
