"""Masked Autoencoder (MAE) for self-supervised image representation learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionBatch, VisionConfig, VisionModel
from torch_like.models.vision.vit import PatchEmbedding


@dataclass
class MAEConfig(VisionConfig):
    patch_size: int = 16
    encoder_layers: int = 12
    encoder_heads: int = 12
    decoder_dim: int = 512
    decoder_layers: int = 4
    decoder_heads: int = 8
    mask_ratio: float = 0.75
    norm_pix_loss: bool = False


class MAE(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 0,
        image_size: int = 224,
        d_model: int = 768,
        dropout: float = 0.0,
        patch_size: int = 16,
        encoder_layers: int = 12,
        encoder_heads: int = 12,
        d_ff: int = 3072,
        decoder_dim: int = 512,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_d_ff: int = 2048,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = False,
    ):
        super().__init__()
        del num_classes
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if not 0 <= mask_ratio < 1:
            raise ValueError("mask_ratio must be in [0, 1)")
        if d_model % encoder_heads != 0 or decoder_dim % decoder_heads != 0:
            raise ValueError("encoder and decoder dimensions must be divisible by their head counts")
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self._config = MAEConfig(
            in_channels=in_channels,
            num_classes=0,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            patch_size=patch_size,
            encoder_layers=encoder_layers,
            encoder_heads=encoder_heads,
            decoder_dim=decoder_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            mask_ratio=mask_ratio,
            norm_pix_loss=norm_pix_loss,
        )
        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_size)
        self.encoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, encoder_heads, d_ff, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_embed = nn.Linear(d_model, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            decoder_dim, decoder_heads, decoder_d_ff, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, decoder_layers, enable_nested_tensor=False)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_channels)
        nn.init.trunc_normal_(self.encoder_pos, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = self.image_size // p
        patches = images.reshape(images.size(0), self.in_channels, h, p, w, p)
        return patches.permute(0, 2, 4, 3, 5, 1).reshape(images.size(0), h * w, p * p * self.in_channels)

    def random_masking(self, tokens: torch.Tensor, mask_ratio: float):
        keep = max(1, int(tokens.size(1) * (1 - mask_ratio)))
        noise = torch.rand(tokens.size(0), tokens.size(1), device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        mask = torch.ones(tokens.size(0), tokens.size(1), device=tokens.device)
        mask[:, :keep] = 0
        return visible, torch.gather(mask, 1, ids_restore), ids_restore

    def encode_image(self, images: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        tokens = self.patch_embed(images) + self.encoder_pos
        visible, _, _ = self.random_masking(tokens, mask_ratio) if mask_ratio else (tokens, None, None)
        return self.encoder_norm(self.encoder(visible))

    def forward(self, images: torch.Tensor, mask_ratio: float | None = None):
        if images.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(
                f"MAE expects images with shape (batch, {self.in_channels}, {self.image_size}, {self.image_size})"
            )
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        tokens = self.patch_embed(images) + self.encoder_pos
        visible, mask, ids_restore = self.random_masking(tokens, ratio)
        encoded = self.encoder_norm(self.encoder(visible))
        decoded_visible = self.decoder_embed(encoded)
        mask_tokens = self.mask_token.expand(images.size(0), self.num_patches - decoded_visible.size(1), -1)
        decoded = torch.cat([decoded_visible, mask_tokens], dim=1)
        decoded = torch.gather(decoded, 1, ids_restore.unsqueeze(-1).expand(-1, -1, decoded.size(-1)))
        reconstruction = self.decoder_pred(self.decoder_norm(self.decoder(decoded + self.decoder_pos)))
        return {"reconstruction": reconstruction, "mask": mask, "latent": encoded}

    def fit_step(self, batch: VisionBatch) -> torch.Tensor:
        outputs = self.predict(batch)
        target = self.patchify(batch.x)
        if self.norm_pix_loss:
            target = (target - target.mean(dim=-1, keepdim=True)) / target.var(dim=-1, keepdim=True).add(1e-6).sqrt()
        loss = F.mse_loss(outputs["reconstruction"], target, reduction="none").mean(dim=-1)
        return (loss * outputs["mask"]).sum() / outputs["mask"].sum().clamp_min(1.0)
