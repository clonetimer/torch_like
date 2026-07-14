"""Mask2Former-style universal segmentation model.

This module exposes a compact model-side interface for universal segmentation:
pixel features, learned mask queries, class logits, mask logits, and semantic
logit composition. Dataset-specific target assignment, Hungarian matching,
panoptic fusion, and benchmark evaluation remain outside the model class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel
from torch_like.models.vision.detr import ConvBNAct


@dataclass
class Mask2FormerConfig(VisionConfig):
    num_queries: int = 100
    backbone_channels: tuple[int, ...] = (64, 128, 256)
    n_heads: int = 8
    decoder_layers: int = 6
    d_ff: int = 2048
    mask_dim: int = 256


class Mask2Former(VisionModel):
    """Compact Mask2Former-style query mask predictor."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 150,
        image_size: int = 224,
        d_model: int = 256,
        dropout: float = 0.1,
        num_queries: int = 100,
        backbone_channels: tuple[int, ...] = (64, 128, 256),
        n_heads: int = 8,
        decoder_layers: int = 6,
        d_ff: int = 2048,
        mask_dim: int = 256,
    ):
        super().__init__()
        if d_model <= 0 or mask_dim <= 0:
            raise ValueError("d_model and mask_dim must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(num_classes, image_size, num_queries, decoder_layers, d_ff) <= 0:
            raise ValueError("Mask2Former dimensions and layer counts must be positive")
        if not backbone_channels or any(channel <= 0 for channel in backbone_channels):
            raise ValueError("backbone_channels must contain positive channels")

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.image_size = image_size
        self._config = Mask2FormerConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            num_queries=num_queries,
            backbone_channels=backbone_channels,
            n_heads=n_heads,
            decoder_layers=decoder_layers,
            d_ff=d_ff,
            mask_dim=mask_dim,
        )

        backbone: list[nn.Module] = []
        current_channels = in_channels
        for idx, out_channels in enumerate(backbone_channels):
            backbone.append(ConvBNAct(current_channels, out_channels, stride=2 if idx > 0 else 1))
            current_channels = out_channels
        self.backbone = nn.Sequential(*backbone)
        self.pixel_proj = nn.Conv2d(current_channels, d_model, kernel_size=1)
        self.mask_feature_proj = nn.Sequential(
            nn.Conv2d(d_model, mask_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, mask_dim),
            nn.GELU(),
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, mask_dim),
        )
        self.feature_dim = d_model
        self._init_weights()

    @classmethod
    def mask2former_tiny(cls, **kwargs) -> "Mask2Former":
        return cls(
            d_model=128,
            n_heads=4,
            decoder_layers=2,
            d_ff=512,
            mask_dim=128,
            backbone_channels=(64, 128),
            **kwargs,
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_pixels(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError("Mask2Former expects x with shape (batch, channels, height, width)")
        features = self.pixel_proj(self.backbone(x))
        mask_features = self.mask_feature_proj(features)
        memory = features.flatten(2).transpose(1, 2)
        return {"features": features, "mask_features": mask_features, "memory": memory}

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode_pixels(x)
        queries = self.query_embed.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        query_features = self.query_norm(self.decoder(queries, encoded["memory"]))
        return {**encoded, "query_features": query_features}

    def predict_masks(
        self,
        query_features: torch.Tensor,
        mask_features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        mask_embed = self.mask_embed(query_features)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        if output_size is not None:
            masks = F.interpolate(masks, size=output_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        query_features = features["query_features"]
        pred_masks = self.predict_masks(query_features, features["mask_features"], output_size=x.shape[-2:])
        pred_logits = self.class_embed(query_features)
        semantic_logits = self.semantic_logits({"pred_logits": pred_logits, "pred_masks": pred_masks})
        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "semantic_logits": semantic_logits,
            "query_features": query_features,
            "mask_features": features["mask_features"],
        }

    def semantic_logits(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        class_probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
        mask_probs = outputs["pred_masks"].sigmoid()
        return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

    def post_process(
        self,
        outputs: dict[str, torch.Tensor],
        score_threshold: float = 0.0,
        top_k: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
        scores, labels = probs.max(dim=-1)
        masks = outputs["pred_masks"].sigmoid()
        detections: list[dict[str, torch.Tensor]] = []
        for sample_scores, sample_labels, sample_masks in zip(scores, labels, masks, strict=True):
            keep = sample_scores >= score_threshold
            sample_scores = sample_scores[keep]
            sample_labels = sample_labels[keep]
            sample_masks = sample_masks[keep]
            if top_k is not None and sample_scores.numel() > top_k:
                values, indices = sample_scores.topk(top_k)
                sample_scores = values
                sample_labels = sample_labels[indices]
                sample_masks = sample_masks[indices]
            detections.append({"scores": sample_scores, "labels": sample_labels, "masks": sample_masks})
        return detections
