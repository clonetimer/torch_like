"""RT-DETRv2-style real-time end-to-end object detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel
from torch_like.models.vision.detr import MLP, PositionEmbeddingSine


@dataclass
class RTDETRConfig(VisionConfig):
    num_queries: int = 300
    backbone_channels: tuple[int, ...] = (64, 128, 256)
    n_heads: int = 8
    encoder_layers: int = 1
    decoder_layers: int = 3
    d_ff: int = 1024


class _ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _HybridEncoder(nn.Module):
    """AIFI on the deepest feature plus lightweight cross-scale fusion."""

    def __init__(self, channels, d_model, n_heads, layers, d_ff, dropout):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(channel, d_model, 1) for channel in channels])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.aifi = nn.TransformerEncoder(encoder_layer, layers)
        self.position = PositionEmbeddingSine(d_model)
        self.top_down = nn.ModuleList([_ConvNormAct(d_model * 2, d_model) for _ in range(len(channels) - 1)])

    def forward(self, features):
        projected = [projection(feature) for projection, feature in zip(self.projections, features, strict=True)]
        deepest = projected[-1]
        tokens = (deepest + self.position(deepest)).flatten(2).transpose(1, 2)
        projected[-1] = self.aifi(tokens).transpose(1, 2).reshape_as(deepest)
        for index in range(len(projected) - 2, -1, -1):
            upsampled = F.interpolate(projected[index + 1], size=projected[index].shape[-2:], mode="nearest")
            fusion = self.top_down[len(projected) - 2 - index]
            projected[index] = fusion(torch.cat([projected[index], upsampled], dim=1))
        return projected


class RTDETR(VisionModel):
    """Portable RT-DETRv2 reference with confidence-based query selection."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
        image_size: int = 640,
        d_model: int = 256,
        dropout: float = 0.0,
        num_queries: int = 300,
        backbone_channels: tuple[int, ...] = (64, 128, 256),
        n_heads: int = 8,
        encoder_layers: int = 1,
        decoder_layers: int = 3,
        d_ff: int = 1024,
    ):
        super().__init__()
        if min(in_channels, num_classes, image_size, d_model, num_queries, n_heads, encoder_layers, decoder_layers, d_ff) <= 0:
            raise ValueError("RTDETR dimensions and layer counts must be positive")
        if d_model % n_heads != 0 or d_model % 4 != 0:
            raise ValueError("d_model must be divisible by n_heads and by 4")
        if len(backbone_channels) < 2 or any(channel <= 0 for channel in backbone_channels):
            raise ValueError("backbone_channels must contain at least two positive values")
        self.num_classes = num_classes
        self.num_queries = num_queries
        self._config = RTDETRConfig(
            in_channels, num_classes, image_size, d_model, dropout, num_queries,
            backbone_channels, n_heads, encoder_layers, decoder_layers, d_ff,
        )
        stages = []
        current = in_channels
        for channel in backbone_channels:
            stages.append(nn.Sequential(_ConvNormAct(current, channel, 2), _ConvNormAct(channel, channel)))
            current = channel
        self.backbone = nn.ModuleList(stages)
        self.encoder = _HybridEncoder(backbone_channels, d_model, n_heads, encoder_layers, d_ff, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.encoder_class = nn.Linear(d_model, num_classes)
        self.encoder_box = MLP(d_model, d_model, 4, 3)
        self.query_embedding = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, num_classes)
        self.box_head = MLP(d_model, d_model, 4, 3)
        self.feature_dim = d_model

    def _backbone_features(self, x):
        if x.dim() != 4:
            raise ValueError("RTDETR expects x with shape (batch, channels, height, width)")
        features = []
        for stage in self.backbone:
            x = stage(x)
            features.append(x)
        return features

    def encode_image(self, x):
        features = self.encoder(self._backbone_features(x))
        return torch.cat([feature.flatten(2).transpose(1, 2) for feature in features], dim=1)

    def select_queries(self, memory):
        encoder_logits = self.encoder_class(memory)
        encoder_boxes = self.encoder_box(memory).sigmoid()
        confidence = encoder_logits.sigmoid().amax(dim=-1)
        count = min(self.num_queries, memory.size(1))
        indices = confidence.topk(count, dim=1).indices
        gather_features = indices.unsqueeze(-1).expand(-1, -1, memory.size(-1))
        gather_boxes = indices.unsqueeze(-1).expand(-1, -1, 4)
        selected = memory.gather(1, gather_features)
        boxes = encoder_boxes.gather(1, gather_boxes)
        if count < self.num_queries:
            padding = self.query_embedding.weight[count:].unsqueeze(0).expand(memory.size(0), -1, -1)
            selected = torch.cat([selected, padding], dim=1)
            boxes = torch.cat([boxes, boxes.new_zeros(memory.size(0), self.num_queries - count, 4)], dim=1)
        return selected + self.query_embedding.weight.unsqueeze(0), boxes, encoder_logits, encoder_boxes, indices

    def forward_features(self, x):
        memory = self.encode_image(x)
        queries, reference_boxes, encoder_logits, encoder_boxes, indices = self.select_queries(memory)
        decoded = self.decoder(queries, memory)
        return {
            "query_features": decoded,
            "reference_boxes": reference_boxes,
            "encoder_logits": encoder_logits,
            "encoder_boxes": encoder_boxes,
            "selected_indices": indices,
        }

    def forward(self, x):
        features = self.forward_features(x)
        box_delta = self.box_head(features["query_features"])
        reference = torch.logit(features["reference_boxes"].clamp(1e-4, 1 - 1e-4))
        return {
            "pred_logits": self.class_head(features["query_features"]),
            "pred_boxes": (box_delta + reference).sigmoid(),
            "encoder_logits": features["encoder_logits"],
            "encoder_boxes": features["encoder_boxes"],
            "selected_indices": features["selected_indices"],
        }

    @staticmethod
    def box_cxcywh_to_xyxy(boxes):
        cx, cy, width, height = boxes.unbind(-1)
        return torch.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), -1).clamp(0, 1)

    def post_process(self, outputs, score_threshold=0.0, top_k=None):
        scores, labels = outputs["pred_logits"].sigmoid().max(dim=-1)
        boxes = self.box_cxcywh_to_xyxy(outputs["pred_boxes"])
        results = []
        for sample_scores, sample_labels, sample_boxes in zip(scores, labels, boxes, strict=True):
            keep = sample_scores >= score_threshold
            sample_scores, sample_labels, sample_boxes = sample_scores[keep], sample_labels[keep], sample_boxes[keep]
            if top_k is not None and sample_scores.numel() > top_k:
                sample_scores, indices = sample_scores.topk(top_k)
                sample_labels, sample_boxes = sample_labels[indices], sample_boxes[indices]
            results.append({"scores": sample_scores, "labels": sample_labels, "boxes": sample_boxes})
        return results
