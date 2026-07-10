"""DETR object detection model.

Framework-oriented implementation of "End-to-End Object Detection with
Transformers". This module exposes the core DETR interface: a CNN feature
extractor, Transformer encoder-decoder, learned object queries, class logits,
and normalized bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class DETRConfig(VisionConfig):
    num_queries: int = 100
    backbone_channels: tuple[int, ...] = (64, 128, 256)
    n_heads: int = 8
    e_layers: int = 6
    d_layers: int = 6
    d_ff: int = 2048


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers: list[nn.Module] = []
        for idx in range(num_layers):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionEmbeddingSine(nn.Module):
    """2D sine-cosine position embedding for CNN feature maps."""

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sine position embedding")
        self.num_pos_feats = d_model // 2
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        y_embed = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
        x_embed = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        y_embed = y_embed[:, None].expand(height, width)
        x_embed = x_embed[None, :].expand(height, width)
        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=x.dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1).unsqueeze(0)
        return pos.expand(batch_size, -1, -1, -1)


class DETR(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 91,
        image_size: int = 224,
        d_model: int = 256,
        dropout: float = 0.1,
        num_queries: int = 100,
        backbone_channels: tuple[int, ...] = (64, 128, 256),
        n_heads: int = 8,
        e_layers: int = 6,
        d_layers: int = 6,
        d_ff: int = 2048,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if not backbone_channels or any(channel <= 0 for channel in backbone_channels):
            raise ValueError("backbone_channels must contain positive channels")

        self.num_classes = num_classes
        self.num_queries = num_queries
        self._config = DETRConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            num_queries=num_queries,
            backbone_channels=backbone_channels,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
        )

        backbone: list[nn.Module] = []
        current_channels = in_channels
        for idx, out_channels in enumerate(backbone_channels):
            backbone.append(ConvBNAct(current_channels, out_channels, stride=2 if idx > 0 else 1))
            current_channels = out_channels
        self.backbone = nn.Sequential(*backbone)
        self.input_proj = nn.Conv2d(current_channels, d_model, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=d_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)
        self.feature_dim = d_model
        self._init_weights()

    @classmethod
    def detr_resnet50_style(cls, **kwargs) -> "DETR":
        return cls(backbone_channels=(64, 128, 256, 512), **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.query_embed.weight, std=0.02)

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("DETR expects x with shape (batch, channels, height, width)")
        features = self.input_proj(self.backbone(x))
        pos = self.position_embedding(features)
        tokens = (features + pos).flatten(2).transpose(1, 2)
        return self.encoder(tokens)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        memory = self._encode_image(x)
        queries = self.query_embed.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        return self.decoder(queries, memory)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        query_features = self.forward_features(x)
        return {
            "pred_logits": self.class_embed(query_features),
            "pred_boxes": self.bbox_embed(query_features).sigmoid(),
        }

    @staticmethod
    def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        x0 = cx - 0.5 * w
        y0 = cy - 0.5 * h
        x1 = cx + 0.5 * w
        y1 = cy + 0.5 * h
        return torch.stack((x0, y0, x1, y1), dim=-1).clamp(0.0, 1.0)

    def post_process(
        self,
        outputs: dict[str, torch.Tensor],
        score_threshold: float = 0.0,
        top_k: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        logits = outputs["pred_logits"]
        boxes = self.box_cxcywh_to_xyxy(outputs["pred_boxes"])
        probs = logits.softmax(dim=-1)[..., :-1]
        scores, labels = probs.max(dim=-1)

        detections: list[dict[str, torch.Tensor]] = []
        for sample_scores, sample_labels, sample_boxes in zip(scores, labels, boxes):
            keep = sample_scores >= score_threshold
            sample_scores = sample_scores[keep]
            sample_labels = sample_labels[keep]
            sample_boxes = sample_boxes[keep]
            if top_k is not None and sample_scores.numel() > top_k:
                values, indices = sample_scores.topk(top_k)
                sample_scores = values
                sample_labels = sample_labels[indices]
                sample_boxes = sample_boxes[indices]
            detections.append({"scores": sample_scores, "labels": sample_labels, "boxes": sample_boxes})
        return detections
