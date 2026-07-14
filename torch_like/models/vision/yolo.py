"""YOLO-style single-stage object detection model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class YOLOConfig(VisionConfig):
    anchors: tuple[tuple[float, float], ...] = ((0.10, 0.13), (0.16, 0.30), (0.33, 0.23))
    backbone_channels: tuple[int, ...] = (32, 64, 128)
    grid_size: int = 8


class _ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class YOLO(VisionModel):
    """Compact YOLO-style detector exposing grid predictions and post-processing."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
        image_size: int = 224,
        d_model: int = 128,
        dropout: float = 0.0,
        anchors: tuple[tuple[float, float], ...] = ((0.10, 0.13), (0.16, 0.30), (0.33, 0.23)),
        backbone_channels: tuple[int, ...] = (32, 64, 128),
        grid_size: int = 8,
    ):
        super().__init__()
        if min(in_channels, num_classes, image_size, d_model, grid_size) <= 0:
            raise ValueError("YOLO dimensions must be positive")
        if not anchors or any(width <= 0 or height <= 0 for width, height in anchors):
            raise ValueError("anchors must contain positive normalized width/height pairs")
        if not backbone_channels or any(channel <= 0 for channel in backbone_channels):
            raise ValueError("backbone_channels must contain positive channels")
        self.num_classes = num_classes
        self.anchors = tuple((float(width), float(height)) for width, height in anchors)
        self.num_anchors = len(self.anchors)
        self.grid_size = grid_size
        self._config = YOLOConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            anchors=self.anchors,
            backbone_channels=backbone_channels,
            grid_size=grid_size,
        )

        layers: list[nn.Module] = []
        current = in_channels
        for out_channels in backbone_channels:
            layers.append(_ConvBNAct(current, out_channels, stride=2))
            layers.append(_ConvBNAct(out_channels, out_channels, stride=1))
            current = out_channels
        self.backbone = nn.Sequential(*layers)
        self.neck = nn.Sequential(
            _ConvBNAct(current, d_model, stride=1),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            _ConvBNAct(d_model, d_model, stride=1),
        )
        self.prediction = nn.Conv2d(d_model, self.num_anchors * (5 + num_classes), kernel_size=1)
        self.register_buffer("anchor_tensor", torch.tensor(self.anchors, dtype=torch.float32), persistent=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("YOLO expects x with shape (batch, channels, height, width)")
        features = self.neck(self.backbone(x))
        if features.shape[-2:] != (self.grid_size, self.grid_size):
            features = F.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
        return features

    def raw_predictions(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        raw = self.prediction(self.forward_features(x))
        raw = raw.view(batch, self.num_anchors, 5 + self.num_classes, self.grid_size, self.grid_size)
        return raw.permute(0, 1, 3, 4, 2).contiguous()

    def decode_predictions(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, num_anchors, height, width, _ = raw.shape
        device = raw.device
        dtype = raw.dtype
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, height, width, 2)
        xy = (raw[..., 0:2].sigmoid() + grid) / torch.tensor([width, height], device=device, dtype=dtype)
        anchors = self.anchor_tensor.to(device=device, dtype=dtype).view(1, num_anchors, 1, 1, 2)
        wh = raw[..., 2:4].exp().clamp(max=4.0) * anchors
        boxes = torch.cat([xy, wh.clamp(max=1.0)], dim=-1).view(batch, -1, 4)
        objectness = raw[..., 4].reshape(batch, -1)
        class_logits = raw[..., 5:].reshape(batch, -1, self.num_classes)
        return {"pred_logits": class_logits, "pred_boxes": boxes, "objectness": objectness}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode_predictions(self.raw_predictions(x))

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
        class_probs = outputs["pred_logits"].softmax(dim=-1)
        class_scores, labels = class_probs.max(dim=-1)
        scores = outputs["objectness"].sigmoid() * class_scores
        boxes = self.box_cxcywh_to_xyxy(outputs["pred_boxes"])
        detections: list[dict[str, torch.Tensor]] = []
        for sample_scores, sample_labels, sample_boxes in zip(scores, labels, boxes, strict=True):
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
