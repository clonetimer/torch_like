"""GroundingDINO-style open-vocabulary object detector.

This implementation captures the reusable framework-side interface of
Grounding DINO: image features, text-token features, object queries, text
alignment logits, objectness scores, and normalized boxes. Tokenizers,
phrase parsing, NMS policy, Grounded-SAM orchestration, and pretrained
checkpoint conversion stay outside the model class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.multimodal.clip import CLIPTextEmbeddings
from torch_like.models.vision.common import VisionConfig, VisionModel
from torch_like.models.vision.detr import MLP, ConvBNAct, PositionEmbeddingSine


@dataclass
class GroundingDINOConfig(VisionConfig):
    vocab_size: int = 30522
    max_text_len: int = 256
    num_queries: int = 100
    backbone_channels: tuple[int, ...] = (64, 128, 256)
    n_heads: int = 8
    e_layers: int = 6
    d_layers: int = 6
    text_layers: int = 2
    d_ff: int = 2048
    pad_token_id: int = 0


class GroundingDINO(VisionModel):
    """Compact language-grounded detector with token-level alignment logits."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        image_size: int = 224,
        d_model: int = 256,
        dropout: float = 0.1,
        vocab_size: int = 30522,
        max_text_len: int = 256,
        num_queries: int = 100,
        backbone_channels: tuple[int, ...] = (64, 128, 256),
        n_heads: int = 8,
        e_layers: int = 6,
        d_layers: int = 6,
        text_layers: int = 2,
        d_ff: int = 2048,
        pad_token_id: int = 0,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")
        if min(vocab_size, max_text_len, num_queries, e_layers, d_layers, text_layers, d_ff) <= 0:
            raise ValueError("GroundingDINO dimensions and layer counts must be positive")
        if not backbone_channels or any(channel <= 0 for channel in backbone_channels):
            raise ValueError("backbone_channels must contain positive channels")

        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.num_queries = num_queries
        self.pad_token_id = pad_token_id
        self._config = GroundingDINOConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
            max_text_len=max_text_len,
            num_queries=num_queries,
            backbone_channels=backbone_channels,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            text_layers=text_layers,
            d_ff=d_ff,
            pad_token_id=pad_token_id,
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
        self.image_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.text_embeddings = CLIPTextEmbeddings(vocab_size, max_text_len, d_model, dropout, pad_token_id)
        text_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=text_layers)
        self.text_norm = nn.LayerNorm(d_model)

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
        self.query_norm = nn.LayerNorm(d_model)
        self.text_projection = nn.Linear(d_model, d_model, bias=False)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.objectness_head = nn.Linear(d_model, 1)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)
        self.feature_dim = d_model
        self._init_weights()

    @classmethod
    def grounding_dino_tiny(cls, **kwargs) -> "GroundingDINO":
        return cls(
            d_model=128,
            n_heads=4,
            e_layers=2,
            d_layers=2,
            text_layers=1,
            d_ff=512,
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
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
        nn.init.normal_(self.query_embed.weight, std=0.02)

    def encode_image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("GroundingDINO expects image with shape (batch, channels, height, width)")
        features = self.input_proj(self.backbone(image))
        pos = self.position_embedding(features)
        tokens = (features + pos).flatten(2).transpose(1, 2)
        return self.image_encoder(tokens)

    def encode_text_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("GroundingDINO expects input_ids with shape (batch, seq_len)")
        if input_ids.size(1) > self.max_text_len:
            raise ValueError(f"Expected text length <= {self.max_text_len}")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        hidden = self.text_embeddings(input_ids)
        hidden = self.text_encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        return self.text_norm(hidden)

    def forward_features(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image_tokens = self.encode_image_tokens(image)
        text_tokens = self.encode_text_tokens(input_ids, attention_mask)
        queries = self.query_embed.weight.unsqueeze(0).expand(image.size(0), -1, -1)
        query_hidden = self.query_norm(self.decoder(queries, image_tokens))
        return {
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "query_features": query_hidden,
        }

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        features = self.forward_features(image, input_ids, attention_mask)
        query_features = features["query_features"]
        text_features = features["text_tokens"]
        query_proj = F.normalize(self.query_projection(query_features), dim=-1)
        text_proj = F.normalize(self.text_projection(text_features), dim=-1)
        token_logits = query_proj @ text_proj.transpose(1, 2)
        token_logits = token_logits.masked_fill(~attention_mask.bool().unsqueeze(1), -1e4)
        objectness = self.objectness_head(query_features).squeeze(-1) + token_logits.max(dim=-1).values
        return {
            **features,
            "token_logits": token_logits,
            "objectness": objectness,
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
        scores = outputs["objectness"].sigmoid()
        token_scores, token_indices = outputs["token_logits"].sigmoid().max(dim=-1)
        boxes = self.box_cxcywh_to_xyxy(outputs["pred_boxes"])
        combined_scores = scores * token_scores

        detections: list[dict[str, torch.Tensor]] = []
        for sample_scores, sample_tokens, sample_boxes in zip(combined_scores, token_indices, boxes, strict=True):
            keep = sample_scores >= score_threshold
            sample_scores = sample_scores[keep]
            sample_tokens = sample_tokens[keep]
            sample_boxes = sample_boxes[keep]
            if top_k is not None and sample_scores.numel() > top_k:
                values, indices = sample_scores.topk(top_k)
                sample_scores = values
                sample_tokens = sample_tokens[indices]
                sample_boxes = sample_boxes[indices]
            detections.append({"scores": sample_scores, "token_indices": sample_tokens, "boxes": sample_boxes})
        return detections
