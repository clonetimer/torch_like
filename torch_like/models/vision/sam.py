"""Segment Anything style promptable segmentation model.

Framework-oriented implementation of SAM: an image encoder produces dense
image tokens, a prompt encoder embeds point/box/mask prompts, and a mask
decoder predicts one or more masks plus mask-quality scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class SAMConfig(VisionConfig):
    patch_size: int = 16
    encoder_layers: int = 12
    encoder_heads: int = 8
    d_ff: int = 2048
    decoder_layers: int = 2
    decoder_heads: int = 8
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256
    prompt_coordinate_format: Literal["normalized", "pixel"] = "normalized"


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
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RandomFourierPositionEmbedding(nn.Module):
    """Random Fourier features used for sparse and dense 2D prompt positions."""

    def __init__(self, d_model: int, scale: float = 10.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for Fourier position embedding")
        self.register_buffer("gaussian_matrix", scale * torch.randn(2, d_model // 2))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords * 2.0 - 1.0
        projected = 2.0 * math.pi * coords @ self.gaussian_matrix.to(dtype=coords.dtype)
        return torch.cat([projected.sin(), projected.cos()], dim=-1)


class SAMImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        d_model: int,
        encoder_layers: int,
        encoder_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, self.grid_size, self.grid_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )
        self.neck = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(1, d_model),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, d_model),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError("SAM expects images with shape (batch, channels, height, width)")
        if images.size(-2) != self.image_size or images.size(-1) != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}x{self.image_size}, got {images.size(-2)}x{images.size(-1)}"
            )
        features = self.patch_embed(images) + self.pos_embed
        batch_size, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens).transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.neck(encoded)


class SAMPromptEncoder(nn.Module):
    def __init__(self, d_model: int, image_size: int, grid_size: int, coordinate_format: str):
        super().__init__()
        if coordinate_format not in {"normalized", "pixel"}:
            raise ValueError("prompt_coordinate_format must be 'normalized' or 'pixel'")
        self.d_model = d_model
        self.image_size = image_size
        self.grid_size = grid_size
        self.coordinate_format = coordinate_format
        self.position_embedding = RandomFourierPositionEmbedding(d_model)
        self.point_label_embed = nn.Embedding(3, d_model)
        self.box_corner_embed = nn.Embedding(2, d_model)
        self.no_mask_embed = nn.Parameter(torch.zeros(1, d_model, 1, 1))
        hidden = max(d_model // 4, 8)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, d_model, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.no_mask_embed, std=0.02)

    def dense_position_encoding(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, self.grid_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dense = torch.stack([xx, yy], dim=-1)
        encoded = self.position_embedding(dense).permute(2, 0, 1).unsqueeze(0)
        return encoded.expand(batch_size, -1, -1, -1)

    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        if self.coordinate_format == "normalized":
            return coords.clamp(0.0, 1.0)
        return (coords / float(max(self.image_size - 1, 1))).clamp(0.0, 1.0)

    def _embed_points(self, points: torch.Tensor, point_labels: torch.Tensor | None) -> torch.Tensor:
        if points.dim() != 3 or points.size(-1) != 2:
            raise ValueError("points must have shape (batch, num_points, 2)")
        if point_labels is None:
            point_labels = torch.ones(points.shape[:-1], device=points.device, dtype=torch.long)
        if point_labels.shape != points.shape[:-1]:
            raise ValueError("point_labels must have shape (batch, num_points)")
        coords = self._normalize(points)
        labels = point_labels.to(device=points.device, dtype=torch.long).clamp(-1, 1) + 1
        return self.position_embedding(coords) + self.point_label_embed(labels)

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(1)
        if boxes.dim() != 3 or boxes.size(-1) != 4:
            raise ValueError("boxes must have shape (batch, 4) or (batch, num_boxes, 4)")
        corners = boxes.reshape(boxes.size(0), boxes.size(1), 2, 2)
        corners = self._normalize(corners)
        labels = torch.arange(2, device=boxes.device).view(1, 1, 2)
        return (self.position_embedding(corners) + self.box_corner_embed(labels)).flatten(1, 2)

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        points: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_embeddings: list[torch.Tensor] = []
        if points is not None:
            sparse_embeddings.append(self._embed_points(points.to(device=device, dtype=dtype), point_labels))
        if boxes is not None:
            sparse_embeddings.append(self._embed_boxes(boxes.to(device=device, dtype=dtype)))

        if sparse_embeddings:
            sparse = torch.cat(sparse_embeddings, dim=1)
        else:
            sparse = torch.empty(batch_size, 0, self.d_model, device=device, dtype=dtype)

        if masks is None:
            dense = self.no_mask_embed.to(device=device, dtype=dtype).expand(
                batch_size, -1, self.grid_size, self.grid_size
            )
        else:
            if masks.dim() != 4 or masks.size(1) != 1:
                raise ValueError("masks must have shape (batch, 1, height, width)")
            masks = masks.to(device=device, dtype=dtype)
            masks = F.interpolate(masks, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
            dense = self.mask_downscaling(masks)
        return sparse, dense


class SAMMaskDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        decoder_layers: int,
        decoder_heads: int,
        d_ff: int,
        dropout: float,
        num_multimask_outputs: int,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
    ):
        super().__init__()
        if num_multimask_outputs <= 0:
            raise ValueError("num_multimask_outputs must be positive")
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, d_model)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        upscaled_dim = max(d_model // 8, 1)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(d_model, max(d_model // 4, 1), kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(max(d_model // 4, 1), upscaled_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks = nn.ModuleList(
            MLP(d_model, d_model, upscaled_dim, num_layers=3) for _ in range(self.num_mask_tokens)
        )
        self.iou_prediction_head = MLP(d_model, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = image_embeddings.size(0)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        image_features = image_embeddings + image_pe + dense_prompt_embeddings
        memory = image_features.flatten(2).transpose(1, 2)
        decoded_tokens = self.decoder(tokens, memory)
        iou_token_out = decoded_tokens[:, 0, :]
        mask_tokens_out = decoded_tokens[:, 1 : 1 + self.num_mask_tokens, :]

        upscaled = self.output_upscaling(image_features)
        hyper_in = torch.stack(
            [head(mask_tokens_out[:, idx, :]) for idx, head in enumerate(self.output_hypernetworks)],
            dim=1,
        )
        masks = torch.einsum("bqc,bchw->bqhw", hyper_in, upscaled)
        iou_predictions = self.iou_prediction_head(iou_token_out)
        return masks, iou_predictions


class SAM(VisionModel):
    """Promptable segmentation model with SAM-style public methods."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        image_size: int = 1024,
        d_model: int = 256,
        dropout: float = 0.0,
        patch_size: int = 16,
        encoder_layers: int = 12,
        encoder_heads: int = 8,
        d_ff: int = 2048,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        prompt_coordinate_format: Literal["normalized", "pixel"] = "normalized",
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_model % encoder_heads != 0 or d_model % decoder_heads != 0:
            raise ValueError("d_model must be divisible by encoder_heads and decoder_heads")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        self.image_size = image_size
        self.num_multimask_outputs = num_multimask_outputs
        self._config = SAMConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            patch_size=patch_size,
            encoder_layers=encoder_layers,
            encoder_heads=encoder_heads,
            d_ff=d_ff,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            num_multimask_outputs=num_multimask_outputs,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            prompt_coordinate_format=prompt_coordinate_format,
        )
        self.image_encoder = SAMImageEncoder(
            in_channels,
            image_size,
            patch_size,
            d_model,
            encoder_layers,
            encoder_heads,
            d_ff,
            dropout,
        )
        grid_size = image_size // patch_size
        self.prompt_encoder = SAMPromptEncoder(d_model, image_size, grid_size, prompt_coordinate_format)
        self.mask_decoder = SAMMaskDecoder(
            d_model,
            decoder_layers,
            decoder_heads,
            d_ff,
            dropout,
            num_multimask_outputs,
            iou_head_depth,
            iou_head_hidden_dim,
        )
        self.feature_dim = d_model

    @classmethod
    def sam_tiny(cls, **kwargs) -> "SAM":
        return cls(image_size=256, d_model=128, encoder_layers=4, encoder_heads=4, d_ff=512, **kwargs)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_prompts(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        points: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prompt_encoder(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            points=points,
            point_labels=point_labels,
            boxes=boxes,
            masks=masks,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        *,
        multimask_output: bool = True,
    ) -> dict[str, torch.Tensor]:
        image_pe = self.prompt_encoder.dense_position_encoding(
            image_embeddings.size(0),
            image_embeddings.device,
            image_embeddings.dtype,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        )
        if multimask_output:
            selected = slice(1, None)
        else:
            selected = slice(0, 1)
        return {
            "low_res_masks": low_res_masks[:, selected],
            "iou_predictions": iou_predictions[:, selected],
        }

    def forward(
        self,
        images: torch.Tensor,
        *,
        points: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        multimask_output: bool = True,
    ) -> dict[str, torch.Tensor]:
        image_embeddings = self.encode_image(images)
        sparse, dense = self.encode_prompts(
            batch_size=images.size(0),
            device=images.device,
            dtype=images.dtype,
            points=points,
            point_labels=point_labels,
            boxes=boxes,
            masks=masks,
        )
        out = self.predict_masks(image_embeddings, sparse, dense, multimask_output=multimask_output)
        out["masks"] = F.interpolate(
            out["low_res_masks"],
            size=(images.size(-2), images.size(-1)),
            mode="bilinear",
            align_corners=False,
        )
        return out
