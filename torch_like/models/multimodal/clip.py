"""CLIP-style image-text dual encoder.

Framework-oriented implementation of "Learning Transferable Visual Models From
Natural Language Supervision". The model accepts already-tokenized text ids and
images, returns normalized image/text embeddings, and exposes image-text
similarity logits through a PyTorch-native API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.multimodal.common import MultimodalConfig, MultimodalModel
from torch_like.models.vision.vit import PatchEmbedding


@dataclass
class CLIPConfig(MultimodalConfig):
    in_channels: int = 3
    vision_layers: int = 12
    vision_heads: int = 8
    text_layers: int = 12
    text_heads: int = 8
    d_ff: int = 2048
    pad_token_id: int = 0


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, max_text_len: int, d_model: int, dropout: float, pad_token_id: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_text_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return self.dropout(self.token_embedding(input_ids) + self.position_embedding(position_ids))


class CLIP(MultimodalModel):
    def __init__(
        self,
        vocab_size: int = 49408,
        image_size: int = 224,
        patch_size: int = 16,
        max_text_len: int = 77,
        in_channels: int = 3,
        d_model: int = 512,
        embed_dim: int = 512,
        dropout: float = 0.0,
        vision_layers: int = 12,
        vision_heads: int = 8,
        text_layers: int = 12,
        text_heads: int = 8,
        d_ff: int = 2048,
        pad_token_id: int = 0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model % vision_heads != 0 or d_model % text_heads != 0:
            raise ValueError("d_model must be divisible by both vision_heads and text_heads")
        if min(vocab_size, max_text_len, d_model, embed_dim, vision_layers, text_layers) <= 0:
            raise ValueError("vocab_size, max_text_len, d_model, embed_dim, vision_layers, and text_layers must be positive")

        self.vocab_size = vocab_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_text_len = max_text_len
        self.pad_token_id = pad_token_id
        self.num_patches = (image_size // patch_size) ** 2
        self._config = CLIPConfig(
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

        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.visual_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        vision_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=vision_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(vision_layer, num_layers=vision_layers)
        self.visual_norm = nn.LayerNorm(d_model)
        self.visual_projection = nn.Linear(d_model, embed_dim, bias=False)

        self.text_embeddings = CLIPTextEmbeddings(vocab_size, max_text_len, d_model, dropout, pad_token_id)
        text_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=text_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=text_layers)
        self.text_norm = nn.LayerNorm(d_model)
        self.text_projection = nn.Linear(d_model, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.feature_dim = embed_dim
        self._init_weights()

    @classmethod
    def clip_tiny(cls, **kwargs) -> "CLIP":
        return cls(d_model=128, embed_dim=128, vision_layers=2, vision_heads=4, text_layers=2, text_heads=4, d_ff=512, **kwargs)

    @classmethod
    def clip_base(cls, **kwargs) -> "CLIP":
        return cls(d_model=512, embed_dim=512, vision_layers=12, vision_heads=8, text_layers=12, text_heads=8, d_ff=2048, **kwargs)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.visual_pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("CLIP expects image with shape (batch, channels, height, width)")
        if image.size(-2) != self.image_size or image.size(-1) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}, got {image.size(-2)}x{image.size(-1)}")
        tokens = self.patch_embed(image)
        cls = self.cls_token.expand(image.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.visual_pos_embed
        hidden = self.visual_norm(self.visual_encoder(tokens))
        image_features = self.visual_projection(hidden[:, 0])
        return F.normalize(image_features, dim=-1) if normalize else image_features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("CLIP expects input_ids with shape (batch, seq_len)")
        if input_ids.size(1) > self.max_text_len:
            raise ValueError(f"Expected text length <= {self.max_text_len}, got {input_ids.size(1)}")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        hidden = self.text_embeddings(input_ids)
        hidden = self.text_encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        hidden = self.text_norm(hidden)
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        pooled = hidden[torch.arange(input_ids.size(0), device=input_ids.device), lengths]
        text_features = self.text_projection(pooled)
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image_embeds = self.encode_image(image)
        text_embeds = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_image.t(),
        }

    def loss_step(self, batch) -> torch.Tensor:
        outputs = self.predict(batch)
        labels = torch.arange(outputs["logits_per_image"].size(0), device=outputs["logits_per_image"].device)
        image_loss = F.cross_entropy(outputs["logits_per_image"], labels)
        text_loss = F.cross_entropy(outputs["logits_per_text"], labels)
        return 0.5 * (image_loss + text_loss)
