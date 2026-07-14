"""BLIP-2-style vision-language bridge with a Q-Former.

This module implements the model-side core of BLIP-2: a vision encoder,
learnable query tokens, a Q-Former that cross-attends to image tokens, and
projection into a language-model feature space. Tokenizers, frozen external
LLM checkpoints, prompt templates, and large-scale pretraining schedules stay
outside the model class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.multimodal.clip import CLIPTextEmbeddings
from torch_like.models.multimodal.common import MultimodalConfig, MultimodalModel
from torch_like.models.vision.vit import PatchEmbedding


@dataclass
class BLIP2Config(MultimodalConfig):
    in_channels: int = 3
    vision_layers: int = 6
    vision_heads: int = 8
    qformer_layers: int = 6
    qformer_heads: int = 8
    num_query_tokens: int = 32
    lm_dim: int = 768
    d_ff: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class BLIP2(MultimodalModel):
    """Compact BLIP-2-style model with query-based image-to-language bridging."""

    def __init__(
        self,
        vocab_size: int = 30522,
        image_size: int = 224,
        patch_size: int = 16,
        max_text_len: int = 77,
        in_channels: int = 3,
        d_model: int = 512,
        embed_dim: int = 512,
        dropout: float = 0.0,
        vision_layers: int = 6,
        vision_heads: int = 8,
        qformer_layers: int = 6,
        qformer_heads: int = 8,
        num_query_tokens: int = 32,
        lm_dim: int = 768,
        d_ff: int = 2048,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model % vision_heads != 0 or d_model % qformer_heads != 0:
            raise ValueError("d_model must be divisible by vision_heads and qformer_heads")
        if (
            min(
                vocab_size,
                max_text_len,
                d_model,
                embed_dim,
                vision_layers,
                qformer_layers,
                num_query_tokens,
                lm_dim,
                d_ff,
            )
            <= 0
        ):
            raise ValueError("BLIP2 dimensions and layer counts must be positive")

        self.vocab_size = vocab_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_text_len = max_text_len
        self.num_query_tokens = num_query_tokens
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_patches = (image_size // patch_size) ** 2
        self._config = BLIP2Config(
            vocab_size=vocab_size,
            image_size=image_size,
            patch_size=patch_size,
            max_text_len=max_text_len,
            d_model=d_model,
            embed_dim=embed_dim,
            dropout=dropout,
            in_channels=in_channels,
            vision_layers=vision_layers,
            vision_heads=vision_heads,
            qformer_layers=qformer_layers,
            qformer_heads=qformer_heads,
            num_query_tokens=num_query_tokens,
            lm_dim=lm_dim,
            d_ff=d_ff,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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

        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, d_model))
        qformer_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=qformer_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.qformer = nn.TransformerDecoder(qformer_layer, num_layers=qformer_layers)
        self.qformer_norm = nn.LayerNorm(d_model)
        self.image_projection = nn.Linear(d_model, embed_dim, bias=False)

        self.text_embeddings = CLIPTextEmbeddings(vocab_size, max_text_len, d_model, dropout, pad_token_id)
        self.text_projection = nn.Linear(d_model, embed_dim, bias=False)
        self.itm_head = nn.Linear(d_model, 2)
        self.lm_projection = nn.Linear(d_model, lm_dim)
        self.token_embeddings = nn.Embedding(vocab_size, lm_dim, padding_idx=pad_token_id)
        self.lm_decoder = nn.Linear(lm_dim, vocab_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.feature_dim = embed_dim
        self._init_weights()

    @classmethod
    def blip2_tiny(cls, **kwargs) -> "BLIP2":
        return cls(
            d_model=128,
            embed_dim=128,
            vision_layers=2,
            vision_heads=4,
            qformer_layers=2,
            qformer_heads=4,
            num_query_tokens=8,
            lm_dim=128,
            d_ff=512,
            **kwargs,
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.visual_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
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

    def encode_image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("BLIP2 expects image with shape (batch, channels, height, width)")
        if image.size(-2) != self.image_size or image.size(-1) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}")
        tokens = self.patch_embed(image)
        cls = self.cls_token.expand(image.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.visual_pos_embed
        return self.visual_norm(self.visual_encoder(tokens))

    def qformer_features(self, image: torch.Tensor) -> torch.Tensor:
        image_tokens = self.encode_image_tokens(image)
        queries = self.query_tokens.expand(image.size(0), -1, -1)
        query_hidden = self.qformer(queries, image_tokens)
        return self.qformer_norm(query_hidden)

    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        query_hidden = self.qformer_features(image)
        image_features = self.image_projection(query_hidden.mean(dim=1))
        return F.normalize(image_features, dim=-1) if normalize else image_features

    def encode_text_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("BLIP2 expects input_ids with shape (batch, seq_len)")
        if input_ids.size(1) > self.max_text_len:
            raise ValueError(f"Expected text length <= {self.max_text_len}")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        hidden = self.text_embeddings(input_ids)
        hidden = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
        return hidden

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        hidden = self.encode_text_tokens(input_ids, attention_mask)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
        pooled = (hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom
        text_features = self.text_projection(pooled)
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def language_prefix(self, image: torch.Tensor) -> torch.Tensor:
        return self.lm_projection(self.qformer_features(image))

    def caption_logits(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix = self.language_prefix(image)
        token_hidden = self.token_embeddings(input_ids)
        if attention_mask is not None:
            token_hidden = token_hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
        context = prefix.mean(dim=1, keepdim=True)
        return self.lm_decoder(token_hidden + context)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image_embeds = self.encode_image(image)
        text_embeds = self.encode_text(input_ids, attention_mask)
        qformer_hidden = self.qformer_features(image)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_image.t(),
            "qformer_embeds": qformer_hidden,
            "language_prefix": self.lm_projection(qformer_hidden),
            "itm_logits": self.itm_head(qformer_hidden.mean(dim=1)),
            "lm_logits": self.caption_logits(image, input_ids, attention_mask),
        }

    def loss_step(self, batch) -> torch.Tensor:
        outputs = self.predict(batch)
        labels = torch.arange(outputs["logits_per_image"].size(0), device=outputs["logits_per_image"].device)
        contrastive = 0.5 * (
            F.cross_entropy(outputs["logits_per_image"], labels) + F.cross_entropy(outputs["logits_per_text"], labels)
        )
        if batch.labels is None:
            return contrastive
        target = batch.labels.long()
        caption_loss = F.cross_entropy(
            outputs["lm_logits"][:, :-1].reshape(-1, self.vocab_size),
            target[:, 1:].reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return contrastive + caption_loss
