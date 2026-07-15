"""LLaVA-style visual instruction language model.

Framework-oriented implementation of Visual Instruction Tuning: a vision
encoder produces image tokens, a projector maps them into the language hidden
space, and a causal decoder predicts text tokens conditioned on the visual
prefix.
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
class LLaVAConfig(MultimodalConfig):
    in_channels: int = 3
    vision_layers: int = 6
    vision_heads: int = 8
    decoder_layers: int = 6
    decoder_heads: int = 8
    d_ff: int = 2048
    projector_hidden_dim: int | None = None
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class LLaVA(MultimodalModel):
    """Compact LLaVA-style model for visual-prefix causal language modeling."""

    def __init__(
        self,
        vocab_size: int = 32000,
        image_size: int = 224,
        patch_size: int = 16,
        max_text_len: int = 512,
        in_channels: int = 3,
        d_model: int = 512,
        embed_dim: int = 512,
        dropout: float = 0.0,
        vision_layers: int = 6,
        vision_heads: int = 8,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        d_ff: int = 2048,
        projector_hidden_dim: int | None = None,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model % vision_heads != 0 or d_model % decoder_heads != 0:
            raise ValueError("d_model must be divisible by vision_heads and decoder_heads")
        if min(vocab_size, max_text_len, d_model, embed_dim, vision_layers, decoder_layers, d_ff) <= 0:
            raise ValueError("LLaVA dimensions and layer counts must be positive")
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_text_len = max_text_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_patches = (image_size // patch_size) ** 2
        self._config = LLaVAConfig(
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
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            d_ff=d_ff,
            projector_hidden_dim=projector_hidden_dim,
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
        self.visual_encoder = nn.TransformerEncoder(
            vision_layer,
            num_layers=vision_layers,
            enable_nested_tensor=False,
        )
        self.visual_norm = nn.LayerNorm(d_model)

        hidden = projector_hidden_dim or max(d_model, embed_dim)
        self.mm_projector = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.text_embeddings = CLIPTextEmbeddings(vocab_size, max_text_len, d_model, dropout, pad_token_id)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=decoder_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.text_embeddings.token_embedding.weight
        self.feature_dim = embed_dim
        self._init_weights()

    @classmethod
    def llava_tiny(cls, **kwargs) -> "LLaVA":
        return cls(
            d_model=128,
            embed_dim=128,
            vision_layers=2,
            vision_heads=4,
            decoder_layers=2,
            decoder_heads=4,
            d_ff=512,
            **kwargs,
        )

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

    def encode_image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("LLaVA expects image with shape (batch, channels, height, width)")
        if image.size(-2) != self.image_size or image.size(-1) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}")
        tokens = self.patch_embed(image)
        cls = self.cls_token.expand(image.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.visual_pos_embed
        return self.visual_norm(self.visual_encoder(tokens))

    def project_image_tokens(self, image_tokens: torch.Tensor, drop_cls: bool = True) -> torch.Tensor:
        if image_tokens.dim() != 3:
            raise ValueError("image_tokens must have shape (batch, image_tokens, d_model)")
        if drop_cls:
            image_tokens = image_tokens[:, 1:, :]
        return self.mm_projector(image_tokens)

    def multimodal_embeddings(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("LLaVA expects input_ids with shape (batch, seq_len)")
        if input_ids.size(1) > self.max_text_len:
            raise ValueError(f"Expected text length <= {self.max_text_len}")
        image_tokens = self.project_image_tokens(self.encode_image_tokens(image), drop_cls=True)
        text_tokens = self.text_embeddings(input_ids)
        return torch.cat([image_tokens, text_tokens], dim=1)

    def _combined_padding_mask(
        self,
        input_ids: torch.Tensor,
        num_image_tokens: int,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        image_mask = torch.ones(input_ids.size(0), num_image_tokens, device=input_ids.device, dtype=torch.bool)
        return torch.cat([image_mask, attention_mask.bool()], dim=1)

    def forward_features(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.multimodal_embeddings(image, input_ids)
        num_image_tokens = embeddings.size(1) - input_ids.size(1)
        combined_mask = self._combined_padding_mask(input_ids, num_image_tokens, attention_mask)
        seq_len = embeddings.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=embeddings.device, dtype=torch.bool), diagonal=1)
        hidden = self.decoder(embeddings, mask=causal_mask, src_key_padding_mask=~combined_mask)
        return self.decoder_norm(hidden)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.forward_features(image, input_ids, attention_mask)
        text_hidden = hidden[:, -input_ids.size(1) :, :]
        return {
            "multimodal_embeds": hidden,
            "text_embeds": text_hidden,
            "lm_logits": self.lm_head(text_hidden),
        }

    def loss_step(self, batch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("LLaVA loss_step requires batch.labels token ids")
        outputs = self.predict(batch)
        logits = outputs["lm_logits"]
        labels = batch.labels.long()
        if labels.shape != batch.input_ids.shape:
            raise ValueError("LLaVA labels must match input_ids shape")
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=self.pad_token_id,
        )

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        generated = input_ids
        for _ in range(max_new_tokens):
            if generated.size(1) >= self.max_text_len:
                break
            logits = self(image, generated)["lm_logits"]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1).eq(self.eos_token_id)):
                break
        return generated
