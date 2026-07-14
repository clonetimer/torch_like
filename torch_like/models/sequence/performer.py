"""Performer-style causal language model with linear attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


@dataclass
class PerformerConfig(SequenceConfig):
    num_features: int = 64
    redraw_features: bool = False


class PerformerAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_features: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_features = num_features
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        projection = torch.randn(n_heads, self.head_dim, num_features)
        projection = projection / projection.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("projection", projection)

    def redraw_projection(self) -> None:
        projection = torch.randn_like(self.projection)
        self.projection.copy_(projection / projection.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = x / (self.head_dim**0.25)
        projected = torch.einsum("bhld,hdf->bhlf", x, self.projection)
        stabilizer = (x.pow(2).sum(dim=-1, keepdim=True) / 2).clamp(max=30)
        return torch.exp(projected - stabilizer).clamp_min(1e-6)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q_features = self._feature_map(q)
        k_features = self._feature_map(k)
        if attention_mask is not None:
            mask = attention_mask.bool().view(batch, 1, seq_len, 1)
            k_features = k_features * mask
            v = v * mask

        kv = torch.einsum("bhlf,bhld->bhlfd", k_features, v).cumsum(dim=2)
        k_sum = k_features.cumsum(dim=2)
        numerator = torch.einsum("bhlf,bhlfd->bhld", q_features, kv)
        denominator = torch.einsum("bhlf,bhlf->bhl", q_features, k_sum).unsqueeze(-1).clamp_min(1e-6)
        out = numerator / denominator
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.dropout(self.out(out))


class PerformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_features: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PerformerAttention(d_model, n_heads, num_features, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        return x + self.ffn(self.norm2(x))


class Performer(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        num_features: int = 64,
        redraw_features: bool = False,
    ):
        super().__init__()
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, num_features) <= 0:
            raise ValueError("all size parameters must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.redraw_features = redraw_features
        self._config = PerformerConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            num_features=num_features,
            redraw_features=redraw_features,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [PerformerBlock(d_model, n_heads, d_ff, num_features, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def redraw_projection(self) -> None:
        for block in self.blocks:
            block.attn.redraw_projection()

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("Performer expects input_ids with shape (batch, seq_len)")
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        if self.training and self.redraw_features:
            self.redraw_projection()
        hidden = self.drop(self.token_embedding(input_ids) + self.position_embedding(position_ids))
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.norm(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del causal_mask
        return self.lm_head(self.forward_features(input_ids, attention_mask=attention_mask, position_ids=position_ids))

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for Performer loss_step")
        return self.causal_lm_loss(self.next_token_logits(batch), batch.labels)

    @torch.no_grad()
    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32) -> torch.Tensor:
        input_ids = batch.input_ids
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_seq_len:
                break
            logits = self.next_token_logits(self.make_sequence_batch(input_ids))
            input_ids = torch.cat([input_ids, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
        return input_ids
