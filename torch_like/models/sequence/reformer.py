"""Reformer-style causal language model with bucketed LSH attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


@dataclass
class ReformerConfig(SequenceConfig):
    bucket_size: int = 16
    num_hashes: int = 2


class LSHCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bucket_size: int, num_hashes: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(bucket_size, num_hashes) <= 0:
            raise ValueError("bucket_size and num_hashes must be positive")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.bucket_size = bucket_size
        self.num_hashes = num_hashes
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        rotations = torch.randn(num_hashes, n_heads, self.head_dim)
        self.register_buffer("rotations", rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-6))

    def _bucket_mask(self, q: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, _ = q.shape
        positions = torch.arange(seq_len, device=q.device)
        local_bucket = positions // self.bucket_size
        same_local = local_bucket.view(1, 1, seq_len, 1) == local_bucket.view(1, 1, 1, seq_len)
        lsh_mask = torch.zeros(batch, heads, seq_len, seq_len, dtype=torch.bool, device=q.device)
        for idx in range(self.num_hashes):
            projection = (q * self.rotations[idx].view(1, heads, 1, self.head_dim)).sum(dim=-1)
            order = projection.argsort(dim=-1)
            rank = torch.empty_like(order)
            rank.scatter_(-1, order, torch.arange(seq_len, device=q.device).view(1, 1, seq_len).expand_as(order))
            bucket = rank // self.bucket_size
            lsh_mask |= bucket.unsqueeze(-1) == bucket.unsqueeze(-2)
        return same_local | lsh_mask

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        allowed = self._bucket_mask(q)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)).view(
            1, 1, seq_len, seq_len
        )
        allowed = allowed & causal
        if attention_mask is not None:
            allowed = allowed & attention_mask.bool().view(batch, 1, 1, seq_len)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.dropout(self.out(out))


class ReformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, bucket_size: int, num_hashes: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LSHCausalAttention(d_model, n_heads, bucket_size, num_hashes, dropout)
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


class Reformer(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        bucket_size: int = 16,
        num_hashes: int = 2,
    ):
        super().__init__()
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, bucket_size, num_hashes) <= 0:
            raise ValueError("all size parameters must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = ReformerConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            bucket_size=bucket_size,
            num_hashes=num_hashes,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ReformerBlock(d_model, n_heads, d_ff, bucket_size, num_hashes, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("Reformer expects input_ids with shape (batch, seq_len)")
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
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
            raise ValueError("SequenceBatch.labels is required for Reformer loss_step")
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
