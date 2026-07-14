"""LLaMA-style decoder-only language model.

Framework-oriented implementation of the LLaMA/Llama architecture family:
RMSNorm pre-normalization, rotary position embeddings, SwiGLU feed-forward
blocks, tied token embeddings, and optional grouped-query attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel, causal_sdpa_mask


@dataclass
class LLaMAConfig(SequenceConfig):
    n_kv_heads: int | None = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    tie_word_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached.to(device=position_ids.device, dtype=dtype).squeeze(0).squeeze(0)
        sin = self.sin_cached.to(device=position_ids.device, dtype=dtype).squeeze(0).squeeze(0)
        return cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1)


class LLaMAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        batch_size, n_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch_size, n_kv_heads, self.n_rep, seq_len, head_dim)
        return x.reshape(batch_size, n_kv_heads * self.n_rep, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_embedding(q, cos, sin)
        k = apply_rotary_embedding(k, cos, sin)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_mask, is_causal = causal_sdpa_mask(
            attention_mask,
            causal_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
        )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.resid_drop(self.o_proj(out))


class LLaMAFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        dropout: float,
    ):
        super().__init__()
        if ffn_dim_multiplier is not None:
            d_ff = int(ffn_dim_multiplier * d_ff)
        if multiple_of > 1:
            d_ff = multiple_of * math.ceil(d_ff / multiple_of)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)))


class LLaMABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float,
        rms_norm_eps: float,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.attention = LLaMAAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.feed_forward = LLaMAFeedForward(d_model, d_ff, multiple_of, ffn_dim_multiplier, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x),
            cos,
            sin,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
        )
        return x + self.feed_forward(self.ffn_norm(x))


class LLaMA(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 4096,
        n_heads: int = 32,
        n_layers: int = 32,
        d_ff: int = 11008,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, n_kv_heads) <= 0:
            raise ValueError("LLaMA dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = LLaMAConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_embedding = RotaryEmbedding(d_model // n_heads, max_seq_len, theta=rope_theta)
        self.layers = nn.ModuleList(
            [
                LLaMABlock(
                    d_model,
                    n_heads,
                    n_kv_heads,
                    d_ff,
                    dropout,
                    rms_norm_eps,
                    multiple_of,
                    ffn_dim_multiplier,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    @classmethod
    def llama_tiny(cls, **kwargs) -> "LLaMA":
        return cls(d_model=128, n_heads=4, n_layers=2, d_ff=256, multiple_of=16, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("LLaMA expects input_ids with shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids)
        cos, sin = self.rotary_embedding(position_ids, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, attention_mask=attention_mask, causal_mask=causal_mask)
        return self.norm(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.lm_head(self.forward_features(input_ids, attention_mask, position_ids, causal_mask))

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for LLaMA loss_step")
        logits = self.next_token_logits(batch)
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError(
                f"LLaMA expects labels with shape {tuple(batch.input_ids.shape)}, got {tuple(batch.labels.shape)}"
            )
        return self.causal_lm_loss(logits, batch.labels)

    @torch.no_grad()
    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32) -> torch.Tensor:
        input_ids = batch.input_ids
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_seq_len:
                break
            step_batch = self.make_sequence_batch(input_ids=input_ids)
            logits = self.next_token_logits(step_batch)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
