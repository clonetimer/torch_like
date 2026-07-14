"""Mixtral-style sparse mixture-of-experts decoder-only language model.

Framework-oriented implementation of Mixtral: a LLaMA-style decoder with RoPE,
RMSNorm, grouped-query attention, and sparse top-k expert routing in the
feed-forward block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel
from torch_like.models.sequence.llama import LLaMAAttention, LLaMAFeedForward, RMSNorm, RotaryEmbedding


@dataclass
class MixtralConfig(SequenceConfig):
    n_kv_heads: int | None = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    tie_word_embeddings: bool = True


class MixtralSparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        dropout: float,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if num_experts_per_tok <= 0 or num_experts_per_tok > num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                LLaMAFeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        flat_x = x.reshape(-1, d_model)
        router_logits = self.gate(flat_x)
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(dtype=flat_x.dtype)

        flat_out = torch.zeros_like(flat_x)
        for expert_idx, expert in enumerate(self.experts):
            token_ids, route_ids = torch.where(selected_experts == expert_idx)
            if token_ids.numel() == 0:
                continue
            expert_out = expert(flat_x[token_ids])
            flat_out.index_add_(0, token_ids, expert_out * routing_weights[token_ids, route_ids].unsqueeze(-1))
        return flat_out.reshape(batch_size, seq_len, d_model), router_logits.reshape(
            batch_size, seq_len, self.num_experts
        )


class MixtralBlock(nn.Module):
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
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.attention = LLaMAAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.block_sparse_moe = MixtralSparseMoE(
            d_model=d_model,
            d_ff=d_ff,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            dropout=dropout,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(
            self.attention_norm(x),
            cos,
            sin,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
        )
        moe_out, router_logits = self.block_sparse_moe(self.ffn_norm(x))
        return x + moe_out, router_logits


class Mixtral(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 4096,
        n_heads: int = 32,
        n_layers: int = 32,
        d_ff: int = 14336,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, n_kv_heads) <= 0:
            raise ValueError("Mixtral dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = MixtralConfig(
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
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_embedding = RotaryEmbedding(d_model // n_heads, max_seq_len, theta=rope_theta)
        self.layers = nn.ModuleList(
            [
                MixtralBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    rms_norm_eps=rms_norm_eps,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
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
    def mixtral_tiny(cls, **kwargs) -> "Mixtral":
        return cls(d_model=128, n_heads=4, n_layers=2, d_ff=256, multiple_of=16, num_experts=4, **kwargs)

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
        return_router_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if input_ids.dim() != 2:
            raise ValueError("Mixtral expects input_ids with shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids)
        cos, sin = self.rotary_embedding(position_ids, hidden.dtype)
        router_logits_by_layer: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, router_logits = layer(hidden, cos, sin, attention_mask=attention_mask, causal_mask=causal_mask)
            if return_router_logits:
                router_logits_by_layer.append(router_logits)
        hidden = self.norm(hidden)
        if return_router_logits:
            return hidden, router_logits_by_layer
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.lm_head(self.forward_features(input_ids, attention_mask, position_ids, causal_mask))

    def router_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        _, router_logits_by_layer = self.forward_features(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            return_router_logits=True,
        )
        return router_logits_by_layer

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for Mixtral loss_step")
        logits = self.next_token_logits(batch)
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError(
                f"Mixtral expects labels with shape {tuple(batch.input_ids.shape)}, got {tuple(batch.labels.shape)}"
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
