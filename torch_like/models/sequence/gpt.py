"""GPT-style decoder-only language model.

Framework-oriented implementation of the Generative Pre-Training / GPT-2
decoder-only Transformer family. The model exposes causal next-token logits,
language-model loss, and greedy generation through the sequence-domain API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel, causal_sdpa_mask


@dataclass
class GPTConfig(SequenceConfig):
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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
        return self.resid_drop(self.out_proj(out))


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, layer_norm_eps: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask, causal_mask=causal_mask)
        return x + self.mlp(self.ln_2(x))


class GPT(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(vocab_size, max_seq_len, d_model, n_layers) <= 0:
            raise ValueError("vocab_size, max_seq_len, d_model, and n_layers must be positive")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
        )

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads, d_ff, dropout, layer_norm_eps) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    @classmethod
    def gpt_tiny(cls, **kwargs) -> "GPT":
        return cls(d_model=128, n_heads=4, n_layers=2, d_ff=512, **kwargs)

    @classmethod
    def gpt_base(cls, **kwargs) -> "GPT":
        return cls(d_model=768, n_heads=12, n_layers=12, d_ff=3072, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
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
            raise ValueError("GPT expects input_ids with shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.drop(self.token_embedding(input_ids) + self.position_embedding(position_ids))
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask, causal_mask=causal_mask)
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
            raise ValueError("SequenceBatch.labels is required for GPT loss_step")
        logits = self.next_token_logits(batch)
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError(
                f"GPT expects labels with shape {tuple(batch.input_ids.shape)}, got {tuple(batch.labels.shape)}"
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
