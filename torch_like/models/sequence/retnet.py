"""RetNet-style causal language model with parallel and recurrent retention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


@dataclass
class RetNetConfig(SequenceConfig):
    retention_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True


class _RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class _GatedFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.value = nn.Linear(d_model, d_ff, bias=False)
        self.output = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.output(F.silu(self.gate(x)) * self.value(x)))


class Retention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float, eps: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_norm = _RMSNorm(self.head_dim, eps)
        self.dropout = nn.Dropout(dropout)

        head_indices = torch.arange(n_heads, dtype=torch.float32)
        decay_exponents = 5.0 + head_indices * 3.0 / max(n_heads - 1, 1)
        self.register_buffer("decays", 1.0 - 2.0 ** (-decay_exponents), persistent=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def _project(self, x: torch.Tensor, position_ids: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        phase = position_ids.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        cos = torch.repeat_interleave(phase.cos(), 2, dim=-1).unsqueeze(1).to(q.dtype)
        sin = torch.repeat_interleave(phase.sin(), 2, dim=-1).unsqueeze(1).to(q.dtype)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q / math.sqrt(self.head_dim), k, v

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _finish(self, x: torch.Tensor, retained: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = retained.shape
        retained = self.head_norm(retained).transpose(1, 2).reshape(batch_size, seq_len, -1)
        gated = F.silu(self.g_proj(x)) * retained
        return self.dropout(self.out_proj(gated))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, k, v = self._project(x, position_ids)
        q_acc, k_acc, v_acc = q.float(), k.float(), v.float()
        scores = torch.matmul(q_acc, k_acc.transpose(-2, -1))
        offsets = torch.arange(seq_len, device=x.device)
        distance = offsets[:, None] - offsets[None, :]
        allowed = distance >= 0
        weights = self.decays.to(scores.dtype)[:, None, None] ** distance.clamp_min(0).to(scores.dtype)
        weights = weights * allowed.to(scores.dtype)
        if causal_mask is not None:
            if causal_mask.shape != (seq_len, seq_len):
                raise ValueError(f"Expected causal_mask shape ({seq_len}, {seq_len})")
            weights = weights * (~causal_mask.bool()).to(scores.dtype).unsqueeze(0)
        scores = scores * weights.unsqueeze(0)
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(f"Expected attention_mask shape ({batch_size}, {seq_len})")
            scores = scores * attention_mask.to(scores.dtype).view(batch_size, 1, 1, seq_len)
        retained = torch.matmul(scores, v_acc)
        return self._finish(x, retained)

    def recurrent(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q, k, v = self._project(x, position_ids)
        q_acc, k_acc, v_acc = q.float(), k.float(), v.float()
        if state is None:
            state = torch.zeros(
                batch_size, self.n_heads, self.head_dim, self.head_dim, device=q.device, dtype=torch.float32
            )
        expected = (batch_size, self.n_heads, self.head_dim, self.head_dim)
        if state.shape != expected:
            raise ValueError(f"Expected retention state shape {expected}, got {tuple(state.shape)}")
        state = state.to(device=q.device, dtype=torch.float32)
        outputs = []
        decay = self.decays.view(1, self.n_heads, 1, 1)
        for index in range(seq_len):
            state = decay * state + torch.einsum("bhd,bhe->bhde", k_acc[:, :, index], v_acc[:, :, index])
            outputs.append(torch.einsum("bhd,bhde->bhe", q_acc[:, :, index], state))
        retained = torch.stack(outputs, dim=2)
        return self._finish(x, retained), state


class RetNetBlock(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.retention_norm = _RMSNorm(config.d_model, config.retention_norm_eps)
        self.retention = Retention(
            config.d_model, config.n_heads, config.max_seq_len, config.dropout, config.retention_norm_eps
        )
        self.ffn_norm = _RMSNorm(config.d_model, config.retention_norm_eps)
        self.ffn = _GatedFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x, position_ids, attention_mask=None, causal_mask=None):
        x = x + self.retention(self.retention_norm(x), position_ids, attention_mask, causal_mask)
        return x + self.ffn(self.ffn_norm(x))

    def recurrent(self, x, position_ids, state=None):
        retained, state = self.retention.recurrent(self.retention_norm(x), position_ids, state)
        x = x + retained
        return x + self.ffn(self.ffn_norm(x)), state


class RetNet(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        retention_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("RetNet dimensions and layer counts must be positive")
        if d_model % n_heads != 0 or (d_model // n_heads) % 2 != 0:
            raise ValueError("d_model / n_heads must be an even integer")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = RetNetConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            retention_norm_eps=retention_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RetNetBlock(self._config) for _ in range(n_layers)])
        self.norm = _RMSNorm(d_model, retention_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward_features(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None):
        if input_ids.dim() != 2:
            raise ValueError("RetNet expects input_ids with shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("RetNet requires at least one input token")
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, position_ids, attention_mask, causal_mask)
        return self.norm(hidden)

    def forward(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None):
        return self.lm_head(self.forward_features(input_ids, attention_mask, position_ids, causal_mask))

    def forward_recurrent(self, input_ids, states=None, position_offset: int = 0):
        if input_ids.dim() != 2:
            raise ValueError("RetNet expects input_ids with shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("RetNet requires at least one input token")
        if position_offset < 0 or position_offset + seq_len > self.max_seq_len:
            raise ValueError("Recurrent positions exceed max_seq_len")
        if states is None:
            states = [None] * len(self.layers)
        if len(states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} layer states, got {len(states)}")
        position_ids = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device).expand(
            batch_size, seq_len
        )
        hidden = self.token_embedding(input_ids)
        new_states = []
        for layer, state in zip(self.layers, states, strict=True):
            hidden, state = layer.recurrent(hidden, position_ids, state)
            new_states.append(state)
        return self.lm_head(self.norm(hidden)), new_states

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for RetNet loss_step")
        return self.causal_lm_loss(self.next_token_logits(batch), batch.labels)

    @torch.no_grad()
    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32) -> torch.Tensor:
        input_ids = batch.input_ids
        logits, states = self.forward_recurrent(input_ids)
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_seq_len:
                break
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            position_offset = input_ids.size(1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            logits, states = self.forward_recurrent(next_token, states, position_offset)
        return input_ids
