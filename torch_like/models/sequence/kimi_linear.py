"""Portable Kimi Delta Attention and a lightweight Kimi Linear model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel
from torch_like.models.sequence.mamba import RMSNorm


@dataclass
class KimiLinearConfig(SequenceConfig):
    full_attention_interval: int = 0
    chunk_size: int = 64
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True


class KimiDeltaAttention(nn.Module):
    """Fine-grained gated delta attention implemented as a portable scan."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, chunk_size: int = 64):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.beta_proj = nn.Linear(embed_dim, embed_dim)
        self.decay_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=device or self.q_proj.weight.device,
            dtype=dtype or self.q_proj.weight.dtype,
        )

    def _project(self, x: torch.Tensor):
        shape = (x.size(0), x.size(1), self.num_heads, self.head_dim)
        q = F.normalize(self.q_proj(x).view(shape), dim=-1)
        k = F.normalize(self.k_proj(x).view(shape), dim=-1)
        v = self.v_proj(x).view(shape)
        beta = torch.sigmoid(self.beta_proj(x).view(shape))
        decay = torch.exp(-F.softplus(self.decay_proj(x).view(shape)))
        return q, k, v, beta, decay

    def _scan(self, x, state=None, attention_mask=None):
        if x.dim() != 3 or x.size(-1) != self.embed_dim:
            raise ValueError(f"Expected x with shape (batch, seq_len, {self.embed_dim})")
        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            raise ValueError("KimiDeltaAttention requires at least one token")
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected attention_mask shape ({batch_size}, {seq_len})")
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        expected = (batch_size, self.num_heads, self.head_dim, self.head_dim)
        if state.shape != expected:
            raise ValueError(f"Expected state shape {expected}, got {tuple(state.shape)}")
        state = state.to(device=x.device, dtype=x.dtype)
        q, k, v, beta, decay = self._project(x)
        outputs = []
        for index in range(seq_len):
            previous = state
            decayed = state * decay[:, index].unsqueeze(-2)
            prediction = torch.einsum("bhd,bhdv->bhv", k[:, index], decayed)
            correction = beta[:, index] * (v[:, index] - prediction)
            updated = decayed + torch.einsum("bhd,bhv->bhdv", k[:, index], correction)
            if attention_mask is not None:
                valid = attention_mask[:, index].bool().view(batch_size, 1, 1, 1)
                state = torch.where(valid, updated, previous)
            else:
                state = updated
            output = torch.einsum("bhd,bhdv->bhv", q[:, index], state)
            if attention_mask is not None:
                output = output * attention_mask[:, index].to(output.dtype).view(batch_size, 1, 1)
            outputs.append(output)
        hidden = torch.stack(outputs, dim=1).reshape(batch_size, seq_len, self.embed_dim)
        hidden = F.silu(self.gate_proj(x)) * hidden
        return self.dropout(self.out_proj(hidden)), state

    def forward(self, x, attention_mask=None, state=None, *, mode="parallel", return_state=False):
        if mode not in {"parallel", "chunkwise", "recurrent"}:
            raise ValueError(f"Unsupported KDA mode: {mode}")
        if mode != "chunkwise":
            output, state = self._scan(x, state, attention_mask)
        else:
            chunks = []
            for start in range(0, x.size(1), self.chunk_size):
                mask = None if attention_mask is None else attention_mask[:, start : start + self.chunk_size]
                output, state = self._scan(x[:, start : start + self.chunk_size], state, mask)
                chunks.append(output)
            output = torch.cat(chunks, dim=1)
        return (output, state) if return_state else output

    def forward_step(self, x, state=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError("forward_step expects (batch, embed_dim) or (batch, 1, embed_dim)")
        output, state = self.forward(x, state=state, mode="recurrent", return_state=True)
        return output[:, 0], state


class _CausalAttention(nn.Module):
    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)

    def forward(self, x, attention_mask=None):
        seq_len = x.size(1)
        causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu(1)
        key_padding_mask = None if attention_mask is None else ~attention_mask.to(torch.bool)
        return self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]


class _KimiLinearBlock(nn.Module):
    def __init__(self, config: KimiLinearConfig, use_full_attention: bool):
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.is_kda = not use_full_attention
        self.mixer = (
            KimiDeltaAttention(config.d_model, config.n_heads, config.dropout, config.chunk_size)
            if self.is_kda
            else _CausalAttention(config)
        )
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attention_mask=None, state=None, mode="parallel"):
        if self.is_kda:
            mixed, state = self.mixer(self.norm(x), attention_mask, state, mode=mode, return_state=True)
        else:
            if state is not None or mode == "recurrent":
                raise ValueError("Full-attention layers do not expose a fixed-size recurrent state")
            mixed = self.mixer(self.norm(x), attention_mask)
        x = x + mixed
        return x + self.ffn(self.ffn_norm(x)), state


class KimiLinear(SequenceModel):
    """Small random-initialized KDA language model, not the 48B checkpoint."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        full_attention_interval: int = 0,
        chunk_size: int = 64,
        norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("KimiLinear dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if full_attention_interval < 0:
            raise ValueError("full_attention_interval must be non-negative")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = KimiLinearConfig(
            vocab_size,
            max_seq_len,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            dropout,
            full_attention_interval,
            chunk_size,
            norm_eps,
            tie_word_embeddings,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                _KimiLinearBlock(
                    self._config,
                    use_full_attention=(full_attention_interval > 0 and (index + 1) % full_attention_interval == 0),
                )
                for index in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward_features(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None, *, mode="parallel"):
        del position_ids, causal_mask
        if input_ids.dim() != 2:
            raise ValueError("KimiLinear expects input_ids with shape (batch, seq_len)")
        if input_ids.size(1) == 0 or input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"Expected 1 <= seq_len <= {self.max_seq_len}")
        hidden = self.token_embedding(input_ids)
        for layer in self.layers:
            hidden, _ = layer(hidden, attention_mask, mode=mode)
        return self.norm(hidden)

    def forward(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None, *, mode="parallel"):
        features = self.forward_features(input_ids, attention_mask, position_ids, causal_mask, mode=mode)
        return self.lm_head(features)

    def forward_recurrent(self, input_ids, states=None):
        if self._config.full_attention_interval > 0:
            raise ValueError("forward_recurrent requires full_attention_interval=0")
        if input_ids.dim() != 2:
            raise ValueError("KimiLinear expects input_ids with shape (batch, seq_len)")
        if states is None:
            states = [None] * len(self.layers)
        if len(states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} states, got {len(states)}")
        hidden = self.token_embedding(input_ids)
        new_states = []
        for layer, state in zip(self.layers, states, strict=True):
            hidden, state = layer(hidden, state=state, mode="recurrent")
            new_states.append(state)
        return self.lm_head(self.norm(hidden)), new_states

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for KimiLinear loss_step")
        return self.causal_lm_loss(self.next_token_logits(batch), batch.labels)

    @torch.no_grad()
    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32) -> torch.Tensor:
        input_ids = batch.input_ids
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_seq_len:
                break
            next_token = self(input_ids)[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
