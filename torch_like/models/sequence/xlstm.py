"""Portable matrix-memory xLSTM reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel
from torch_like.models.sequence.mamba import RMSNorm


@dataclass
class xLSTMConfig(SequenceConfig):
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True


class mLSTMCell(nn.Module):
    """Multi-head matrix-memory LSTM with stabilized exponential gates."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.input_gate = nn.Linear(embed_dim, num_heads)
        self.forget_gate = nn.Linear(embed_dim, num_heads)
        self.output_gate = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = device or self.q_proj.weight.device
        dtype = dtype or self.q_proj.weight.dtype
        matrix = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
        normalizer = torch.zeros(batch_size, self.num_heads, self.head_dim, device=device, dtype=dtype)
        stabilizer = torch.full((batch_size, self.num_heads), -torch.inf, device=device, dtype=dtype)
        return matrix, normalizer, stabilizer

    def _validate_state(self, state, batch_size):
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError("mLSTM state must be a (matrix, normalizer, stabilizer) tuple")
        expected_matrix = (batch_size, self.num_heads, self.head_dim, self.head_dim)
        expected_vector = (batch_size, self.num_heads, self.head_dim)
        expected_scalar = (batch_size, self.num_heads)
        if state[0].shape != expected_matrix or state[1].shape != expected_vector or state[2].shape != expected_scalar:
            raise ValueError(f"Expected state shapes {expected_matrix}, {expected_vector}, {expected_scalar}")

    def forward(self, x, state=None, attention_mask=None, *, return_state=False):
        if x.dim() != 3 or x.size(-1) != self.embed_dim:
            raise ValueError(f"Expected x with shape (batch, seq_len, {self.embed_dim})")
        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            raise ValueError("mLSTMCell requires at least one token")
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected attention_mask shape ({batch_size}, {seq_len})")
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        self._validate_state(state, batch_size)
        matrix, normalizer, stabilizer = (item.to(device=x.device, dtype=x.dtype) for item in state)
        shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape) / self.head_dim**0.5
        k = self.k_proj(x).view(shape) / self.head_dim**0.5
        v = self.v_proj(x).view(shape)
        log_input = self.input_gate(x)
        log_forget = F.logsigmoid(self.forget_gate(x))
        output_gate = torch.sigmoid(self.output_gate(x)).view(shape)
        outputs = []

        for index in range(seq_len):
            previous = (matrix, normalizer, stabilizer)
            next_stabilizer = torch.maximum(log_forget[:, index] + stabilizer, log_input[:, index])
            forget = torch.exp(log_forget[:, index] + stabilizer - next_stabilizer)
            input_weight = torch.exp(log_input[:, index] - next_stabilizer)
            matrix = forget[..., None, None] * matrix + input_weight[..., None, None] * torch.einsum(
                "bhv,bhk->bhvk", v[:, index], k[:, index]
            )
            normalizer = forget[..., None] * normalizer + input_weight[..., None] * k[:, index]
            stabilizer = next_stabilizer
            if attention_mask is not None:
                valid = attention_mask[:, index].bool().view(batch_size, 1)
                matrix = torch.where(valid[..., None, None], matrix, previous[0])
                normalizer = torch.where(valid[..., None], normalizer, previous[1])
                stabilizer = torch.where(valid, stabilizer, previous[2])
            numerator = torch.einsum("bhvk,bhk->bhv", matrix, q[:, index])
            denominator = torch.einsum("bhk,bhk->bh", normalizer, q[:, index]).abs().clamp_min(1.0)
            output = output_gate[:, index] * numerator / denominator[..., None]
            if attention_mask is not None:
                output = output * attention_mask[:, index].to(output.dtype).view(batch_size, 1, 1)
            outputs.append(output)

        hidden = torch.stack(outputs, dim=1).reshape(batch_size, seq_len, self.embed_dim)
        hidden = self.dropout(self.out_proj(hidden))
        result_state = (matrix, normalizer, stabilizer)
        return (hidden, result_state) if return_state else hidden

    def forward_step(self, x, state=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError("forward_step expects (batch, embed_dim) or (batch, 1, embed_dim)")
        output, state = self.forward(x, state, return_state=True)
        return output[:, 0], state


class _xLSTMBlock(nn.Module):
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mixer = mLSTMCell(config.d_model, config.n_heads, config.dropout)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attention_mask=None, state=None):
        mixed, state = self.mixer(self.norm(x), state, attention_mask, return_state=True)
        x = x + mixed
        return x + self.ffn(self.ffn_norm(x)), state


class xLSTM(SequenceModel):
    """Causal language model built from portable mLSTM residual blocks."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("xLSTM dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = xLSTMConfig(
            vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout, norm_eps, tie_word_embeddings
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_xLSTMBlock(self._config) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward_features(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None):
        del position_ids, causal_mask
        hidden, _ = self._forward_state(input_ids, attention_mask)
        return hidden

    def _forward_state(self, input_ids, attention_mask=None, states=None):
        if input_ids.dim() != 2:
            raise ValueError("xLSTM expects input_ids with shape (batch, seq_len)")
        if input_ids.size(1) == 0 or input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"Expected 1 <= seq_len <= {self.max_seq_len}")
        if states is None:
            states = [None] * len(self.layers)
        if len(states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} states, got {len(states)}")
        hidden = self.token_embedding(input_ids)
        new_states = []
        for layer, state in zip(self.layers, states, strict=True):
            hidden, state = layer(hidden, attention_mask, state)
            new_states.append(state)
        return self.norm(hidden), new_states

    def forward(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None):
        return self.lm_head(self.forward_features(input_ids, attention_mask, position_ids, causal_mask))

    def forward_recurrent(self, input_ids, states=None):
        hidden, states = self._forward_state(input_ids, states=states)
        return self.lm_head(hidden), states

    def loss_step(self, batch: SequenceBatch):
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for xLSTM loss_step")
        return self.causal_lm_loss(self.next_token_logits(batch), batch.labels)

    @torch.no_grad()
    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32):
        input_ids = batch.input_ids
        logits, states = self.forward_recurrent(input_ids)
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_seq_len:
                break
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            logits, states = self.forward_recurrent(next_token, states)
        return input_ids
