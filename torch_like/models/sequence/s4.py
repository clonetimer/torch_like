"""Structured State Space (S4) causal sequence model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


@dataclass
class S4Config(SequenceConfig):
    d_state: int = 64
    rank: int = 1


class S4Layer(nn.Module):
    """Portable DPLR state-space layer using a recurrent scan."""

    def __init__(self, d_model: int, d_state: int, rank: int, dropout: float):
        super().__init__()
        if min(d_state, rank) <= 0:
            raise ValueError("d_state and rank must be positive")
        self.d_state = d_state
        self.log_dt = nn.Parameter(torch.empty(d_model).uniform_(-6.0, -2.0))
        self.a_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.p = nn.Parameter(torch.randn(d_model, d_state, rank) / d_state**0.5)
        self.q = nn.Parameter(torch.randn(d_model, d_state, rank) / d_state**0.5)
        self.b = nn.Parameter(torch.randn(d_model, d_state) / d_state**0.5)
        self.c = nn.Parameter(torch.randn(d_model, d_state) / d_state**0.5)
        self.d = nn.Parameter(torch.ones(d_model))
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _discrete_parameters(self, dtype: torch.dtype, device: torch.device):
        diagonal = -torch.exp(self.a_log).to(device=device, dtype=dtype)
        low_rank = torch.einsum("cnr,cmr->cnm", self.p, self.q).to(device=device, dtype=dtype)
        a = torch.diag_embed(diagonal.expand(self.p.size(0), -1)) - low_rank / self.d_state
        dt = F.softplus(self.log_dt).to(device=device, dtype=dtype)
        identity = torch.eye(self.d_state, device=device, dtype=dtype).expand(a.size(0), -1, -1)
        left = identity - 0.5 * dt[:, None, None] * a
        right = identity + 0.5 * dt[:, None, None] * a
        a_bar = torch.linalg.solve(left, right)
        b_bar = torch.linalg.solve(
            left,
            dt[:, None, None] * self.b.to(device=device, dtype=dtype).unsqueeze(-1),
        ).squeeze(-1)
        return a_bar, b_bar

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        a_bar, b_bar = self._discrete_parameters(x.dtype, x.device)
        state = x.new_zeros(x.size(0), x.size(2), self.d_state)
        outputs = []
        for step in range(x.size(1)):
            next_state = torch.einsum("cij,bcj->bci", a_bar, state) + b_bar.unsqueeze(0) * x[:, step].unsqueeze(-1)
            if attention_mask is not None:
                active = attention_mask[:, step, None, None].bool()
                state = torch.where(active, next_state, state)
            else:
                state = next_state
            y = torch.einsum("cn,bcn->bc", self.c.to(x.dtype), state) + self.d.to(x.dtype) * x[:, step]
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        return self.dropout(y * torch.sigmoid(self.gate(x)))


class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, rank: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4 = S4Layer(d_model, d_state, rank, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.s4(self.norm(x), attention_mask)
        return x + self.ffn(self.ffn_norm(x))


class S4(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 1,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        d_state: int = 64,
        rank: int = 1,
    ):
        super().__init__()
        del n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = S4Config(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=1,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            d_state=d_state,
            rank=rank,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([S4Block(d_model, d_state, rank, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None):
        del causal_mask
        if input_ids.dim() != 2 or input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"S4 expects input_ids with shape (batch, seq_len <= {self.max_seq_len})")
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return self.lm_head(self.norm(hidden))

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for loss_step")
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
