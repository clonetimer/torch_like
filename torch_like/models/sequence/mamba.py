"""Mamba-style selective state-space model for the sequence domain."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SelectiveStateSpaceMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.empty(self.d_inner))
        nn.init.uniform_(self.dt_bias, math.log(1e-3), math.log(1e-1))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_proj, z = self.in_proj(x).chunk(2, dim=-1)

        conv_input = x_proj.transpose(1, 2)
        conv_output = self.conv1d(conv_input)[..., :seq_len].transpose(1, 2)
        conv_output = F.silu(conv_output)

        params = self.x_proj(conv_output)
        dt_part, b_part, c_part = torch.split(params, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_part) + self.dt_bias.view(1, 1, -1))

        a = -torch.exp(self.A_log).to(x.dtype)
        state = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        direct = self.D.to(x.dtype).view(1, self.d_inner)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
            conv_output = conv_output * mask
            z = z * mask
        else:
            mask = None

        for t in range(seq_len):
            x_t = conv_output[:, t, :]
            dt_t = delta[:, t, :]
            b_t = b_part[:, t, :].unsqueeze(1)
            c_t = c_part[:, t, :].unsqueeze(1)

            transition = torch.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
            drive = dt_t.unsqueeze(-1) * b_t * x_t.unsqueeze(-1)
            state = transition * state + drive

            y_t = (state * c_t).sum(dim=-1) + direct * x_t
            if mask is not None:
                y_t = y_t * mask[:, t, :]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = SelectiveStateSpaceMixer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), attention_mask=attention_mask)
        return x + self.ffn(self.ffn_norm(x))


class Mamba(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ):
        super().__init__()
        del n_heads
        if dt_rank is None:
            dt_rank = max(1, math.ceil(d_model / 16))
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = SequenceConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=1,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del causal_mask
        if input_ids.dim() != 2:
            raise ValueError("Mamba expects input_ids with shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        hidden = self.norm(hidden)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return self.lm_head(hidden)

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for loss_step")
        logits = self.next_token_logits(batch)
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError(f"Mamba expects labels with shape {tuple(batch.input_ids.shape)}, got {tuple(batch.labels.shape)}")
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
