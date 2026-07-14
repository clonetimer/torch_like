"""Hyena causal sequence model with implicit long convolutions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


@dataclass
class HyenaConfig(SequenceConfig):
    filter_order: int = 64
    short_kernel: int = 3


def causal_fft_convolution(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    length = x.size(1)
    fft_size = 2 * length
    x_frequency = torch.fft.rfft(x.float(), n=fft_size, dim=1)
    kernel_frequency = torch.fft.rfft(kernel.float(), n=fft_size, dim=0)
    return torch.fft.irfft(x_frequency * kernel_frequency.unsqueeze(0), n=fft_size, dim=1)[:, :length].to(x.dtype)


class ImplicitHyenaFilter(nn.Module):
    def __init__(self, d_model: int, filter_order: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(max_seq_len), filter_order // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)
        feature_dim = 1 + 2 * frequencies.numel()
        self.mlp = nn.Sequential(nn.Linear(feature_dim, filter_order), nn.GELU(), nn.Linear(filter_order, d_model))
        self.log_decay = nn.Parameter(torch.zeros(d_model))

    def forward(self, length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        time = torch.linspace(0, 1, length, device=device).unsqueeze(-1)
        phase = time * self.frequencies.to(device=device).unsqueeze(0)
        features = torch.cat([time, torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=-1)
        decay = torch.exp(-time * torch.nn.functional.softplus(self.log_decay).unsqueeze(0))
        return (self.mlp(features) * decay).to(dtype)


class HyenaOperator(nn.Module):
    def __init__(self, d_model: int, filter_order: int, max_seq_len: int, short_kernel: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(d_model, 3 * d_model)
        self.short_filter = nn.Conv1d(
            3 * d_model, 3 * d_model, short_kernel, padding=short_kernel - 1, groups=3 * d_model
        )
        self.filter = ImplicitHyenaFilter(d_model, filter_order, max_seq_len)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x).transpose(1, 2)
        projected = self.short_filter(projected)[..., : x.size(1)].transpose(1, 2)
        gate_1, gate_2, value = projected.chunk(3, dim=-1)
        kernel = self.filter(x.size(1), x.dtype, x.device)
        value = causal_fft_convolution(gate_2 * value, kernel)
        return self.dropout(self.output_projection(gate_1 * value))


class HyenaBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, filter_order: int, max_seq_len: int, short_kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(d_model, filter_order, max_seq_len, short_kernel, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.hyena(self.norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Hyena(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 1,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        filter_order: int = 64,
        short_kernel: int = 3,
    ):
        super().__init__()
        del n_heads
        if filter_order < 2 or short_kernel <= 0:
            raise ValueError("filter_order must be at least 2 and short_kernel must be positive")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = HyenaConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=1,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            filter_order=filter_order,
            short_kernel=short_kernel,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [HyenaBlock(d_model, d_ff, filter_order, max_seq_len, short_kernel, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None):
        del position_ids, causal_mask
        if input_ids.dim() != 2 or input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"Hyena expects input_ids with shape (batch, seq_len <= {self.max_seq_len})")
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
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
