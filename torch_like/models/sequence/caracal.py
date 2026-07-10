"""Caracal-style spectral causal sequence model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class MultiHeadFourier(nn.Module):
    def __init__(self, d_model: int, dropout: float, conv_kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.pre_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.value_proj = nn.Linear(d_model, d_model)
        self.gate_linear = nn.Linear(d_model, d_model)
        self.gate_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.pre_conv(x)
        value = self.value_proj(local)
        gate = self.gate_conv(F.silu(self.gate_linear(local)))

        seq_len = value.size(1)
        padded_len = seq_len * 2
        value_fft = torch.fft.rfft(F.pad(value, (0, 0, 0, seq_len)), dim=1)
        gate_fft = torch.fft.rfft(F.pad(gate, (0, 0, 0, seq_len)), dim=1)
        mixed_fft = value_fft * gate_fft
        mixed = torch.fft.irfft(mixed_fft, n=padded_len, dim=1)[:, :seq_len, :]
        return self.out_proj(self.dropout(mixed))


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(1)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        if self.window_size < seq_len:
            distance = torch.arange(seq_len, device=x.device)
            local = (distance.unsqueeze(1) - distance.unsqueeze(0)).abs() <= self.window_size
            local = ~local
            attn_mask = causal | local
        else:
            attn_mask = causal
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(dtype=torch.bool)
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class CaracalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_attention: bool,
        window_size: int,
        conv_kernel_size: int,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.mhf = MultiHeadFourier(d_model=d_model, dropout=dropout, conv_kernel_size=conv_kernel_size)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = SlidingWindowAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, window_size=window_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm_1(x + self.mhf(x))
        if self.use_attention:
            x = self.norm_attn(x + self.attn(x, attention_mask=attention_mask))
        return self.norm_2(x + self.ffn(x))


class Caracal(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.1,
        attention_interval: int = 3,
        window_size: int = 16,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.attention_interval = max(1, attention_interval)
        self._config = SequenceConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                CaracalBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_attention=((idx + 1) % self.attention_interval == 0),
                    window_size=window_size,
                    conv_kernel_size=conv_kernel_size,
                )
                for idx in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_ids, causal_mask
        if input_ids.dim() != 2:
            raise ValueError("Caracal expects input_ids with shape (batch, seq_len)")
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for loss_step")
        logits = self.next_token_logits(batch)
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError(f"Caracal expects labels with shape {tuple(batch.input_ids.shape)}, got {tuple(batch.labels.shape)}")
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
