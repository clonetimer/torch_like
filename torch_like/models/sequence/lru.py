"""LRU-style linear recurrent unit for the sequence domain."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel
from torch_like.models.sequence.mamba import RMSNorm


class LRUMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float,
        gate_mode: str = "sigmoid",
        skip_mode: str = "learnable",
        radius_min: float = 0.8,
        radius_max: float = 0.99,
        phase_max: float = math.pi / 2,
    ):
        super().__init__()
        if gate_mode not in {"sigmoid", "none"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        if skip_mode not in {"learnable", "none"}:
            raise ValueError(f"Unsupported skip_mode: {skip_mode}")
        if not (0.0 < radius_min < radius_max < 1.0):
            raise ValueError("Expected 0 < radius_min < radius_max < 1")
        if phase_max <= 0.0:
            raise ValueError("phase_max must be positive")
        self.d_model = d_model
        self.d_state = d_state
        self.gate_mode = gate_mode
        self.skip_mode = skip_mode
        self.in_proj = nn.Linear(d_model, d_state * 4)
        self.out_proj = nn.Linear(d_state * 2, d_model)
        self.dropout = nn.Dropout(dropout)

        radius_grid = torch.linspace(radius_min, radius_max, d_state)
        phase_grid = torch.linspace(0.0, phase_max, d_state)
        self.radius_logit = nn.Parameter(torch.logit(radius_grid.clamp(1e-5, 1 - 1e-5)))
        self.phase = nn.Parameter(phase_grid)
        self.skip = nn.Parameter(torch.ones(d_state * 2))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        projected = self.in_proj(x)
        input_real, input_imag, gate_real, gate_imag = projected.chunk(4, dim=-1)

        if attention_mask is not None:
            mask = attention_mask.to(x.dtype).unsqueeze(-1)
            input_real = input_real * mask
            input_imag = input_imag * mask
            gate_real = gate_real * mask
            gate_imag = gate_imag * mask
        else:
            mask = None

        radius = torch.sigmoid(self.radius_logit).to(x.dtype)
        cos_phase = torch.cos(self.phase).to(x.dtype)
        sin_phase = torch.sin(self.phase).to(x.dtype)
        lambda_real = radius * cos_phase
        lambda_imag = radius * sin_phase

        state_real = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        state_imag = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            u_real = input_real[:, t, :]
            u_imag = input_imag[:, t, :]

            next_real = lambda_real.unsqueeze(0) * state_real - lambda_imag.unsqueeze(0) * state_imag + u_real
            next_imag = lambda_real.unsqueeze(0) * state_imag + lambda_imag.unsqueeze(0) * state_real + u_imag
            state_real, state_imag = next_real, next_imag

            output_t = torch.cat([state_real, state_imag], dim=-1)
            if self.gate_mode == "sigmoid":
                gate_t = torch.sigmoid(torch.cat([gate_real[:, t, :], gate_imag[:, t, :]], dim=-1))
                output_t = gate_t * output_t
            if self.skip_mode == "learnable":
                output_t = output_t + self.skip.view(1, -1) * output_t
            if mask is not None:
                output_t = output_t * mask[:, t, :]
            outputs.append(output_t)

        hidden = torch.stack(outputs, dim=1)
        return self.dropout(self.out_proj(hidden))


class LRUBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_ff: int,
        dropout: float,
        gate_mode: str = "sigmoid",
        skip_mode: str = "learnable",
        radius_min: float = 0.8,
        radius_max: float = 0.99,
        phase_max: float = math.pi / 2,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = LRUMixer(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            gate_mode=gate_mode,
            skip_mode=skip_mode,
            radius_min=radius_min,
            radius_max=radius_max,
            phase_max=phase_max,
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


class LRU(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        d_state: int = 64,
        gate_mode: str = "sigmoid",
        skip_mode: str = "learnable",
        radius_min: float = 0.8,
        radius_max: float = 0.99,
        phase_max: float = math.pi / 2,
    ):
        super().__init__()
        del n_heads
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
                LRUBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_ff=d_ff,
                    dropout=dropout,
                    gate_mode=gate_mode,
                    skip_mode=skip_mode,
                    radius_min=radius_min,
                    radius_max=radius_max,
                    phase_max=phase_max,
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
            raise ValueError("LRU expects input_ids with shape (batch, seq_len)")
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
            raise ValueError(
                f"LRU expects labels with shape {tuple(batch.input_ids.shape)}, got {tuple(batch.labels.shape)}"
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
