"""Portable Mamba-2 selective state-space model based on SSD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel
from torch_like.models.sequence.mamba import RMSNorm


@dataclass
class Mamba2Config(SequenceConfig):
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    head_dim: int = 64
    n_groups: int = 1
    chunk_size: int = 64
    norm_eps: float = 1e-5


@dataclass
class SSDState:
    """Fixed-size recurrent state for one SSD layer."""

    ssm: torch.Tensor
    convolution: torch.Tensor


class SSD(nn.Module):
    """Multi-head scalar-A SSD layer with equivalent scan/chunk/recurrent paths.

    This is a portable reference implementation. It preserves Mamba-2's
    head-wise scalar transition and grouped B/C parameterization, but does not
    depend on the optional fused CUDA kernels from the author repository.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        chunk_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_inner = d_model * expand
        if min(d_model, d_state, d_conv, expand, head_dim, n_groups, chunk_size) <= 0:
            raise ValueError("SSD dimensions and chunk_size must be positive")
        if d_inner % head_dim != 0:
            raise ValueError("d_model * expand must be divisible by head_dim")
        n_heads = d_inner // head_dim
        if n_heads % n_groups != 0:
            raise ValueError("number of SSD heads must be divisible by n_groups")

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.xbc_dim = d_inner + 2 * n_groups * d_state

        self.in_proj = nn.Linear(d_model, d_inner + self.xbc_dim + n_heads, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.xbc_dim, d_conv))
        self.conv_bias = nn.Parameter(torch.zeros(self.xbc_dim))
        self.a_log = nn.Parameter(torch.zeros(n_heads))
        self.dt_bias = nn.Parameter(torch.empty(n_heads))
        self.d_skip = nn.Parameter(torch.ones(n_heads))
        self.norm = RMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.conv_weight, a=5**0.5)
        nn.init.uniform_(self.dt_bias, -4.0, -1.0)

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> SSDState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = device or self.in_proj.weight.device
        dtype = dtype or self.in_proj.weight.dtype
        return SSDState(
            ssm=torch.zeros(batch_size, self.n_heads, self.head_dim, self.d_state, device=device, dtype=dtype),
            convolution=torch.zeros(batch_size, self.xbc_dim, self.d_conv - 1, device=device, dtype=dtype),
        )

    def _validate_state(self, state: SSDState, batch_size: int) -> None:
        if not isinstance(state, SSDState):
            raise TypeError("state must be an SSDState")
        expected_ssm = (batch_size, self.n_heads, self.head_dim, self.d_state)
        expected_conv = (batch_size, self.xbc_dim, self.d_conv - 1)
        if state.ssm.shape != expected_ssm or state.convolution.shape != expected_conv:
            raise ValueError(
                f"Expected state shapes {expected_ssm} and {expected_conv}, got "
                f"{tuple(state.ssm.shape)} and {tuple(state.convolution.shape)}"
            )

    def _step(self, projected: torch.Tensor, state: SSDState, active: torch.Tensor | None):
        z, xbc, dt_raw = torch.split(projected, [self.d_inner, self.xbc_dim, self.n_heads], dim=-1)
        window = torch.cat([state.convolution, xbc.unsqueeze(-1)], dim=-1)
        convolved = (window * self.conv_weight.unsqueeze(0)).sum(dim=-1) + self.conv_bias
        convolved = F.silu(convolved)
        next_conv = window[:, :, 1:]

        x, b, c = torch.split(
            convolved,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        x = x.view(x.size(0), self.n_heads, self.head_dim)
        b = b.view(b.size(0), self.n_groups, self.d_state)
        c = c.view(c.size(0), self.n_groups, self.d_state)
        repeats = self.n_heads // self.n_groups
        b = b.repeat_interleave(repeats, dim=1)
        c = c.repeat_interleave(repeats, dim=1)

        dt = F.softplus(dt_raw + self.dt_bias).to(x.dtype)
        decay = torch.exp(-torch.exp(self.a_log).to(x.dtype) * dt)
        next_ssm = state.ssm * decay[:, :, None, None]
        next_ssm = next_ssm + torch.einsum("bh,bhn,bhp->bhpn", dt, b, x)
        output = torch.einsum("bhpn,bhn->bhp", next_ssm, c)
        output = output + self.d_skip.to(x.dtype)[None, :, None] * x

        if active is not None:
            state_mask = active[:, None, None, None]
            next_ssm = torch.where(state_mask, next_ssm, state.ssm)
            next_conv = torch.where(active[:, None, None], next_conv, state.convolution)
            output = output * active[:, None, None].to(output.dtype)

        output = output.reshape(output.size(0), self.d_inner)
        output = self.norm(output) * F.silu(z)
        return self.dropout(self.out_proj(output)), SSDState(next_ssm, next_conv)

    def _scan(
        self,
        x: torch.Tensor,
        state: SSDState | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SSDState]:
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"Expected x with shape (batch, seq_len, {self.d_model})")
        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            raise ValueError("SSD requires at least one token")
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected attention_mask shape ({batch_size}, {seq_len})")
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        self._validate_state(state, batch_size)
        state = SSDState(
            state.ssm.to(device=x.device, dtype=x.dtype),
            state.convolution.to(device=x.device, dtype=x.dtype),
        )
        projected = self.in_proj(x)
        outputs = []
        for index in range(seq_len):
            active = None if attention_mask is None else attention_mask[:, index].to(torch.bool)
            output, state = self._step(projected[:, index], state, active)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: SSDState | None = None,
        *,
        mode: str = "parallel",
        return_state: bool = False,
    ):
        if mode not in {"parallel", "chunkwise", "recurrent"}:
            raise ValueError(f"Unsupported SSD mode: {mode}")
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

    def forward_step(self, x: torch.Tensor, state: SSDState | None = None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError("forward_step expects (batch, d_model) or (batch, 1, d_model)")
        output, state = self.forward(x, state=state, mode="recurrent", return_state=True)
        return output[:, 0], state


class Mamba2Block(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.ssd = SSD(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            head_dim=config.head_dim,
            n_groups=config.n_groups,
            chunk_size=config.chunk_size,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attention_mask=None, state=None, *, mode="parallel"):
        mixed, state = self.ssd(self.norm(x), attention_mask=attention_mask, state=state, mode=mode, return_state=True)
        x = x + mixed
        return x + self.ffn(self.ffn_norm(x)), state


class Mamba2(SequenceModel):
    """Causal language model using portable Mamba-2 SSD blocks."""

    def __init__(self, vocab_size: int, max_seq_len: int, **kwargs):
        super().__init__()
        self.config = Mamba2Config(vocab_size=vocab_size, max_seq_len=max_seq_len, **kwargs)
        if self.config.d_model * self.config.expand % self.config.head_dim != 0:
            raise ValueError("d_model * expand must be divisible by head_dim")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, self.config.d_model)
        self.position_embedding = nn.Embedding(max_seq_len, self.config.d_model)
        self.layers = nn.ModuleList([Mamba2Block(self.config) for _ in range(self.config.n_layers)])
        self.norm = RMSNorm(self.config.d_model, self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)

    def _forward_state(self, input_ids, attention_mask=None, states=None, *, mode="parallel", position_offset=0):
        if input_ids.dim() != 2:
            raise ValueError("Mamba2 expects input_ids with shape (batch, seq_len)")
        if input_ids.size(1) + position_offset > self.max_seq_len:
            raise ValueError("input sequence and position_offset exceed max_seq_len")
        if states is None:
            states = [None] * len(self.layers)
        if len(states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} layer states, got {len(states)}")
        positions = torch.arange(
            position_offset, position_offset + input_ids.size(1), device=input_ids.device
        ).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        new_states = []
        for layer, state in zip(self.layers, states, strict=True):
            hidden, state = layer(hidden, attention_mask, state, mode=mode)
            new_states.append(state)
        hidden = self.norm(hidden)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return self.lm_head(hidden), new_states

    def forward(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None):
        del causal_mask
        if position_ids is not None:
            expected = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            if not torch.equal(position_ids, expected.expand_as(input_ids)):
                raise ValueError("Mamba2 only supports contiguous position_ids starting at zero in forward")
        return self._forward_state(input_ids, attention_mask)[0]

    def forward_chunkwise(self, input_ids, attention_mask=None):
        return self._forward_state(input_ids, attention_mask, mode="chunkwise")[0]

    def forward_recurrent(self, input_ids, states=None, position_offset: int = 0):
        return self._forward_state(input_ids, states=states, mode="recurrent", position_offset=position_offset)

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for loss_step")
        logits = self.next_token_logits(batch)
        return self.causal_lm_loss(logits, batch.labels)

    @torch.no_grad()
    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32) -> torch.Tensor:
        input_ids = batch.input_ids
        logits, states = self.forward_recurrent(input_ids)
        position_offset = input_ids.size(1)
        for _ in range(max_new_tokens):
            if position_offset >= self.max_seq_len:
                break
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            logits, states = self.forward_recurrent(next_token, states, position_offset)
            position_offset += 1
        return input_ids
