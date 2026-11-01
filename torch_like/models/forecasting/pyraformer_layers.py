"""Native indexed Pyraformer operators (independent mathematical implementation).

Reference: Liu et al., ICLR 2022; Apache-2.0 ant-research/Pyraformer,
84af4dbd93b7b96975b5034f0dde412005260123. No TVM or dense mask is used.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _pyramid_graph(
    fine_length: int, factors: tuple[int, ...], inner_size: int
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build O(S*K) neighbor storage; callers validate nonempty coarse levels.

    Parent assignment absorbs ragged tails into the last parent, unlike the
    strided construction convolutions, which drop them. Padding uses valid key
    indices but MUST be masked before softmax to avoid stealing probability.
    """
    lengths = [fine_length]
    for factor in factors:
        lengths.append(lengths[-1] // factor)
    offsets = [0]
    for size in lengths[:-1]:
        offsets.append(offsets[-1] + size)
    rows = [set() for _ in range(sum(lengths))]
    radius = inner_size // 2
    for level, (offset, size) in enumerate(zip(offsets, lengths, strict=True)):
        for local in range(size):
            node = offset + local
            rows[node].update(range(offset + max(0, local - radius), offset + min(size, local + radius + 1)))
            if level < len(factors):
                parent = offsets[level + 1] + min(local // factors[level], lengths[level + 1] - 1)
                rows[node].add(parent)
                rows[parent].add(node)
    width = max(map(len, rows))
    neighbors = torch.zeros(len(rows), width, dtype=torch.long)
    valid = torch.zeros(len(rows), width, dtype=torch.bool)
    for node, row in enumerate(rows):
        keys = sorted(row)
        neighbors[node, : len(keys)] = torch.tensor(keys, dtype=torch.long)
        valid[node, : len(keys)] = True
    last = torch.tensor([offset + size - 1 for offset, size in zip(offsets, lengths, strict=True)], dtype=torch.long)
    return tuple(lengths), neighbors, valid, last


class _PyramidalAttention(nn.Module):
    """Postnorm self-attention over padded per-query neighbor lists.

    Scores have shape (B,heads,S,K), gathered keys/values (B,heads,S,K,dh).
    For fixed factors/window K is bounded independently of sequence length.
    This does not promise a wall-clock speedup over optimized dense attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(projection.weight)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = [
            projection(x).reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        keys = k[:, :, neighbors, :]
        values = v[:, :, neighbors, :]
        scores = ((q / self.head_dim**0.5).unsqueeze(-2) * keys).sum(-1)
        weights = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        context = (self.attn_dropout(weights).unsqueeze(-1) * values).sum(-2)
        context = context.transpose(1, 2).reshape(b, s, d)
        return self.norm(x + self.proj_dropout(self.out_proj(context)))


class _PyramidEncoderLayer(nn.Module):
    """Source-aligned postnorm attention followed by a GELU feed-forward block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = _PyramidalAttention(d_model, n_heads, dropout)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        state = self.attention(x, neighbors, valid)
        hidden = self.ff_dropout1(F.gelu(self.ff1(state)))
        return self.norm(state + self.ff_dropout2(self.ff2(hidden)))


class _BottleneckCSCM(nn.Module):
    """Convolutional coarser-scale construction with source BatchNorm semantics."""

    def __init__(self, d_model: int, d_bottleneck: int, factors: tuple[int, ...]):
        super().__init__()
        self.factors = factors
        self.down = nn.Linear(d_model, d_bottleneck)
        self.convs = nn.ModuleList([nn.Conv1d(d_bottleneck, d_bottleneck, f, stride=f) for f in factors])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d_bottleneck) for _ in factors])
        self.up = nn.Linear(d_bottleneck, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarsest = x.shape[1]
        for factor in self.factors:
            coarsest //= factor
        if self.training and x.shape[0] * coarsest <= 1:
            # Validate every scale before running the first BN, so rejected
            # training batches cannot partially mutate running statistics.
            raise ValueError(
                "Pyraformer BatchNorm training requires batch * coarsest_length > 1; use eval for inference"
            )
        state = self.down(x).transpose(1, 2)
        levels = []
        for conv, bn in zip(self.convs, self.bns, strict=True):
            state = F.elu(bn(conv(state)))
            levels.append(state.transpose(1, 2))
        coarse = self.up(torch.cat(levels, dim=1))
        return self.norm(torch.cat((x, coarse), dim=1))
