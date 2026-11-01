"""Native operators for Zheng et al.'s GMAN (AAAI 2020).

Mathematical reference: Apache-2.0 zhengchuanpan/GMAN, commit
45ed232f4ee6a7ee9dd248d07f4952b697ecf320, METR/PeMS model.py and tf_utils.py.
This independent PyTorch implementation keeps the attention/fusion equations,
not TensorFlow checkpoints. Population BN uses initial running variance one
instead of TF tensor-EMA zero; decay is fixed and masks use negative infinity.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _ChannelBatchNorm(nn.Module):
    """Channel-last population moments, including valid singleton reductions.

    Unlike torch BatchNorm's running variance, the EMA has no unbiased sample
    correction. Initial variance one is an explicit portable cold-eval choice.
    Training moments span every time/node in the call, not only a causal prefix.
    """

    def __init__(self, channels: int, decay: float):
        super().__init__()
        self.decay = decay
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            axes = tuple(range(x.ndim - 1))
            variance, mean = torch.var_mean(x, dim=axes, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), 1.0 - self.decay)
                self.running_var.lerp_(variance.detach(), 1.0 - self.decay)
                self.num_batches_tracked.add_(1)
        else:
            mean, variance = self.running_mean, self.running_var
        return (x - mean) * torch.rsqrt(variance + 1e-3) * self.weight + self.bias


def _linear(in_features: int, out_features: int, *, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class _FC(nn.Module):
    """Input dropout → positionwise affine → optional BN/ReLU.

    Linear-only layers never normalize, even when batch_norm is enabled.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activate: bool,
        batch_norm: bool,
        bn_decay: float,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.activate = activate
        self.dropout = nn.Dropout(dropout)
        self.linear = _linear(in_features, out_features, bias=bias)
        self.norm = _ChannelBatchNorm(out_features, bn_decay) if activate and batch_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(self.dropout(x))
        if self.norm is not None:
            x = self.norm(x)
        return F.relu(x) if self.activate else x


def _mlp(
    in_features: int,
    hidden: int,
    out_features: int,
    *,
    batch_norm: bool,
    bn_decay: float,
    dropout: float = 0.0,
) -> nn.Sequential:
    return nn.Sequential(
        _FC(in_features, hidden, activate=True, batch_norm=batch_norm, bn_decay=bn_decay, dropout=dropout),
        _FC(hidden, out_features, activate=False, batch_norm=batch_norm, bn_decay=bn_decay, dropout=dropout),
    )


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    *,
    temporal: bool,
    causal: bool = False,
) -> torch.Tensor:
    """Dense attention over nodes or time, preserving contiguous head channels."""
    batch, query_steps, nodes, width = query.shape
    depth = width // heads
    order = (0, 2, 3, 1, 4) if temporal else (0, 1, 3, 2, 4)
    q, k, v = [
        tensor.reshape(batch, tensor.size(1), nodes, heads, depth).permute(order) for tensor in (query, key, value)
    ]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(depth)
    if causal:
        allowed = (
            torch.arange(key.size(1), device=key.device)[None, :]
            <= torch.arange(query_steps, device=key.device)[:, None]
        )
        scores = scores.masked_fill(~allowed, -torch.inf)
    output = scores.softmax(dim=-1) @ v
    inverse = (0, 3, 1, 2, 4) if temporal else (0, 1, 3, 2, 4)
    return output.permute(inverse).reshape(batch, query_steps, nodes, width)


class _SpatialAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, batch_norm: bool, bn_decay: float):
        super().__init__()
        self.n_heads = n_heads
        opts = dict(activate=True, batch_norm=batch_norm, bn_decay=bn_decay)
        self.query = _FC(2 * d_model, d_model, **opts)
        self.key = _FC(2 * d_model, d_model, **opts)
        self.value = _FC(2 * d_model, d_model, **opts)
        self.output = _mlp(d_model, d_model, d_model, batch_norm=batch_norm, bn_decay=bn_decay)

    def forward(self, x: torch.Tensor, ste: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((x, ste), dim=-1)
        return self.output(
            _attend(self.query(joined), self.key(joined), self.value(joined), self.n_heads, temporal=False)
        )


class _TemporalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, batch_norm: bool, bn_decay: float, causal: bool):
        super().__init__()
        self.n_heads = n_heads
        self.causal = causal
        opts = dict(activate=True, batch_norm=batch_norm, bn_decay=bn_decay)
        self.query = _FC(2 * d_model, d_model, **opts)
        self.key = _FC(2 * d_model, d_model, **opts)
        self.value = _FC(2 * d_model, d_model, **opts)
        self.output = _mlp(d_model, d_model, d_model, batch_norm=batch_norm, bn_decay=bn_decay)

    def forward(self, x: torch.Tensor, ste: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((x, ste), dim=-1)
        return self.output(
            _attend(
                self.query(joined),
                self.key(joined),
                self.value(joined),
                self.n_heads,
                temporal=True,
                causal=self.causal,
            )
        )


class _GatedFusion(nn.Module):
    def __init__(self, d_model: int, *, batch_norm: bool, bn_decay: float):
        super().__init__()
        self.spatial_gate = _linear(d_model, d_model, bias=False)
        self.temporal_gate = _linear(d_model, d_model)
        self.output = _mlp(d_model, d_model, d_model, batch_norm=batch_norm, bn_decay=bn_decay)

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.spatial_gate(spatial) + self.temporal_gate(temporal))
        return self.output(gate * spatial + (1.0 - gate) * temporal)


class _STAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, batch_norm: bool, bn_decay: float, causal: bool):
        super().__init__()
        opts = dict(batch_norm=batch_norm, bn_decay=bn_decay)
        self.spatial = _SpatialAttention(d_model, n_heads, **opts)
        self.temporal = _TemporalAttention(d_model, n_heads, causal=causal, **opts)
        self.fusion = _GatedFusion(d_model, **opts)

    def forward(self, x: torch.Tensor, ste: torch.Tensor) -> torch.Tensor:
        return x + self.fusion(self.spatial(x, ste), self.temporal(x, ste))


class _TransformAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, batch_norm: bool, bn_decay: float):
        super().__init__()
        self.n_heads = n_heads
        opts = dict(activate=True, batch_norm=batch_norm, bn_decay=bn_decay)
        self.query = _FC(d_model, d_model, **opts)
        self.key = _FC(d_model, d_model, **opts)
        self.value = _FC(d_model, d_model, **opts)
        self.output = _mlp(d_model, d_model, d_model, batch_norm=batch_norm, bn_decay=bn_decay)

    def forward(self, x: torch.Tensor, ste_history: torch.Tensor, ste_future: torch.Tensor) -> torch.Tensor:
        return self.output(
            _attend(self.query(ste_future), self.key(ste_history), self.value(x), self.n_heads, temporal=True)
        )


class _STEmbedding(nn.Module):
    def __init__(self, d_model: int, time_slots_per_day: int, *, batch_norm: bool, bn_decay: float):
        super().__init__()
        self.time_slots_per_day = time_slots_per_day
        opts = dict(batch_norm=batch_norm, bn_decay=bn_decay)
        self.spatial = _mlp(d_model, d_model, d_model, **opts)
        self.temporal = _mlp(7 + time_slots_per_day, d_model, d_model, **opts)

    def forward(self, spatial_embedding: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        day = F.one_hot(calendar[..., 0].long(), num_classes=7)
        slot = F.one_hot(calendar[..., 1].long(), num_classes=self.time_slots_per_day)
        temporal = torch.cat((day, slot), dim=-1).to(dtype=spatial_embedding.dtype)
        return self.spatial(spatial_embedding[None, None]) + self.temporal(temporal[:, :, None])
