"""Model-private, dense de-stationary temporal attention for forecasting.

This is a portable expression of the tau/delta attention operator from
Liu et al., NeurIPS 2022 (https://arxiv.org/abs/2205.14415).
It is deliberately separate from the shared attention classes: those classes
do not carry the de-stationary factors and have a different mask contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _DeStationaryAttention(nn.Module):
    """Attend along temporal keys with sample scale and encoder-key bias.

    Q/K/V have shape ``(B, heads, query, dim)``. ``tau`` is ``(B,)`` and
    optional ``delta`` is ``(B, key)``. An external mask is a bool keep-mask:
    ``True`` allows attention. The causal mask, when requested, is always
    intersected with the external mask. Fully blocked rows are invalid.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        if not isinstance(dropout, (float, int)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a number in [0, 1)")
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("attention q, k and v must have shape (batch, heads, time, dim)")
        batch, heads, queries, head_dim = q.shape
        keys = k.shape[2]
        if (
            batch == 0
            or queries == 0
            or keys == 0
            or head_dim == 0
            or k.shape[:2] != (batch, heads)
            or v.shape[:2] != (batch, heads)
            or k.shape[3] != head_dim
            or v.shape[2] != keys
            or v.shape[3] == 0
        ):
            raise ValueError("attention q, k and v shapes are incompatible")
        if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("attention q, k and v must have matching device and dtype")
        if not isinstance(tau, torch.Tensor) or tau.shape != (batch,):
            raise ValueError("tau must have shape (batch,)")
        if tau.device != q.device or tau.dtype != q.dtype or not torch.isfinite(tau).all() or not (tau > 0).all():
            raise ValueError("tau must be finite, positive, and match attention device/dtype")
        if delta is not None:
            if not isinstance(delta, torch.Tensor) or delta.shape != (batch, keys):
                raise ValueError("delta must have shape (batch, encoder_key_length)")
            if delta.device != q.device or delta.dtype != q.dtype or not torch.isfinite(delta).all():
                raise ValueError("delta must be finite and match attention device/dtype")

        if causal and queries != keys:
            raise ValueError("causal attention requires equal query and key length")
        allowed = None
        if keep_mask is not None:
            if not isinstance(keep_mask, torch.Tensor) or keep_mask.dtype != torch.bool:
                raise TypeError("attention mask must be a boolean True=allowed tensor")
            if keep_mask.device != q.device:
                raise ValueError("attention mask must be on the attention device")
            if keep_mask.ndim == 2 and tuple(keep_mask.shape) == (queries, keys):
                allowed = keep_mask[None, None]
            elif keep_mask.ndim == 3 and tuple(keep_mask.shape) == (batch, queries, keys):
                allowed = keep_mask[:, None]
            elif (
                keep_mask.ndim == 4
                and keep_mask.shape[0] == batch
                and keep_mask.shape[1] in (1, heads)
                and tuple(keep_mask.shape[2:]) == (queries, keys)
            ):
                allowed = keep_mask
            else:
                raise ValueError("attention mask must have (Q,K), (B,Q,K), or (B,1/H,Q,K) shape")
        if causal:
            triangle = torch.ones(queries, keys, dtype=torch.bool, device=q.device).tril()
            allowed = triangle[None, None] if allowed is None else allowed & triangle
        if allowed is not None:
            allowed = torch.broadcast_to(allowed, (batch, heads, queries, keys))
            if not allowed.any(dim=-1).all():
                raise ValueError("attention mask must leave at least one allowed key in every query row")

        # Both learned factors appear inside the 1/sqrt(d) scaling, as in
        # the source equation. The delta vector is keyed by K, not by Q.
        scores = torch.matmul(q, k.transpose(-2, -1)) * tau[:, None, None, None]
        if delta is not None:
            scores = scores + delta[:, None, None, :]
        scores = scores / math.sqrt(head_dim)
        if allowed is not None:
            scores = scores.masked_fill(~allowed, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(self.dropout(probabilities), v)


class _DeStationaryMultiHeadAttention(nn.Module):
    """Learn Q/K/V/out projections around the private de-stationary kernel."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if (
            isinstance(d_model, bool)
            or isinstance(n_heads, bool)
            or not isinstance(d_model, int)
            or not isinstance(n_heads, int)
            or d_model <= 0
            or n_heads <= 0
            or d_model % n_heads
        ):
            raise ValueError("d_model and n_heads must be positive integers with d_model divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = _DeStationaryAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("multihead query, key and value must have (batch,time,d_model) shape")
        if query.shape[2] != self.d_model or key.shape[2] != self.d_model or value.shape[2] != self.d_model:
            raise ValueError("multihead inputs must have configured d_model channels")

        def project(values: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
            batch, length, _ = values.shape
            return linear(values).reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        heads = self.attention(
            project(query, self.q_proj),
            project(key, self.k_proj),
            project(value, self.v_proj),
            tau=tau,
            delta=delta,
            keep_mask=keep_mask,
            causal=causal,
        )
        concatenated = heads.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[1], self.d_model)
        return self.out_proj(concatenated)
