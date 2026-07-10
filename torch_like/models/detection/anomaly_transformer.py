"""Anomaly Transformer-style detector with association discrepancy.

This is a framework-oriented implementation of the key Anomaly Transformer
idea: compare learned series associations from self-attention with a Gaussian
prior association over temporal distances, and combine that discrepancy with
reconstruction error for anomaly scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


@dataclass
class AnomalyTransformerConfig(DetectionConfig):
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    discrepancy_weight: float = 0.1


class AssociationBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.log_sigma = nn.Parameter(torch.zeros(n_heads))
        distance = torch.arange(max_len, dtype=torch.float32)
        distance = (distance[None, :] - distance[:, None]).abs()
        self.register_buffer("distance", distance, persistent=False)

    def prior_association(self, length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        distance = self.distance[:length, :length].to(device=device, dtype=dtype)
        sigma = F.softplus(self.log_sigma).to(device=device, dtype=dtype) + 1e-3
        sigma = sigma.view(self.n_heads, 1, 1)
        prior = torch.exp(-(distance.unsqueeze(0) ** 2) / (2 * sigma.pow(2)))
        return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, series = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))

        prior = self.prior_association(x.size(1), x.dtype, x.device).unsqueeze(0)
        series = series.clamp_min(1e-6)
        prior = prior.clamp_min(1e-6)
        series = series / series.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        discrepancy = (series * (series.log() - prior.log())).sum(dim=-1)
        discrepancy = discrepancy + (prior * (prior.log() - series.log())).sum(dim=-1)
        return x, discrepancy.mean(dim=1)


class AnomalyTransformer(DetectionModel):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        dropout: float = 0.1,
        threshold: float = 0.5,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        discrepancy_weight: float = 0.1,
    ):
        super().__init__()
        del threshold
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.discrepancy_weight = discrepancy_weight
        self._config = AnomalyTransformerConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            discrepancy_weight=discrepancy_weight,
        )

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList(
            [
                AssociationBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, max_len=seq_len)
                for _ in range(e_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_dim),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("AnomalyTransformer expects x with shape (batch, seq_len, input_dim)")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {x.size(-1)}")

        tokens = self.input_proj(x) + self.pos_embed
        discrepancies = []
        for block in self.blocks:
            tokens, discrepancy = block(tokens)
            discrepancies.append(discrepancy)
        association_discrepancy = torch.stack(discrepancies, dim=0).mean(dim=0)
        return tokens, association_discrepancy

    def reconstruct(self, batch: DetectionBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, discrepancy = self.encode(batch.x)
        recon_seq = self.decoder(tokens)
        recon_last = recon_seq[:, -1, :]
        return recon_last, recon_seq, discrepancy

    def forward(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        del x_mark
        batch = self.make_detection_batch(x)
        recon_last, _, _ = self.reconstruct(batch)
        return recon_last

    def fit_step(self, batch: DetectionBatch):
        recon_last, recon_seq, discrepancy = self.reconstruct(batch)
        last_loss = F.mse_loss(recon_last, batch.x[:, -1, :])
        seq_loss = F.mse_loss(recon_seq, batch.x)
        return last_loss + seq_loss + self.discrepancy_weight * discrepancy.mean()

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        recon_last, _, discrepancy = self.reconstruct(batch)
        recon_score = (recon_last - batch.x[:, -1, :]) ** 2
        discrepancy_score = discrepancy[:, -1:].expand_as(recon_score)
        score = recon_score + self.discrepancy_weight * discrepancy_score
        if reduction == "none":
            return score
        if reduction == "sum":
            return score.sum(dim=-1)
        return score.mean(dim=-1)
