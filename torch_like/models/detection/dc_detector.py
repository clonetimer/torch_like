"""DCdetector-style dual-attention anomaly detector.

This implementation keeps DCdetector's dual representation idea in a
framework-oriented form: time attention and channel attention produce two
views of the same window, and anomaly scores combine reconstruction error with
their representation discrepancy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


@dataclass
class DCDetectorConfig(DetectionConfig):
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    discrepancy_weight: float = 0.1


class DualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.channel_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, time_tokens: torch.Tensor, channel_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_out, _ = self.time_attn(time_tokens, time_tokens, time_tokens, need_weights=False)
        channel_out, _ = self.channel_attn(channel_tokens, channel_tokens, channel_tokens, need_weights=False)
        time_tokens = self.time_norm(time_tokens + self.dropout(time_out))
        channel_tokens = self.channel_norm(channel_tokens + self.dropout(channel_out))

        channel_context = channel_tokens.mean(dim=1, keepdim=True).expand(-1, time_tokens.size(1), -1)
        time_tokens = self.fusion_norm(time_tokens + channel_context)
        time_tokens = time_tokens + self.ffn(time_tokens)
        return time_tokens, channel_tokens


class DCDetector(DetectionModel):
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
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.discrepancy_weight = discrepancy_weight
        self._config = DCDetectorConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            discrepancy_weight=discrepancy_weight,
        )

        self.time_proj = nn.Linear(input_dim, d_model)
        self.channel_proj = nn.Linear(seq_len, d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.channel_pos = nn.Parameter(torch.zeros(1, input_dim, d_model))
        self.blocks = nn.ModuleList(
            [DualAttentionBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(e_layers)]
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_dim),
        )
        self.channel_to_time = nn.Linear(input_dim, seq_len)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.channel_pos, std=0.02)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("DCDetector expects x with shape (batch, seq_len, input_dim)")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {x.size(-1)}")

        time_tokens = self.time_proj(x) + self.time_pos
        channel_tokens = self.channel_proj(x.transpose(1, 2)) + self.channel_pos
        for block in self.blocks:
            time_tokens, channel_tokens = block(time_tokens, channel_tokens)
        channel_as_time = self.channel_to_time(channel_tokens.transpose(1, 2)).transpose(1, 2)
        discrepancy = (F.normalize(time_tokens, dim=-1) - F.normalize(channel_as_time, dim=-1)).pow(2).mean(dim=-1)
        return time_tokens, channel_tokens, discrepancy

    def reconstruct(self, batch: DetectionBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_tokens, _, discrepancy = self.encode(batch.x)
        recon_seq = self.decoder(time_tokens)
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
        # Normal windows should be reconstructable and keep both views aligned.
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
