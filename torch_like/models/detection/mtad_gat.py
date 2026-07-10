"""MTAD-GAT-style multivariate time-series anomaly detector.

This framework-oriented implementation keeps the core MTAD-GAT structure:
feature-oriented graph attention, temporal graph attention, recurrent fusion,
and joint forecasting/reconstruction objectives for anomaly scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


@dataclass
class MTADGATConfig(DetectionConfig):
    n_heads: int = 4
    e_layers: int = 1
    d_ff: int = 256
    forecast_weight: float = 1.0
    recon_weight: float = 1.0


class FeatureTemporalGATBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.feature_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feature_norm = nn.LayerNorm(d_model)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_tokens: torch.Tensor, temporal_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_out, _ = self.feature_attn(feature_tokens, feature_tokens, feature_tokens, need_weights=False)
        temporal_out, _ = self.temporal_attn(temporal_tokens, temporal_tokens, temporal_tokens, need_weights=False)
        feature_tokens = self.feature_norm(feature_tokens + self.dropout(feature_out))
        temporal_tokens = self.temporal_norm(temporal_tokens + self.dropout(temporal_out))

        feature_context = feature_tokens.mean(dim=1, keepdim=True).expand(-1, temporal_tokens.size(1), -1)
        fused = self.fusion(torch.cat([temporal_tokens, feature_context], dim=-1))
        temporal_tokens = self.fusion_norm(temporal_tokens + self.dropout(fused))
        return feature_tokens, temporal_tokens


class MTADGAT(DetectionModel):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        dropout: float = 0.1,
        threshold: float = 0.5,
        n_heads: int = 4,
        e_layers: int = 1,
        d_ff: int = 256,
        forecast_weight: float = 1.0,
        recon_weight: float = 1.0,
    ):
        super().__init__()
        del threshold
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.forecast_weight = forecast_weight
        self.recon_weight = recon_weight
        self._config = MTADGATConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            forecast_weight=forecast_weight,
            recon_weight=recon_weight,
        )

        self.feature_proj = nn.Linear(seq_len, d_model)
        self.temporal_proj = nn.Linear(input_dim, d_model)
        self.feature_pos = nn.Parameter(torch.zeros(1, input_dim, d_model))
        self.temporal_pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList(
            [FeatureTemporalGATBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(e_layers)]
        )
        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_dim),
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_dim),
        )
        nn.init.trunc_normal_(self.feature_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("MTADGAT expects x with shape (batch, seq_len, input_dim)")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {x.size(-1)}")

        feature_tokens = self.feature_proj(x.transpose(1, 2)) + self.feature_pos
        temporal_tokens = self.temporal_proj(x) + self.temporal_pos
        for block in self.blocks:
            feature_tokens, temporal_tokens = block(feature_tokens, temporal_tokens)
        encoded, _ = self.gru(temporal_tokens)
        return encoded

    def reconstruct(self, batch: DetectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(batch.x)
        forecast_last = self.forecast_head(encoded[:, -1, :])
        recon_seq = self.reconstruction_head(encoded)
        return forecast_last, recon_seq

    def forward(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        del x_mark
        batch = self.make_detection_batch(x)
        forecast_last, _ = self.reconstruct(batch)
        return forecast_last

    def fit_step(self, batch: DetectionBatch):
        forecast_last, recon_seq = self.reconstruct(batch)
        forecast_loss = F.mse_loss(forecast_last, batch.x[:, -1, :])
        recon_loss = F.mse_loss(recon_seq, batch.x)
        return self.forecast_weight * forecast_loss + self.recon_weight * recon_loss

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        forecast_last, recon_seq = self.reconstruct(batch)
        forecast_score = (forecast_last - batch.x[:, -1, :]) ** 2
        recon_score = (recon_seq[:, -1, :] - batch.x[:, -1, :]) ** 2
        score = self.forecast_weight * forecast_score + self.recon_weight * recon_score
        if reduction == "none":
            return score
        if reduction == "sum":
            return score.sum(dim=-1)
        return score.mean(dim=-1)
