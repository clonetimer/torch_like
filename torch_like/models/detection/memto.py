"""MEMTO-inspired memory-guided Transformer for multivariate anomaly detection.

This implementation follows the MEMTO paper direction and task formulation,
but is integrated for torch_like's detection contract rather than reproducing
the official training code line by line.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.detection.common import DetectionBatch, DetectionModel


@dataclass
class MEMTOConfig:
    input_dim: int
    seq_len: int
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    mem_size: int = 32
    latent_dim: int = 64
    commitment_weight: float = 0.1


class MemoryBank(nn.Module):
    def __init__(self, mem_size: int, latent_dim: int):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, latent_dim))
        nn.init.xavier_uniform_(self.memory)

    def forward(self, z: torch.Tensor):
        # z: (B, D)
        attn_logits = z @ self.memory.T
        attn = torch.softmax(attn_logits, dim=-1)
        memory_read = attn @ self.memory
        return memory_read, attn


class MEMTO(DetectionModel):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        mem_size: int = 32,
        latent_dim: int = 64,
        commitment_weight: float = 0.1,
    ):
        super().__init__()
        self._config = MEMTOConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            mem_size=mem_size,
            latent_dim=latent_dim,
            commitment_weight=commitment_weight,
        )
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.commitment_weight = commitment_weight

        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers, enable_nested_tensor=False)
        self.latent_proj = nn.Linear(d_model, latent_dim)
        self.memory_bank = MemoryBank(mem_size=mem_size, latent_dim=latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )
        self.sequence_decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, seq_len * input_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.input_proj(x)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        z = self.latent_proj(pooled)
        memory_read, attn = self.memory_bank(z)
        return z, memory_read, attn

    def reconstruct(self, batch: DetectionBatch):
        z, memory_read, attn = self.encode(batch.x)
        recon_last = self.decoder(memory_read)
        recon_seq = self.sequence_decoder(memory_read).view(batch.x.size(0), self.seq_len, self.input_dim)
        return recon_last, recon_seq, z, memory_read, attn

    def forward(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        del x_mark
        batch = self.make_detection_batch(x)
        recon_last, _, _, _, _ = self.reconstruct(batch)
        return recon_last

    def fit_step(self, batch: DetectionBatch):
        recon_last, recon_seq, z, memory_read, _ = self.reconstruct(batch)
        target_last = batch.x[:, -1, :]
        recon_last_loss = F.mse_loss(recon_last, target_last)
        recon_seq_loss = F.mse_loss(recon_seq, batch.x)
        commitment = F.mse_loss(z, memory_read.detach())
        return recon_last_loss + recon_seq_loss + self.commitment_weight * commitment

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        recon_last, recon_seq, z, memory_read, _ = self.reconstruct(batch)
        input_error = (recon_last - batch.x[:, -1, :]) ** 2
        latent_error = (z - memory_read) ** 2
        latent_score = latent_error.mean(dim=-1, keepdim=True).expand_as(input_error)
        score = input_error + latent_score
        if reduction == "none":
            return score
        if reduction == "sum":
            return score.sum(dim=-1)
        return score.mean(dim=-1)
