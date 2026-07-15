"""TranAD-style self-conditioned Transformer anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


@dataclass
class TranADConfig(DetectionConfig):
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256


class TranAD(DetectionModel):
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
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self._config = TranADConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            threshold=threshold,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
        )
        self.input_projection = nn.Linear(2 * input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, e_layers, enable_nested_tensor=False)
        self.reconstruction_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, input_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _phase(self, x: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(torch.cat([x, focus], dim=-1)) + self.position_embedding
        return self.reconstruction_head(self.encoder(hidden))

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(f"TranAD expects x with shape (batch, {self.seq_len}, {self.input_dim})")
        phase_1 = self._phase(x, torch.zeros_like(x))
        focus = (phase_1 - x).pow(2).detach()
        phase_2 = self._phase(x, focus)
        return phase_1, phase_2

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        _, phase_2 = self.reconstruct(x)
        return phase_2[:, -1]

    def fit_step(self, batch: DetectionBatch) -> torch.Tensor:
        phase_1, phase_2 = self.reconstruct(batch.x)
        return 0.5 * ((phase_1 - batch.x).pow(2).mean() + (phase_2 - batch.x).pow(2).mean())

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        phase_1, phase_2 = self.reconstruct(batch.x)
        score = 0.5 * ((phase_1[:, -1] - batch.x[:, -1]).pow(2) + (phase_2[:, -1] - batch.x[:, -1]).pow(2))
        if reduction == "none":
            return score
        if reduction == "sum":
            return score.sum(dim=-1)
        return score.mean(dim=-1)
