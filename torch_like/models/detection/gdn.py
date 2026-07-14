"""GDN: graph deviation network style multivariate anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


@dataclass
class GDNConfig(DetectionConfig):
    n_heads: int = 4
    top_k: int | None = None
    node_dim: int = 16
    graph_mode: str = "learned"
    prior_weight: float = 0.5


class SensorGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, node_dim: int, top_k: int | None):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.top_k = top_k if top_k is not None else num_nodes
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self) -> torch.Tensor:
        emb = F.normalize(self.embeddings, dim=-1)
        scores = F.relu(emb @ emb.T)
        if self.top_k < scores.size(0):
            mask = torch.zeros_like(scores)
            _, indices = scores.topk(self.top_k, dim=-1)
            mask.scatter_(1, indices, 1.0)
            scores = scores * mask
        return torch.softmax(scores, dim=-1)


class GraphAttentionBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        logits = torch.einsum("bnd,bmd->bnm", q, k) / (h.size(-1) ** 0.5)
        logits = logits + torch.log(adj.clamp_min(1e-6)).unsqueeze(0)
        attn = torch.softmax(logits, dim=-1)
        msg = torch.einsum("bnm,bmd->bnd", attn, v)
        return self.norm(h + self.out(msg))


class GDN(DetectionModel):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        dropout: float = 0.1,
        threshold: float = 0.5,
        n_heads: int = 4,
        top_k: int | None = None,
        node_dim: int = 16,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        del n_heads, threshold
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.graph_mode = graph_mode
        self.prior_weight = prior_weight
        self._config = GDNConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            top_k=top_k,
            node_dim=node_dim,
            graph_mode=graph_mode,
            prior_weight=prior_weight,
        )

        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (input_dim, input_dim):
                raise ValueError(f"adjacency must have shape ({input_dim}, {input_dim}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)

        self.temporal_encoder = nn.GRU(input_size=1, hidden_size=d_model, batch_first=True)
        self.graph = SensorGraphLearner(input_dim, node_dim, top_k)
        self.attention = GraphAttentionBlock(d_model, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _normalize(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj.clamp_min(0)
        return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _resolve_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        learned_adj = self.graph().to(device=device, dtype=dtype)
        prior_adj = (
            None
            if self.prior_adjacency is None
            else self._normalize(self.prior_adjacency.to(device=device, dtype=dtype))
        )
        if self.graph_mode == "learned":
            return learned_adj
        if self.graph_mode == "prior":
            if prior_adj is None:
                raise ValueError("graph_mode='prior' requires adjacency")
            return prior_adj
        if self.graph_mode == "hybrid":
            if prior_adj is None:
                raise ValueError("graph_mode='hybrid' requires adjacency")
            return self._normalize(self.prior_weight * prior_adj + (1.0 - self.prior_weight) * learned_adj)
        raise ValueError(f"Unsupported graph_mode: {self.graph_mode}")

    def learned_adjacency(self) -> torch.Tensor:
        return self.graph()

    def forward(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        del x_mark
        if x.dim() != 3:
            raise ValueError("GDN expects x with shape (batch, seq_len, input_dim)")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {x.size(-1)}")
        sensor_series = x.transpose(1, 2).reshape(x.size(0) * self.input_dim, x.size(1), 1)
        _, h = self.temporal_encoder(sensor_series)
        h = h[-1].view(x.size(0), self.input_dim, -1)
        adj = self._resolve_adjacency(x.device, x.dtype)
        h = self.attention(h, adj)
        return self.decoder(h).squeeze(-1)

    def fit_step(self, batch: DetectionBatch):
        pred = self.predict(batch)
        target = batch.x[:, -1, :]
        return F.mse_loss(pred, target)

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        score = (self.predict(batch) - batch.x[:, -1, :]) ** 2
        if reduction == "none":
            return score
        if reduction == "sum":
            return score.sum(dim=-1)
        return score.mean(dim=-1)
