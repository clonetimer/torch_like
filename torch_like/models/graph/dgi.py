"""Deep Graph Infomax-style self-supervised graph representation model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool
from torch_like.models.graph.gcn import _normalize_adjacency


@dataclass
class DGIConfig(GraphConfig):
    num_layers: int = 2
    readout: str = "mean"


class DGI(GraphModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 2,
        readout: str = "mean",
    ):
        super().__init__()
        if readout != "mean":
            raise ValueError("DGI currently supports readout='mean'")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self._config = DGIConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            readout=readout,
        )
        dims = [input_dim] + [d_model] * max(1, num_layers)
        self.layers = nn.ModuleList(nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1))
        self.discriminator = nn.Bilinear(d_model, d_model, 1)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError(f"DGI expects x with shape (batch, {self.num_nodes}, {self.input_dim})")
        adj = _normalize_adjacency(adjacency.to(device=x.device, dtype=x.dtype))
        h = x
        for layer in self.layers:
            h = torch.relu(layer(torch.matmul(adj, h)))
            h = self.dropout(h)
        return h

    def summary(self, node_embeddings: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        if node_mask is not None:
            weights = node_mask.to(device=node_embeddings.device, dtype=node_embeddings.dtype).unsqueeze(-1)
            pooled = (node_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = node_embeddings.mean(dim=1)
        return torch.sigmoid(pooled)

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        permutation = torch.randperm(x.size(1), device=x.device)
        return x[:, permutation, :]

    def dgi_logits(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        negative_x: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive = self.encode(x, adjacency)
        if negative_x is None:
            negative_x = self.corrupt(x)
        negative = self.encode(negative_x, adjacency)
        summary = self.summary(positive, node_mask=node_mask)
        summary_expanded = summary.unsqueeze(1).expand_as(positive)
        pos_logits = self.discriminator(positive, summary_expanded).squeeze(-1)
        neg_logits = self.discriminator(negative, summary_expanded).squeeze(-1)
        return pos_logits, neg_logits

    def dgi_loss(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        negative_x: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pos_logits, neg_logits = self.dgi_logits(x, adjacency, negative_x=negative_x, node_mask=node_mask)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="none")
        if node_mask is not None:
            mask = node_mask.to(device=x.device, dtype=pos_loss.dtype)
            return ((pos_loss + neg_loss) * mask).sum() / mask.sum().clamp_min(1.0)
        return 0.5 * (pos_loss.mean() + neg_loss.mean())

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_attr
        if adjacency is None:
            raise ValueError("DGI requires adjacency")
        out = self.output_projection(self.encode(x, adjacency))
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
