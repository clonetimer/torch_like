"""GCNII-style deep graph convolution network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool
from torch_like.models.graph.gcn import _normalize_adjacency


@dataclass
class GCNIIConfig(GraphConfig):
    num_layers: int = 8
    alpha: float = 0.1
    theta: float = 0.5


class GCNIILayer(nn.Module):
    def __init__(self, d_model: int, layer_idx: int, alpha: float, theta: float, dropout: float):
        super().__init__()
        self.alpha = alpha
        self.beta = torch.log(torch.tensor(theta / layer_idx + 1.0)).item()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, h0: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.matmul(adjacency, h)
        initial_residual = (1.0 - self.alpha) * propagated + self.alpha * h0
        transformed = self.linear(initial_residual)
        out = (1.0 - self.beta) * initial_residual + self.beta * transformed
        return self.dropout(torch.relu(out))


class GCNII(GraphModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 8,
        alpha: float = 0.1,
        theta: float = 0.5,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self._config = GCNIIConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            alpha=alpha,
            theta=theta,
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([GCNIILayer(d_model, idx + 1, alpha, theta, dropout) for idx in range(num_layers)])
        self.output_projection = nn.Linear(d_model, output_dim)

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
            raise ValueError("GCNII requires adjacency")
        if x.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError(f"GCNII expects x with shape (batch, {self.num_nodes}, {self.input_dim})")
        adj = _normalize_adjacency(adjacency.to(device=x.device, dtype=x.dtype))
        h0 = torch.relu(self.input_projection(x))
        h = h0
        for layer in self.layers:
            h = layer(h, h0, adj)
        out = self.output_projection(h)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
