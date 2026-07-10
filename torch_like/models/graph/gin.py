"""GIN baseline for the general graph domain."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


class GINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        agg = torch.matmul(adjacency, x)
        return self.mlp((1.0 + self.eps) * x + agg)


class GIN(GraphModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self._config = GraphConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
        )
        dims = [input_dim] + [d_model] * max(1, num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [GINLayer(dims[i], dims[i + 1], dropout if i < len(dims) - 2 else 0.0) for i in range(len(dims) - 1)]
        )

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
            raise ValueError("GIN requires adjacency")
        if x.dim() != 3:
            raise ValueError("GIN expects x with shape (batch, num_nodes, input_dim)")
        if x.size(1) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        adjacency = adjacency.to(device=x.device, dtype=x.dtype)
        identity = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype).unsqueeze(0)
        adjacency = (adjacency + identity).clamp(max=1.0)
        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out, adjacency)
            if idx < len(self.layers) - 1:
                out = torch.relu(out)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
