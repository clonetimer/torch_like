"""GATv2 graph attention network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


@dataclass
class GATv2Config(GraphConfig):
    num_layers: int = 2
    heads: int = 4


class GATv2Layer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float, concat: bool = True):
        super().__init__()
        if min(in_dim, out_dim, heads) <= 0:
            raise ValueError("in_dim, out_dim, and heads must be positive")
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.dst = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, _ = x.shape
        src = self.src(x).view(batch, num_nodes, self.heads, self.out_dim)
        dst = self.dst(x).view(batch, num_nodes, self.heads, self.out_dim)
        pair = torch.nn.functional.leaky_relu(src.unsqueeze(2) + dst.unsqueeze(1), negative_slope=0.2)
        scores = (pair * self.attn.view(1, 1, 1, self.heads, self.out_dim)).sum(dim=-1)
        scores = scores.masked_fill(adjacency.unsqueeze(-1) <= 0, torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores, dim=2))
        out = torch.einsum("bijh,bjhd->bihd", attention, dst)
        if self.concat:
            return out.reshape(batch, num_nodes, self.heads * self.out_dim)
        return out.mean(dim=2)


class GATv2(GraphModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        if min(input_dim, output_dim, num_nodes, d_model, num_layers, heads) <= 0:
            raise ValueError("input_dim, output_dim, num_nodes, d_model, num_layers, and heads must be positive")
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        if task_level not in {"node", "graph"}:
            raise ValueError("task_level must be 'node' or 'graph'")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self._config = GATv2Config(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            heads=heads,
        )
        layers = []
        current = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(GATv2Layer(current, d_model // heads, heads=heads, dropout=dropout, concat=True))
            current = d_model
        layers.append(GATv2Layer(current, output_dim, heads=1, dropout=0.0, concat=False))
        self.layers = nn.ModuleList(layers)

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
            raise ValueError("GATv2 requires adjacency")
        if x.dim() != 3:
            raise ValueError("GATv2 expects x with shape (batch, num_nodes, input_dim)")
        if x.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError(f"expected x shape (*, {self.num_nodes}, {self.input_dim}), got {tuple(x.shape)}")
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(x.size(0), -1, -1)
        if adjacency.shape != (x.size(0), self.num_nodes, self.num_nodes):
            raise ValueError(f"adjacency must have shape ({x.size(0)}, {self.num_nodes}, {self.num_nodes})")
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
