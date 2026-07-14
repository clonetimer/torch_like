"""GPS-style graph network combining local propagation and global attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


@dataclass
class GPSConfig(GraphConfig):
    num_layers: int = 4
    n_heads: int = 8
    d_ff: int = 256


class GPSLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.local_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.global_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_local = nn.LayerNorm(d_model)
        self.norm_global = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local = self.local_mlp(torch.matmul(adjacency, x) / degree)
        local = self.norm_local(x + local)
        global_out, _ = self.global_attention(
            x,
            x,
            x,
            key_padding_mask=None if node_mask is None else ~node_mask.bool(),
            need_weights=False,
        )
        x = local + self.norm_global(x + global_out)
        return x + self.ffn(self.norm_ffn(x))


class GPS(GraphModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 128,
        dropout: float = 0.1,
        task_level: str = "graph",
        num_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 256,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self._config = GPSConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            n_heads=n_heads,
            d_ff=d_ff,
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.degree_embedding = nn.Embedding(num_nodes + 1, d_model)
        self.layers = nn.ModuleList([GPSLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x, adjacency=None, edge_index=None, edge_attr=None, batch_index=None, node_mask=None):
        del edge_index, edge_attr
        if adjacency is None:
            raise ValueError("GPS requires adjacency")
        if x.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError(f"GPS expects x with shape (batch, {self.num_nodes}, {self.input_dim})")
        adjacency = adjacency.to(device=x.device, dtype=x.dtype)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.size(0) == 1:
            adjacency = adjacency.expand(x.size(0), -1, -1)
        degree = adjacency.gt(0).sum(dim=-1).long().clamp_max(self.num_nodes)
        hidden = self.input_projection(x) + self.degree_embedding(degree)
        for layer in self.layers:
            hidden = layer(hidden, adjacency, node_mask)
        out = self.output_projection(hidden)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
