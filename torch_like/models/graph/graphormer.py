"""Graphormer with centrality and shortest-path spatial encodings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


@dataclass
class GraphormerConfig(GraphConfig):
    num_layers: int = 4
    n_heads: int = 8
    d_ff: int = 256
    max_degree: int = 64
    max_distance: int = 32


def shortest_path_distance(adjacency: torch.Tensor, max_distance: int) -> torch.Tensor:
    if adjacency.dim() == 2:
        adjacency = adjacency.unsqueeze(0)
    reachable = adjacency > 0
    batch_size, num_nodes, _ = reachable.shape
    identity = torch.eye(num_nodes, device=adjacency.device, dtype=torch.bool).expand(batch_size, -1, -1)
    distance = torch.full((batch_size, num_nodes, num_nodes), max_distance, device=adjacency.device, dtype=torch.long)
    distance[identity] = 0
    frontier = identity
    for step in range(1, max_distance + 1):
        frontier = torch.bmm(frontier.to(adjacency.dtype), reachable.to(adjacency.dtype)) > 0
        unseen = (distance == max_distance) & frontier
        distance[unseen] = step
    return distance


class GraphormerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.norm_1(x)
        mask = attention_bias.reshape(-1, attention_bias.size(-2), attention_bias.size(-1))
        key_padding_mask = None if node_mask is None else ~node_mask.bool()
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.ffn(self.norm_2(x))


class Graphormer(GraphModel):
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
        max_degree: int = 64,
        max_distance: int = 32,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if task_level not in {"node", "graph"}:
            raise ValueError("task_level must be 'node' or 'graph'")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.max_degree = max_degree
        self.max_distance = max_distance
        self._config = GraphormerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_degree=max_degree,
            max_distance=max_distance,
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.in_degree_embedding = nn.Embedding(max_degree + 1, d_model)
        self.out_degree_embedding = nn.Embedding(max_degree + 1, d_model)
        self.spatial_embedding = nn.Embedding(max_distance + 1, n_heads)
        self.blocks = nn.ModuleList([GraphormerBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x, adjacency=None, edge_index=None, edge_attr=None, batch_index=None, node_mask=None):
        del edge_index, edge_attr
        if adjacency is None:
            raise ValueError("Graphormer requires adjacency")
        if x.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError(f"Graphormer expects x with shape (batch, {self.num_nodes}, {self.input_dim})")
        adjacency = adjacency.to(device=x.device, dtype=x.dtype)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.size(0) == 1:
            adjacency = adjacency.expand(x.size(0), -1, -1)
        if adjacency.shape != (x.size(0), self.num_nodes, self.num_nodes):
            raise ValueError("adjacency must be shared (nodes, nodes) or batched (batch, nodes, nodes)")
        in_degree = adjacency.gt(0).sum(dim=1).long().clamp_max(self.max_degree)
        out_degree = adjacency.gt(0).sum(dim=2).long().clamp_max(self.max_degree)
        hidden = self.input_projection(x) + self.in_degree_embedding(in_degree) + self.out_degree_embedding(out_degree)
        distances = shortest_path_distance(adjacency, self.max_distance)
        attention_bias = self.spatial_embedding(distances).permute(0, 3, 1, 2)
        for block in self.blocks:
            hidden = block(hidden, attention_bias, node_mask)
        out = self.output_projection(self.norm(hidden))
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
