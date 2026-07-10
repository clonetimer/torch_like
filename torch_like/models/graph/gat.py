"""GAT baseline for the general graph domain."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float, concat: bool = True):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, _ = x.shape
        h = self.linear(x).view(batch, num_nodes, self.heads, self.out_dim)
        src_score = (h * self.attn_src.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)
        dst_score = (h * self.attn_dst.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)
        scores = torch.relu(src_score.unsqueeze(2) + dst_score.unsqueeze(1))
        mask = adjacency.unsqueeze(-1) > 0
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=2)
        attn = self.dropout(attn)
        out = torch.einsum("bijh,bjhd->bihd", attn, h)
        if self.concat:
            return out.reshape(batch, num_nodes, self.heads * self.out_dim)
        return out.mean(dim=2)


class GAT(GraphModel):
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
        layers = []
        in_dim = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(GATLayer(in_dim, d_model // heads, heads=heads, dropout=dropout, concat=True))
            in_dim = d_model
        layers.append(GATLayer(in_dim, output_dim, heads=1, dropout=0.0, concat=False))
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
            raise ValueError("GAT requires adjacency")
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("GAT expects x with shape (batch, num_nodes, input_dim)")
        if x.size(1) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        out = x
        adjacency = adjacency.to(device=x.device, dtype=x.dtype)
        identity = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype).unsqueeze(0)
        adjacency = (adjacency + identity).clamp(max=1.0)
        for idx, layer in enumerate(self.layers):
            out = layer(out, adjacency)
            if idx < len(self.layers) - 1:
                out = torch.relu(out)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
