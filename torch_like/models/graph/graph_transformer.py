"""Graph Transformer baseline for the general graph domain."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None) -> torch.Tensor:
        key_padding_mask = None
        attn_mask = None
        if adjacency is not None:
            if adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0)
            full_adj = (adjacency > 0).to(dtype=torch.bool)
            # MultiheadAttention expects True entries to be masked out. A
            # three-dimensional mask is indexed by batch * attention head.
            if full_adj.size(0) == 1:
                attn_mask = ~full_adj[0]
            else:
                if full_adj.size(0) != x.size(0):
                    raise ValueError(f"Expected adjacency batch size 1 or {x.size(0)}, got {full_adj.size(0)}")
                attn_mask = (~full_adj).repeat_interleave(self.n_heads, dim=0)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm_1(x + attn_out)
        return self.norm_2(x + self.ffn(x))


class GraphTransformer(GraphModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 128,
        use_adjacency_bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.use_adjacency_bias = use_adjacency_bias
        self._config = GraphConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
        )
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                GraphTransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, output_dim)

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
        if x.dim() != 3:
            raise ValueError("GraphTransformer expects x with shape (batch, num_nodes, input_dim)")
        if x.size(1) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        adj = None
        if self.use_adjacency_bias and adjacency is not None:
            adj = adjacency.to(device=x.device)
            identity = torch.eye(self.num_nodes, device=x.device, dtype=adj.dtype).unsqueeze(0)
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)
            adj = (adj + identity).clamp(max=1.0)

        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out, adj)
        out = self.output_proj(out)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
