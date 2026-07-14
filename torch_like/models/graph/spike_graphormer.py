"""SpikeGraphormer-inspired graph transformer for the graph domain."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool
from torch_like.models.graph.gcn import _normalize_adjacency


def _surrogate_spike(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Hard spike in the forward pass with a straight-through gradient."""
    hard = (x > threshold).to(x.dtype)
    return hard.detach() + x - x.detach()


class SpikingGraphAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, spike_threshold: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.spike_threshold = spike_threshold
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        batch, num_nodes, dim = x.shape
        q = self.q_proj(x).view(batch, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)

        q_spike = _surrogate_spike(q, self.spike_threshold)
        k_spike = _surrogate_spike(k, self.spike_threshold)
        scores = torch.matmul(q_spike, k_spike.transpose(-2, -1)) * self.scale

        if adjacency is not None:
            if adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0)
            mask = adjacency.unsqueeze(1) > 0
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, num_nodes, dim)
        return self.out_proj(out)


class SpikeGraphormerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = SpikingGraphAttention(d_model, n_heads, dropout)
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
        x = self.norm_1(x + self.attn(x, adjacency))
        return self.norm_2(x + self.ffn(x))


class LocalGraphBranch(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(adjacency, x)
        return self.dropout(torch.relu(self.linear(x)))


class SpikeGraphormer(GraphModel):
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
        fusion_weight: float = 0.5,
        use_adjacency_bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.fusion_weight = fusion_weight
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
        self.local_branch = LocalGraphBranch(d_model, dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                SpikeGraphormerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
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
            raise ValueError("SpikeGraphormer expects x with shape (batch, num_nodes, input_dim)")
        if x.size(1) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")
        if adjacency is None:
            raise ValueError("SpikeGraphormer requires adjacency in the current framework-oriented implementation")

        adj = adjacency.to(device=x.device, dtype=x.dtype)
        norm_adj = _normalize_adjacency(adj)
        attn_adj = None
        if self.use_adjacency_bias:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)
            identity = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype).unsqueeze(0)
            attn_adj = (adj + identity).clamp(max=1.0)

        out = self.input_proj(x)
        local = self.local_branch(out, norm_adj)
        global_out = out
        for block in self.transformer_blocks:
            global_out = block(global_out, attn_adj)
        out = self.fusion_weight * local + (1.0 - self.fusion_weight) * global_out
        out = self.output_proj(out)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
