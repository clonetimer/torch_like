"""GNAN-style graph neural additive network for the general graph domain."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


def _batched_shortest_path_distance(adjacency: torch.Tensor) -> torch.Tensor:
    if adjacency.dim() == 2:
        adjacency = adjacency.unsqueeze(0)
    batch_size, num_nodes, _ = adjacency.shape
    device = adjacency.device
    reachable = adjacency > 0
    identity = torch.eye(num_nodes, device=device, dtype=torch.bool).unsqueeze(0)
    reachable = reachable | identity

    dist = torch.full((batch_size, num_nodes, num_nodes), float(num_nodes), device=device, dtype=torch.float32)
    dist = torch.where(reachable, torch.zeros_like(dist), dist)
    dist = dist.masked_fill(identity, 0.0)

    frontier = reachable.clone()
    for step in range(1, num_nodes):
        frontier = torch.matmul(frontier.to(torch.float32), reachable.to(torch.float32)) > 0
        newly_reached = frontier & (dist >= float(num_nodes))
        dist = torch.where(newly_reached, torch.full_like(dist, float(step)), dist)
    return dist


class AdditiveDistanceKernel(nn.Module):
    def __init__(self, d_model: int, num_bases: int, dropout: float):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0.0, 1.0, num_bases))
        self.log_widths = nn.Parameter(torch.zeros(num_bases))
        self.weight_proj = nn.Linear(d_model, num_bases)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_repr: torch.Tensor, normalized_distance: torch.Tensor) -> torch.Tensor:
        basis_distance = normalized_distance.unsqueeze(-1) - self.centers.view(1, 1, 1, -1)
        basis = torch.exp(-basis_distance.pow(2) / torch.exp(self.log_widths).view(1, 1, 1, -1).clamp_min(1e-4))
        weights = torch.softmax(self.weight_proj(node_repr), dim=-1).unsqueeze(2)
        return self.dropout((weights * basis).sum(dim=-1))


class GNANLayer(nn.Module):
    def __init__(self, d_model: int, num_bases: int, d_ff: int, dropout: float):
        super().__init__()
        self.kernel = AdditiveDistanceKernel(d_model=d_model, num_bases=num_bases, dropout=dropout)
        self.message_proj = nn.Linear(d_model, d_model)
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, normalized_distance: torch.Tensor, node_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_weight = self.kernel(x, normalized_distance)
        messages = self.message_proj(x)
        aggregated = torch.matmul(kernel_weight, messages)
        if node_mask is not None:
            aggregated = aggregated * node_mask.unsqueeze(-1)
        updated = self.norm(x + self.update(torch.cat([x, aggregated], dim=-1)))
        return updated, kernel_weight


class GNAN(GraphModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 2,
        d_ff: int = 128,
        num_bases: int = 8,
        distance_mode: str = "shortest_path",
    ):
        super().__init__()
        if distance_mode != "shortest_path":
            raise ValueError(f"Unsupported distance_mode: {distance_mode}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.distance_mode = distance_mode
        self._config = GraphConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
        )
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [GNANLayer(d_model=d_model, num_bases=num_bases, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def _normalize_distance(self, adjacency: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        distance = _batched_shortest_path_distance(adjacency)
        return (distance / max(1, self.num_nodes - 1)).to(dtype=dtype)

    def explain(
        self, batch_x: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        _, contributions = self._forward_internal(
            batch_x, adjacency=adjacency, node_mask=node_mask, return_contributions=True
        )
        return contributions

    def _forward_internal(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        return_contributions: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        normalized_distance = self._normalize_distance(adjacency.to(device=x.device), dtype=x.dtype)
        hidden = self.input_proj(x)
        layer_contributions = []
        for layer in self.layers:
            hidden, contribution = layer(hidden, normalized_distance, node_mask=node_mask)
            layer_contributions.append(contribution)
        out = self.output_proj(hidden)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        contributions = None
        if return_contributions:
            contributions = {
                "distance_weights": torch.stack(layer_contributions, dim=1),
                "normalized_distance": normalized_distance,
            }
        return out, contributions

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
            raise ValueError("GNAN requires adjacency")
        if x.dim() != 3:
            raise ValueError("GNAN expects x with shape (batch, num_nodes, input_dim)")
        if x.size(1) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        out, _ = self._forward_internal(x, adjacency=adjacency, node_mask=node_mask, return_contributions=False)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
