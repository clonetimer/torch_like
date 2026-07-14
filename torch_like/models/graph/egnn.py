"""E(n)-equivariant graph neural network for dense geometric graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


@dataclass
class EGNNConfig(GraphConfig):
    num_layers: int = 4
    coordinate_dim: int = 3
    edge_dim: int = 0
    coordinate_scale: float = 1.0


@dataclass
class GeometricGraphBatch:
    x: torch.Tensor
    coordinates: torch.Tensor
    adjacency: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    batch_index: Optional[torch.Tensor] = None
    node_mask: Optional[torch.Tensor] = None


class EGNNLayer(nn.Module):
    """Equivariant coordinate update plus invariant node-feature update."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float, coordinate_scale: float):
        super().__init__()
        edge_input_dim = hidden_dim * 2 + 1 + edge_dim
        self.edge_dim = edge_dim
        self.coordinate_scale = coordinate_scale
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1, bias=False)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden, coordinates, adjacency, edge_attr=None, node_mask=None):
        batch_size, num_nodes, _ = hidden.shape
        source = hidden.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        target = hidden.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        difference = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        squared_distance = difference.square().sum(dim=-1, keepdim=True)

        pieces = [source, target, squared_distance]
        if self.edge_dim:
            if edge_attr is None:
                edge_attr = hidden.new_zeros(batch_size, num_nodes, num_nodes, self.edge_dim)
            if edge_attr.shape != (batch_size, num_nodes, num_nodes, self.edge_dim):
                raise ValueError(f"Expected edge_attr shape ({batch_size}, {num_nodes}, {num_nodes}, {self.edge_dim})")
            pieces.append(edge_attr)
        elif edge_attr is not None:
            raise ValueError("edge_attr was provided but edge_dim is zero")

        edge_mask = adjacency.to(dtype=torch.bool)
        edge_mask = edge_mask & ~torch.eye(num_nodes, device=hidden.device, dtype=torch.bool).unsqueeze(0)
        if node_mask is not None:
            valid = node_mask.to(torch.bool)
            edge_mask = edge_mask & valid.unsqueeze(2) & valid.unsqueeze(1)
        mask = edge_mask.unsqueeze(-1).to(hidden.dtype)

        messages = self.edge_mlp(torch.cat(pieces, dim=-1)) * mask
        degree = mask.sum(dim=2).clamp_min(1.0)
        coordinate_weight = self.coordinate_mlp(messages) * mask
        coordinate_delta = (difference * coordinate_weight).sum(dim=2) / degree
        next_coordinates = coordinates + self.coordinate_scale * coordinate_delta

        aggregated = messages.sum(dim=2) / degree
        next_hidden = hidden + self.node_mlp(torch.cat([hidden, aggregated], dim=-1))
        if node_mask is not None:
            valid = node_mask.unsqueeze(-1).to(hidden.dtype)
            next_hidden = next_hidden * valid
            next_coordinates = torch.where(valid.bool(), next_coordinates, coordinates)
        return next_hidden, next_coordinates


class EGNN(GraphModel):
    """Dense EGNN with invariant task outputs and optional updated coordinates."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 4,
        coordinate_dim: int = 3,
        edge_dim: int = 0,
        coordinate_scale: float = 1.0,
    ):
        super().__init__()
        if min(input_dim, output_dim, num_nodes, d_model, num_layers, coordinate_dim) <= 0:
            raise ValueError("EGNN dimensions and num_layers must be positive")
        if edge_dim < 0:
            raise ValueError("edge_dim must be non-negative")
        if task_level not in {"node", "graph"}:
            raise ValueError("task_level must be 'node' or 'graph'")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.config = EGNNConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            coordinate_dim=coordinate_dim,
            edge_dim=edge_dim,
            coordinate_scale=coordinate_scale,
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [EGNNLayer(d_model, edge_dim, dropout, coordinate_scale) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, output_dim)

    def _validate_inputs(self, x, coordinates, adjacency, edge_attr, node_mask):
        if x.dim() != 3 or x.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError(f"Expected x shape (batch, {self.num_nodes}, {self.input_dim})")
        expected_coordinates = (x.size(0), self.num_nodes, self.config.coordinate_dim)
        if coordinates.shape != expected_coordinates:
            raise ValueError(f"Expected coordinates shape {expected_coordinates}, got {tuple(coordinates.shape)}")
        if adjacency is None:
            adjacency = torch.ones(x.size(0), self.num_nodes, self.num_nodes, device=x.device, dtype=x.dtype)
        elif adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(x.size(0), -1, -1)
        if adjacency.shape != (x.size(0), self.num_nodes, self.num_nodes):
            raise ValueError(f"Expected adjacency shape ({x.size(0)}, {self.num_nodes}, {self.num_nodes})")
        if node_mask is not None and node_mask.shape != (x.size(0), self.num_nodes):
            raise ValueError(f"Expected node_mask shape ({x.size(0)}, {self.num_nodes})")
        if edge_attr is not None and edge_attr.dim() == 3:
            edge_attr = edge_attr.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        return adjacency.to(device=x.device), None if edge_attr is None else edge_attr.to(device=x.device), node_mask

    def forward_geometric(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency, edge_attr, node_mask = self._validate_inputs(x, coordinates, adjacency, edge_attr, node_mask)
        hidden = self.input_projection(x)
        coordinates = coordinates.to(device=x.device, dtype=x.dtype)
        for layer in self.layers:
            hidden, coordinates = layer(hidden, coordinates, adjacency, edge_attr, node_mask)
        output = self.output_projection(hidden)
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1).to(output.dtype)
        if self.task_level == "graph":
            output = graph_mean_pool(output, node_mask=node_mask, batch_index=batch_index)
        return output, coordinates

    def forward(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index
        return self.forward_geometric(x, coordinates, adjacency, edge_attr, batch_index, node_mask)[0]

    def predict(self, batch: GeometricGraphBatch) -> torch.Tensor:
        if not isinstance(batch, GeometricGraphBatch):
            raise TypeError("EGNN.predict expects a GeometricGraphBatch")
        return self(
            batch.x,
            batch.coordinates,
            adjacency=batch.adjacency,
            edge_attr=batch.edge_attr,
            batch_index=batch.batch_index,
            node_mask=batch.node_mask,
        )

    def make_geometric_batch(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> GeometricGraphBatch:
        return GeometricGraphBatch(x, coordinates, adjacency, edge_attr, batch_index, node_mask)
