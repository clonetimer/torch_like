"""PNA graph neural network.

Principal Neighbourhood Aggregation combines multiple neighborhood
aggregators and degree-aware scalers to improve message-passing expressivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.graph.common import GraphConfig, GraphModel, graph_mean_pool


@dataclass
class PNAConfig(GraphConfig):
    num_layers: int = 3
    aggregators: tuple[str, ...] = ("mean", "max", "min", "std")
    scalers: tuple[str, ...] = ("identity", "amplification", "attenuation")
    avg_degree: float = 1.0


def _as_batched_adjacency(adjacency: torch.Tensor, batch_size: int, num_nodes: int, x: torch.Tensor) -> torch.Tensor:
    if adjacency.dim() == 2:
        adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
    if adjacency.shape != (batch_size, num_nodes, num_nodes):
        raise ValueError(f"adjacency must have shape ({batch_size}, {num_nodes}, {num_nodes})")
    return adjacency.to(device=x.device, dtype=x.dtype)


class PNALayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregators: tuple[str, ...],
        scalers: tuple[str, ...],
        avg_degree: float,
        dropout: float,
    ):
        super().__init__()
        supported_aggregators = {"mean", "max", "min", "std"}
        supported_scalers = {"identity", "amplification", "attenuation"}
        if any(name not in supported_aggregators for name in aggregators):
            raise ValueError(f"aggregators must be a subset of {sorted(supported_aggregators)}")
        if any(name not in supported_scalers for name in scalers):
            raise ValueError(f"scalers must be a subset of {sorted(supported_scalers)}")
        self.aggregators = aggregators
        self.scalers = scalers
        self.avg_degree = max(float(avg_degree), 1.0)
        self.update = nn.Sequential(
            nn.Linear(in_dim + in_dim * len(aggregators) * len(scalers), out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def _aggregate(self, x: torch.Tensor, adjacency: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        mask = adjacency > 0
        weighted = adjacency.unsqueeze(-1) * x.unsqueeze(1)
        outputs = []
        if "mean" in self.aggregators:
            outputs.append(weighted.sum(dim=2) / degree.clamp_min(1.0).unsqueeze(-1))
        if "max" in self.aggregators:
            outputs.append(x.unsqueeze(1).masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=2).values.nan_to_num())
        if "min" in self.aggregators:
            outputs.append(x.unsqueeze(1).masked_fill(~mask.unsqueeze(-1), torch.inf).min(dim=2).values.nan_to_num())
        if "std" in self.aggregators:
            mean = weighted.sum(dim=2) / degree.clamp_min(1.0).unsqueeze(-1)
            var = (adjacency.unsqueeze(-1) * (x.unsqueeze(1) - mean.unsqueeze(2)).pow(2)).sum(dim=2)
            outputs.append(torch.sqrt(var / degree.clamp_min(1.0).unsqueeze(-1) + 1e-8))
        return torch.cat(outputs, dim=-1)

    def _scale(self, aggregated: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        log_degree = torch.log(degree.clamp_min(1.0) + 1.0)
        avg = torch.log(torch.tensor(self.avg_degree + 1.0, device=aggregated.device, dtype=aggregated.dtype))
        scaled = []
        if "identity" in self.scalers:
            scaled.append(aggregated)
        if "amplification" in self.scalers:
            scaled.append(aggregated * (log_degree / avg).unsqueeze(-1))
        if "attenuation" in self.scalers:
            scaled.append(aggregated * (avg / log_degree.clamp_min(1.0)).unsqueeze(-1))
        return torch.cat(scaled, dim=-1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1)
        aggregated = self._aggregate(x, adjacency, degree)
        scaled = self._scale(aggregated, degree)
        return self.update(torch.cat([x, scaled], dim=-1))


class PNA(GraphModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 3,
        aggregators: tuple[str, ...] = ("mean", "max", "min", "std"),
        scalers: tuple[str, ...] = ("identity", "amplification", "attenuation"),
        avg_degree: float = 1.0,
    ):
        super().__init__()
        if min(input_dim, output_dim, num_nodes, d_model, num_layers) <= 0:
            raise ValueError("input_dim, output_dim, num_nodes, d_model, and num_layers must be positive")
        if task_level not in {"node", "graph"}:
            raise ValueError("task_level must be 'node' or 'graph'")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self._config = PNAConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            aggregators=aggregators,
            scalers=scalers,
            avg_degree=avg_degree,
        )
        dims = [input_dim] + [d_model] * num_layers
        self.layers = nn.ModuleList(
            [PNALayer(dims[i], dims[i + 1], aggregators, scalers, avg_degree, dropout) for i in range(num_layers)]
        )
        self.output = nn.Linear(d_model, output_dim)

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
            raise ValueError("PNA requires adjacency")
        if x.dim() != 3:
            raise ValueError("PNA expects x with shape (batch, num_nodes, input_dim)")
        if x.size(1) != self.num_nodes or x.size(-1) != self.input_dim:
            raise ValueError(f"expected x shape (*, {self.num_nodes}, {self.input_dim}), got {tuple(x.shape)}")
        adjacency = _as_batched_adjacency(adjacency, x.size(0), self.num_nodes, x)
        out = x
        for layer in self.layers:
            out = layer(out, adjacency)
        out = self.output(out)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(out, node_mask=node_mask, batch_index=batch_index)
        return out
