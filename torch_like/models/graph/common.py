"""Common graph domain configs and model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torch_like.models.base import BaseModel


@dataclass
class GraphConfig:
    input_dim: int
    output_dim: int
    num_nodes: int
    d_model: int = 256
    dropout: float = 0.1
    task_level: str = "node"


@dataclass
class GraphBatch:
    x: torch.Tensor
    adjacency: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    batch_index: Optional[torch.Tensor] = None
    node_mask: Optional[torch.Tensor] = None


class GraphModel(BaseModel):
    """Base class for general graph models."""

    def predict(self, batch: GraphBatch) -> torch.Tensor:
        return self(
            batch.x,
            adjacency=batch.adjacency,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch_index=batch.batch_index,
            node_mask=batch.node_mask,
        )

    def make_graph_batch(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> GraphBatch:
        return GraphBatch(
            x=x,
            adjacency=adjacency,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch_index=batch_index,
            node_mask=node_mask,
        )


def graph_mean_pool(
    out: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    batch_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean-pool dense graph outputs while excluding padded nodes."""
    batch_size, num_nodes, output_dim = out.shape
    weights = out.new_ones((batch_size, num_nodes))
    if node_mask is not None:
        if node_mask.shape != (batch_size, num_nodes):
            raise ValueError(f"node_mask must have shape ({batch_size}, {num_nodes}), got {tuple(node_mask.shape)}")
        weights = node_mask.to(device=out.device, dtype=out.dtype)

    if batch_index is None:
        counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return out.sum(dim=1) / counts

    if batch_index.numel() == batch_size:
        indices = batch_index.reshape(batch_size, 1).expand(batch_size, num_nodes).reshape(-1)
    elif batch_index.numel() == batch_size * num_nodes:
        indices = batch_index.reshape(-1)
    else:
        raise ValueError(
            f"batch_index must contain {batch_size} graph ids or {batch_size * num_nodes} node ids, "
            f"got {batch_index.numel()}"
        )
    if indices.is_floating_point() or indices.is_complex():
        raise TypeError("batch_index must use an integer dtype")
    indices = indices.to(device=out.device, dtype=torch.long)
    if torch.any(indices < 0):
        raise ValueError("batch_index must not contain negative graph ids")

    pooled_batch_size = int(indices.max().item()) + 1
    flat_weights = weights.reshape(-1)
    pooled = out.new_zeros((pooled_batch_size, output_dim))
    pooled.index_add_(0, indices, out.reshape(-1, output_dim))
    counts = out.new_zeros(pooled_batch_size)
    counts.index_add_(0, indices, flat_weights)
    return pooled / counts.clamp_min(1.0).unsqueeze(-1)
