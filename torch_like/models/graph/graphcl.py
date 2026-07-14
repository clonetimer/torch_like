"""GraphCL-style self-supervised graph contrastive representation model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.graph.common import GraphBatch, GraphConfig, GraphModel, graph_mean_pool
from torch_like.models.graph.gcn import _normalize_adjacency


@dataclass
class GraphCLConfig(GraphConfig):
    num_layers: int = 2
    projection_dim: int = 128
    temperature: float = 0.2
    augmentation_dropout: float = 0.2


class GraphCL(GraphModel):
    """Graph-level contrastive encoder using dense adjacency graph convolutions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "graph",
        num_layers: int = 2,
        projection_dim: int = 128,
        temperature: float = 0.2,
        augmentation_dropout: float = 0.2,
    ):
        super().__init__()
        if task_level not in {"node", "graph"}:
            raise ValueError("task_level must be 'node' or 'graph'")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.temperature = temperature
        self.augmentation_dropout = augmentation_dropout
        self._config = GraphCLConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            projection_dim=projection_dim,
            temperature=temperature,
            augmentation_dropout=augmentation_dropout,
        )

        dims = [input_dim] + [d_model] * num_layers
        self.layers = nn.ModuleList(nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1))
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, projection_dim),
        )

    def _validate_inputs(self, x: torch.Tensor, adjacency: torch.Tensor | None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("GraphCL expects x with shape (batch, num_nodes, input_dim)")
        if x.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError(f"GraphCL expects x with shape (batch, {self.num_nodes}, {self.input_dim})")
        if adjacency is None:
            adjacency = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype)
        return _normalize_adjacency(adjacency.to(device=x.device, dtype=x.dtype))

    def encode_nodes(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        """Encode nodes for one graph view."""
        adj = self._validate_inputs(x, adjacency)
        h = x
        for layer in self.layers:
            h = torch.matmul(adj, h)
            h = torch.relu(layer(h))
            h = self.dropout(h)
        return self.output_projection(h)

    def encode_graph(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Encode one graph view into graph-level embeddings."""
        node_embeddings = self.encode_nodes(x, adjacency)
        graph_embeddings = graph_mean_pool(node_embeddings, node_mask=node_mask, batch_index=batch_index)
        if normalize:
            graph_embeddings = F.normalize(graph_embeddings, dim=-1)
        return graph_embeddings

    def project(self, graph_embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Project graph embeddings into the contrastive space."""
        projected = self.projection_head(graph_embeddings)
        if normalize:
            projected = F.normalize(projected, dim=-1)
        return projected

    def contrastive_logits(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Pairwise graph-view similarity logits."""
        if z1.dim() != 2 or z2.dim() != 2:
            raise ValueError("contrastive logits expect two 2D embedding tensors")
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 must have the same shape, got {tuple(z1.shape)} and {tuple(z2.shape)}")
        return torch.matmul(F.normalize(z1, dim=-1), F.normalize(z2, dim=-1).transpose(0, 1)) / self.temperature

    def contrastive_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        adjacency1: torch.Tensor | None = None,
        adjacency2: torch.Tensor | None = None,
        node_mask1: torch.Tensor | None = None,
        node_mask2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Symmetric InfoNCE loss between two augmented graph views."""
        g1 = self.encode_graph(x1, adjacency=adjacency1, node_mask=node_mask1)
        g2 = self.encode_graph(x2, adjacency=adjacency2, node_mask=node_mask2)
        z1 = self.project(g1)
        z2 = self.project(g2)
        logits = self.contrastive_logits(z1, z2)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def loss_step(self, batch: GraphBatch, positive_batch: GraphBatch | None = None) -> torch.Tensor:
        """Compute GraphCL loss for two graph views.

        If ``positive_batch`` is omitted, a lightweight feature-dropout view is
        generated as a fallback. Production augmentation policies should live
        outside the model class.
        """
        if positive_batch is None:
            x2 = F.dropout(batch.x, p=self.augmentation_dropout, training=True)
            return self.contrastive_loss(
                batch.x,
                x2,
                adjacency1=batch.adjacency,
                adjacency2=batch.adjacency,
                node_mask1=batch.node_mask,
                node_mask2=batch.node_mask,
            )
        return self.contrastive_loss(
            batch.x,
            positive_batch.x,
            adjacency1=batch.adjacency,
            adjacency2=positive_batch.adjacency,
            node_mask1=batch.node_mask,
            node_mask2=positive_batch.node_mask,
        )

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
        node_embeddings = self.encode_nodes(x, adjacency)
        if node_mask is not None:
            node_embeddings = node_embeddings * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(node_embeddings, node_mask=node_mask, batch_index=batch_index)
        return node_embeddings
