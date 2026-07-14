"""GraphMAE-style masked graph autoencoder for self-supervised pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.graph.common import GraphBatch, GraphConfig, GraphModel, graph_mean_pool
from torch_like.models.graph.gcn import _normalize_adjacency


@dataclass
class GraphMAEConfig(GraphConfig):
    num_layers: int = 2
    decoder_layers: int = 1
    mask_ratio: float = 0.3
    loss_type: str = "sce"


class GraphMAE(GraphModel):
    """Masked node-feature reconstruction model over dense graph batches."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        d_model: int = 64,
        dropout: float = 0.1,
        task_level: str = "node",
        num_layers: int = 2,
        decoder_layers: int = 1,
        mask_ratio: float = 0.3,
        loss_type: str = "sce",
    ):
        super().__init__()
        if task_level not in {"node", "graph"}:
            raise ValueError("task_level must be 'node' or 'graph'")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be >= 1")
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")
        if loss_type not in {"sce", "mse"}:
            raise ValueError("loss_type must be 'sce' or 'mse'")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.mask_ratio = mask_ratio
        self.loss_type = loss_type
        self._config = GraphMAEConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            task_level=task_level,
            num_layers=num_layers,
            decoder_layers=decoder_layers,
            mask_ratio=mask_ratio,
            loss_type=loss_type,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        dims = [input_dim] + [d_model] * num_layers
        self.encoder_layers = nn.ModuleList(nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1))
        decoder_dims = [d_model] + [d_model] * max(0, decoder_layers - 1) + [input_dim]
        self.decoder_layers = nn.ModuleList(
            nn.Linear(decoder_dims[idx], decoder_dims[idx + 1]) for idx in range(len(decoder_dims) - 1)
        )
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _validate_inputs(self, x: torch.Tensor, adjacency: torch.Tensor | None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("GraphMAE expects x with shape (batch, num_nodes, input_dim)")
        if x.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError(f"GraphMAE expects x with shape (batch, {self.num_nodes}, {self.input_dim})")
        if adjacency is None:
            adjacency = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype)
        return _normalize_adjacency(adjacency.to(device=x.device, dtype=x.dtype))

    def sample_mask(self, x: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Sample a boolean node mask for reconstruction."""
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")
        random_mask = torch.rand(x.shape[:2], device=x.device) < self.mask_ratio
        if node_mask is not None:
            random_mask = random_mask & node_mask.to(device=x.device, dtype=torch.bool)
        if not torch.any(random_mask):
            random_mask[:, 0] = True
            if node_mask is not None:
                random_mask = random_mask & node_mask.to(device=x.device, dtype=torch.bool)
        return random_mask

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace masked node features with the learned mask token."""
        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask must have shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}")
        mask_token = self.mask_token.to(device=x.device, dtype=x.dtype).expand_as(x)
        return torch.where(mask.unsqueeze(-1), mask_token, x)

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        """Encode visible and masked node features."""
        adj = self._validate_inputs(x, adjacency)
        h = x
        for layer in self.encoder_layers:
            h = torch.matmul(adj, h)
            h = torch.relu(layer(h))
            h = self.dropout(h)
        return h

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent node embeddings back to input feature space."""
        h = latent
        for idx, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if idx < len(self.decoder_layers) - 1:
                h = self.dropout(torch.relu(h))
        return h

    def reconstruct(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstructed features and the mask used for reconstruction."""
        if mask is None:
            mask = self.sample_mask(x, node_mask=node_mask)
        masked_x = self.apply_mask(x, mask)
        latent = self.encode(masked_x, adjacency)
        return self.decode(latent), mask

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked node-feature reconstruction loss."""
        reconstructed, used_mask = self.reconstruct(x, adjacency=adjacency, mask=mask, node_mask=node_mask)
        if node_mask is not None:
            used_mask = used_mask & node_mask.to(device=x.device, dtype=torch.bool)
        if not torch.any(used_mask):
            raise ValueError("reconstruction mask does not select any valid nodes")
        pred = reconstructed[used_mask]
        target = x[used_mask]
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return (1.0 - (pred * target).sum(dim=-1)).pow(2).mean()

    def loss_step(self, batch: GraphBatch, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.reconstruction_loss(batch.x, adjacency=batch.adjacency, mask=mask, node_mask=batch.node_mask)

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
        node_embeddings = self.output_projection(self.encode(x, adjacency))
        if node_mask is not None:
            node_embeddings = node_embeddings * node_mask.unsqueeze(-1)
        if self.task_level == "graph":
            return graph_mean_pool(node_embeddings, node_mask=node_mask, batch_index=batch_index)
        return node_embeddings
