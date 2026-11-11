"""HSNN-IB-inspired hierarchical subgraph RUL model.

This is a paper-aligned, portable PyTorch implementation of the main
HMGPA/HGSFU/MFIBM ideas from Zheng et al., Information Fusion 127 (2026)
103953. It is not a claim of exact source-code reproduction or benchmark
reproduction. Sensor normalization and run-to-failure label construction
remain the caller's responsibility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.rul import RULBatch, RULConfig, RULModel


class HSNNIBRUL(RULModel):
    """Hierarchical subgraph neural network with an information bottleneck.

    Input uses the common RUL contract with shape (batch, time, sensors).
    The model builds a per-window sensor graph from cosine similarity,
    dynamically recalibrates its retained edges, partitions the graph into
    non-overlapping representative subgraphs, and performs local plus gated
    cross-subgraph message passing. A variational information bottleneck
    fuses the subgraph summaries before scalar RUL regression.

    Nodes not selected by the configured subgraph-count and node-budget
    settings are intentionally omitted from the subgraph summaries, matching
    the paper's bounded representative-subgraph design. Configure a sufficient
    budget for the sensor count.
    """

    def __init__(
        self,
        enc_in: int = 14,
        seq_len: int = 60,
        hidden_dim: int = 60,
        num_layers: int = 2,
        num_subgraphs: int = 3,
        max_nodes_per_subgraph: int = 5,
        initial_edge_threshold: float = 0.5,
        centrality_alpha: float = 0.5,
        ib_weight: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        integer_values = {
            "enc_in": enc_in,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_subgraphs": num_subgraphs,
            "max_nodes_per_subgraph": max_nodes_per_subgraph,
        }
        if any(type(value) is not int or value <= 0 for value in integer_values.values()):
            raise ValueError("all HSNN-IB dimensions and layer counts must be positive integers")
        if num_subgraphs > enc_in:
            raise ValueError("num_subgraphs cannot exceed enc_in")
        if not math.isfinite(initial_edge_threshold) or not 0.0 <= initial_edge_threshold <= 1.0:
            raise ValueError("initial_edge_threshold must be finite and in [0, 1]")
        if not math.isfinite(centrality_alpha) or not 0.0 <= centrality_alpha <= 1.0:
            raise ValueError("centrality_alpha must be finite and in [0, 1]")
        if not math.isfinite(ib_weight) or ib_weight < 0:
            raise ValueError("ib_weight must be finite and nonnegative")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_subgraphs = num_subgraphs
        self.max_nodes_per_subgraph = max_nodes_per_subgraph
        self.initial_edge_threshold = float(initial_edge_threshold)
        self.centrality_alpha = float(centrality_alpha)
        self.ib_weight = float(ib_weight)
        self.dropout_rate = float(dropout)

        self._config = RULConfig(
            enc_in=enc_in,
            seq_len=seq_len,
            d_model=hidden_dim,
            n_heads=1,
            e_layers=num_layers,
            d_ff=hidden_dim * 2,
            dropout=dropout,
        )

        self.sensor_encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.edge_similarity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.intra_subgraph = nn.Sequential(
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(1, hidden_dim // 2), hidden_dim),
            nn.Tanh(),
        )
        self.inter_similarity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.gate_eta = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_mu = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ib_encoder = nn.Linear(num_subgraphs * hidden_dim, hidden_dim * 2)
        self.ib_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.ReLU(),
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("HSNNIBRUL expects x with shape (batch, seq_len, enc_in)")
        if x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"HSNNIBRUL expects x with shape (batch, {self.seq_len}, {self.enc_in}), "
                f"got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all().item():
            raise ValueError("HSNNIBRUL requires finite inputs; normalize/impute signals upstream")

    def _initial_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        sensor_series = x.transpose(1, 2)
        normalized = F.normalize(sensor_series, p=2, dim=-1, eps=1e-8)
        cosine = torch.bmm(normalized, normalized.transpose(1, 2)).clamp(min=0.0, max=1.0)
        adjacency = (cosine >= self.initial_edge_threshold).to(dtype=x.dtype)
        identity = torch.eye(self.enc_in, device=x.device, dtype=x.dtype).unsqueeze(0)
        return torch.maximum(adjacency, identity)

    def partition_graph(self, edge_weights: torch.Tensor) -> tuple[tuple[int, ...], ...]:
        """Return deterministic HMGPA-style disjoint groups for one sensor graph."""
        if not isinstance(edge_weights, torch.Tensor) or edge_weights.ndim != 2:
            raise ValueError("edge_weights must have shape (sensors, sensors)")
        if edge_weights.shape[0] != edge_weights.shape[1] or edge_weights.shape[0] != self.enc_in:
            raise ValueError(f"edge_weights must have shape ({self.enc_in}, {self.enc_in})")
        if not torch.isfinite(edge_weights).all().item() or torch.any(edge_weights < 0).item():
            raise ValueError("edge_weights must be finite and nonnegative")

        weights = edge_weights.detach()
        row_sum = weights.sum(dim=-1)
        probabilities = weights / row_sum.clamp_min(torch.finfo(weights.dtype).eps).unsqueeze(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        strength = row_sum / self.enc_in
        centrality = self.centrality_alpha * strength - (1.0 - self.centrality_alpha) * entropy
        ranking = torch.argsort(centrality, descending=True).tolist()

        remaining = set(range(self.enc_in))
        groups: list[tuple[int, ...]] = []
        for center in ranking:
            if center not in remaining:
                continue
            candidates = [
                node
                for node in remaining
                if node != center and float(weights[center, node].item()) > 0.0
            ]
            candidates.sort(key=lambda node: float(weights[center, node].item()), reverse=True)
            selected = (center, *candidates[: self.max_nodes_per_subgraph - 1])
            groups.append(selected)
            remaining.difference_update(selected)
            if len(groups) == self.num_subgraphs:
                break

        if len(groups) != self.num_subgraphs:
            raise RuntimeError("HMGPA could not construct the configured number of nonempty subgraphs")
        return tuple(groups)

    def _dynamic_edge_weights(self, hidden: torch.Tensor, initial_adjacency: torch.Tensor) -> torch.Tensor:
        differences = hidden.unsqueeze(2) - hidden.unsqueeze(1)
        similarities = self.edge_similarity(differences).squeeze(-1)
        return similarities * initial_adjacency

    def _update_one_graph(
        self,
        hidden: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = self.partition_graph(edge_weights)
        local = torch.zeros_like(hidden)

        for group in groups:
            indices = torch.as_tensor(group, device=hidden.device, dtype=torch.long)
            group_hidden = hidden.index_select(0, indices)
            group_edges = edge_weights.index_select(0, indices).index_select(1, indices)
            normalized_edges = group_edges / group_edges.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            aggregated = normalized_edges @ group_hidden
            local = local.index_copy(0, indices, self.intra_subgraph(aggregated))

        updated = local.clone()
        for group_index, group in enumerate(groups):
            indices = torch.as_tensor(group, device=hidden.device, dtype=torch.long)
            target_features = local.index_select(0, indices)
            target_updates = []

            for target in target_features:
                cross_messages = []
                for other_index, other_group in enumerate(groups):
                    if other_index == group_index:
                        continue
                    other_indices = torch.as_tensor(other_group, device=hidden.device, dtype=torch.long)
                    other_features = local.index_select(0, other_indices)
                    differences = target.unsqueeze(0) - other_features
                    scores = self.inter_similarity(differences).squeeze(-1)
                    weights = scores / scores.sum().clamp_min(1e-8)
                    context = torch.sum(weights.unsqueeze(-1) * other_features, dim=0)
                    joined = torch.cat((context, target), dim=-1)
                    eta = torch.tanh(self.gate_eta(joined))
                    mu = torch.sigmoid(self.gate_mu(joined))
                    cross_messages.append(eta * mu)

                if cross_messages:
                    cross_update = torch.stack(cross_messages, dim=0).mean(dim=0)
                    target_updates.append(target + cross_update)
                else:
                    target_updates.append(target)

            updated = updated.index_copy(0, indices, torch.stack(target_updates, dim=0))

        summaries = torch.stack(
            [updated.index_select(0, torch.as_tensor(group, device=hidden.device)).mean(dim=0) for group in groups],
            dim=0,
        )
        return updated, summaries

    def _forward_with_kl(self, x: torch.Tensor, *, stochastic: bool) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(x)
        batch_size = x.shape[0]
        initial_adjacency = self._initial_adjacency(x)

        sensor_sequences = x.transpose(1, 2).reshape(batch_size * self.enc_in, self.seq_len, 1)
        encoded, _ = self.sensor_encoder(sensor_sequences)
        hidden = encoded[:, -1, :].reshape(batch_size, self.enc_in, self.hidden_dim)

        summary_accumulator = x.new_zeros((batch_size, self.num_subgraphs, self.hidden_dim))
        for _ in range(self.num_layers):
            edge_weights = self._dynamic_edge_weights(hidden, initial_adjacency)
            next_hidden = []
            batch_summaries = []
            for batch_index in range(batch_size):
                updated, summaries = self._update_one_graph(hidden[batch_index], edge_weights[batch_index])
                next_hidden.append(updated)
                batch_summaries.append(summaries)
            hidden = torch.stack(next_hidden, dim=0)
            summary_accumulator = summary_accumulator + torch.stack(batch_summaries, dim=0)

        subgraph_summaries = summary_accumulator / self.num_layers
        fused = subgraph_summaries.reshape(batch_size, self.num_subgraphs * self.hidden_dim)
        parameters = self.ib_encoder(fused)
        mean, raw_variance = parameters.chunk(2, dim=-1)
        variance = F.softplus(raw_variance) + 1e-6
        if stochastic and self.training:
            latent = mean + torch.randn_like(mean) * torch.sqrt(variance)
        else:
            latent = mean

        kl = 0.5 * (mean.square() + variance - variance.log() - 1.0).sum(dim=-1).mean()
        decoded = self.ib_decoder(latent)
        prediction = self.rul_head(decoded).squeeze(-1)
        return prediction, kl

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        prediction, _ = self._forward_with_kl(x, stochastic=True)
        return prediction

    def training_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        *,
        x_mark: torch.Tensor | None = None,
        loss: str = "mse",
    ) -> torch.Tensor:
        """RUL prediction loss plus the variational information-bottleneck KL term."""
        del x_mark
        if not isinstance(targets, torch.Tensor) or targets.shape != (x.shape[0],):
            raise ValueError("targets must have shape (batch,)")
        if not torch.isfinite(targets).all().item() or torch.any(targets < 0).item():
            raise ValueError("RUL targets must be finite and nonnegative")
        if loss not in {"mse", "mae", "huber"}:
            raise ValueError("loss must be mse, mae or huber")

        prediction, kl = self._forward_with_kl(x, stochastic=True)
        if loss == "mse":
            prediction_loss = F.mse_loss(prediction, targets)
        elif loss == "mae":
            prediction_loss = F.l1_loss(prediction, targets)
        else:
            prediction_loss = F.huber_loss(prediction, targets)
        return prediction_loss + self.ib_weight * kl

    def loss_step(self, batch: RULBatch, loss: str = "mse") -> torch.Tensor:
        if batch.targets is None:
            raise ValueError("RULBatch.targets is required for loss_step")
        return self.training_loss(batch.x, batch.targets, x_mark=batch.x_mark, loss=loss)


__all__ = ["HSNNIBRUL"]
