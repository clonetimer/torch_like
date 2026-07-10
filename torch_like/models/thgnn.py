"""THGNN-inspired graph-temporal model for RUL regression.

This implementation follows the graph-temporal direction of the 2024 THGNN
paper, but is not a line-by-line reproduction of an official codebase.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.rul import RULConfig, RULModel


class SensorGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.src = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.dst = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def forward(self) -> torch.Tensor:
        logits = torch.relu(self.src @ self.dst.T)
        return torch.softmax(logits, dim=-1)


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class GraphPropagation(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.proj = nn.Linear(hidden_dim * (depth + 1), hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        states = [x]
        h = x
        for _ in range(self.depth):
            h = torch.einsum("nm,btmd->btnd", adj, h)
            states.append(h)
        h = torch.cat(states, dim=-1)
        h = self.proj(h)
        return F.dropout(h, self.dropout, training=self.training)


class FiLMFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        gamma = torch.sigmoid(self.gamma(global_feat))
        beta = self.beta(global_feat)
        return gamma * local_feat + beta


class THGNN(RULModel):
    def __init__(
        self,
        enc_in: int = 2,
        seq_len: int = 2048,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        graph_hidden_dim: int = 32,
        graph_depth: int = 2,
        temporal_kernel: int = 5,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        del n_heads, e_layers, d_ff
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.graph_mode = graph_mode
        self.prior_weight = prior_weight
        self._config = RULConfig(
            enc_in=enc_in,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=4,
            e_layers=2,
            d_ff=512,
            dropout=dropout,
        )

        self.input_proj = nn.Linear(1, d_model)
        self.temporal_block = TemporalConvBlock(d_model, temporal_kernel, dropout)
        self.graph_learner = SensorGraphLearner(enc_in, graph_hidden_dim)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)
        self.graph_prop = GraphPropagation(d_model, graph_depth, dropout)
        self.global_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.film = FiLMFusion(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * enc_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _resolve_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        learned_adj = self.graph_learner()
        prior_adj = None if self.prior_adjacency is None else self.prior_adjacency.to(device=device, dtype=dtype)
        if self.graph_mode == "learned":
            return learned_adj
        if self.graph_mode == "prior":
            if prior_adj is None:
                raise ValueError("graph_mode='prior' requires adjacency")
            return prior_adj
        if self.graph_mode == "hybrid":
            if prior_adj is None:
                raise ValueError("graph_mode='hybrid' requires adjacency")
            return self.prior_weight * prior_adj + (1.0 - self.prior_weight) * learned_adj
        raise ValueError(f"Unsupported graph_mode: {self.graph_mode}")

    def forward(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        del x_mark
        if x.dim() != 3:
            raise ValueError("THGNN expects x with shape (batch, seq_len, num_features)")
        if x.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x.size(-1)}")

        adj = self._resolve_adjacency(x.device, x.dtype)
        local = x.transpose(1, 2).unsqueeze(-1)
        local = self.input_proj(local).permute(0, 1, 3, 2)
        batch, nodes, hidden, steps = local.shape
        local = local.reshape(batch * nodes, hidden, steps)
        local = self.temporal_block(local)
        local = local.reshape(batch, nodes, hidden, steps).permute(0, 3, 1, 2)
        local = self.graph_prop(local, adj)

        global_feat = x.mean(dim=-1, keepdim=True)
        global_feat = self.input_proj(global_feat)
        global_feat, _ = self.global_encoder(global_feat)
        global_feat = global_feat.unsqueeze(2).expand(-1, -1, self.enc_in, -1)

        fused = self.film(local, global_feat)
        fused = fused.mean(dim=1).reshape(batch, nodes * hidden)
        return self.head(fused).squeeze(-1)
