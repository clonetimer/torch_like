"""AGCRN: adaptive graph convolutional recurrent forecasting.

This is a framework-oriented implementation of the AGCRN architecture family.
It keeps the torch_like graph forecasting contract while exposing the same graph
mode convention used by the other graph spatiotemporal models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class AGCRNConfig(ForecastingConfig):
    node_dim: int = 16
    rnn_units: int = 32
    num_layers: int = 1
    graph_mode: str = "learned"
    prior_weight: float = 0.5


class NodeAdaptiveGraph(nn.Module):
    def __init__(self, num_nodes: int, node_dim: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self) -> torch.Tensor:
        scores = F.relu(self.embeddings @ self.embeddings.T)
        return torch.softmax(scores, dim=-1)


class AdaptiveGraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, node_dim: int):
        super().__init__()
        self.weight_pool = nn.Parameter(torch.empty(node_dim, input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.empty(node_dim, output_dim))
        self.node_embeddings = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.weight_pool)
        nn.init.zeros_(self.bias_pool)
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C), adj: (N, N)
        x = torch.einsum("nm,bmc->bnc", adj, x)
        weights = torch.einsum("nd,dio->nio", self.node_embeddings, self.weight_pool)
        bias = self.node_embeddings @ self.bias_pool
        return torch.einsum("bni,nio->bno", x, weights) + bias


class AGCRNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int, node_dim: int):
        super().__init__()
        gate_dim = input_dim + hidden_dim
        self.hidden_dim = hidden_dim
        self.gate = AdaptiveGraphConv(gate_dim, 2 * hidden_dim, num_nodes, node_dim)
        self.update = AdaptiveGraphConv(gate_dim, hidden_dim, num_nodes, node_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x, h], dim=-1)
        z_r = torch.sigmoid(self.gate(gate_input, adj))
        z, r = torch.chunk(z_r, 2, dim=-1)
        candidate_input = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.update(candidate_input, adj))
        return z * h + (1.0 - z) * h_tilde


class AGCRN(ForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 32,
        n_heads: int = 4,
        e_layers: int = 1,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        node_dim: int = 16,
        rnn_units: int | None = None,
        num_layers: int | None = None,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, d_ff, embed, freq, activation, output_attention
        hidden_dim = rnn_units if rnn_units is not None else d_model
        layers = num_layers if num_layers is not None else e_layers
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_nodes = enc_in
        self.graph_mode = graph_mode
        self.prior_weight = prior_weight
        self._config = AGCRNConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=hidden_dim,
            e_layers=layers,
            dropout=dropout,
            node_dim=node_dim,
            rnn_units=hidden_dim,
            num_layers=layers,
            graph_mode=graph_mode,
            prior_weight=prior_weight,
        )

        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)

        self.graph = NodeAdaptiveGraph(enc_in, node_dim)
        self.cells = nn.ModuleList(
            [
                AGCRNCell(
                    input_dim=1 if layer_idx == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    num_nodes=enc_in,
                    node_dim=node_dim,
                )
                for layer_idx in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.horizon_proj = nn.Linear(hidden_dim, pred_len)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _normalize(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj.clamp_min(0)
        return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _resolve_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        learned_adj = self.graph().to(device=device, dtype=dtype)
        prior_adj = (
            None
            if self.prior_adjacency is None
            else self._normalize(self.prior_adjacency.to(device=device, dtype=dtype))
        )
        if self.graph_mode == "learned":
            return learned_adj
        if self.graph_mode == "prior":
            if prior_adj is None:
                raise ValueError("graph_mode='prior' requires adjacency")
            return prior_adj
        if self.graph_mode == "hybrid":
            if prior_adj is None:
                raise ValueError("graph_mode='hybrid' requires adjacency")
            return self._normalize(self.prior_weight * prior_adj + (1.0 - self.prior_weight) * learned_adj)
        raise ValueError(f"Unsupported graph_mode: {self.graph_mode}")

    def learned_adjacency(self) -> torch.Tensor:
        return self.graph()

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        if x_enc.dim() != 3:
            raise ValueError("AGCRN expects x_enc with shape (batch, seq_len, num_nodes)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input features, got {x_enc.size(-1)}")

        adj = self._resolve_adjacency(x_enc.device, x_enc.dtype)
        batch_size = x_enc.size(0)
        states = [x_enc.new_zeros(batch_size, self.num_nodes, cell.hidden_dim) for cell in self.cells]
        for step in range(x_enc.size(1)):
            layer_input = x_enc[:, step, :].unsqueeze(-1)
            for idx, cell in enumerate(self.cells):
                states[idx] = cell(layer_input, states[idx], adj)
                layer_input = self.dropout(states[idx])

        node_forecast = self.horizon_proj(states[-1]).transpose(1, 2)
        return self.output_proj(node_forecast)
