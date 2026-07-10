"""GTS: discrete graph structure learning for multivariate forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class GTSGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, node_dim: int, temperature: float = 0.5):
        super().__init__()
        self.src = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.dst = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.temperature = temperature
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def edge_logits(self) -> torch.Tensor:
        return self.src @ self.dst.transpose(0, 1)

    def forward(self, hard: bool = False) -> torch.Tensor:
        logits = self.edge_logits()
        logits = logits - torch.diag_embed(torch.diagonal(logits))
        stacked = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        sample = F.gumbel_softmax(stacked, tau=self.temperature, hard=hard, dim=-1)[..., 1]
        return sample


class GTSDiffusionGraphConv(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k_steps: int):
        super().__init__()
        self.k_steps = k_steps
        self.proj = nn.Linear((2 * k_steps + 1) * input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        supports = [x]
        x_fwd = x
        x_bwd = x
        adj_t = adj.transpose(0, 1)
        for _ in range(self.k_steps):
            x_fwd = torch.einsum("bnc,nm->bmc", x_fwd, adj)
            x_bwd = torch.einsum("bnc,nm->bmc", x_bwd, adj_t)
            supports.extend([x_fwd, x_bwd])
        return self.proj(torch.cat(supports, dim=-1))


class GTSGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k_steps: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_conv = GTSDiffusionGraphConv(input_dim + hidden_dim, 2 * hidden_dim, k_steps)
        self.cand_conv = GTSDiffusionGraphConv(input_dim + hidden_dim, hidden_dim, k_steps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        z_r = torch.sigmoid(self.gate_conv(gate_input, adj))
        z, r = torch.chunk(z_r, 2, dim=-1)
        cand_input = torch.cat([x_t, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.cand_conv(cand_input, adj))
        h = (1.0 - z) * h_prev + z * h_tilde
        return self.dropout(h)


class GTS(ForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        diffusion_steps: int = 2,
        node_dim: int = 16,
        graph_temperature: float = 0.5,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
        hard_graph_sampling: bool = False,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, d_ff, embed, freq, activation, output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.hidden_dim = d_model
        self.graph_mode = graph_mode
        self.prior_weight = prior_weight
        self.hard_graph_sampling = hard_graph_sampling
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=1,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_model,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="tanh",
            output_attention=False,
        )
        self.graph_learner = GTSGraphLearner(enc_in, node_dim=node_dim, temperature=graph_temperature)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)

        self.input_proj = nn.Linear(1, d_model)
        self.cells = nn.ModuleList(
            [GTSGRUCell(input_dim=d_model, hidden_dim=d_model, k_steps=diffusion_steps, dropout=dropout) for _ in range(e_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(seq_len * d_model, max(d_model, pred_len * 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model, pred_len * 2), pred_len),
        )
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def learned_adjacency(self, hard: bool | None = None) -> torch.Tensor:
        if hard is None:
            hard = self.hard_graph_sampling
        return self.graph_learner(hard=hard)

    def _resolve_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        learned_adj = self.learned_adjacency().to(device=device, dtype=dtype)
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
            raise ValueError("GTS expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        adj = self._resolve_adjacency(x_enc.device, x_enc.dtype)
        x = self.input_proj(x_enc.transpose(1, 2).unsqueeze(-1))
        batch_size, num_nodes, steps, hidden_dim = x.shape
        states = [x.new_zeros(batch_size, num_nodes, hidden_dim) for _ in self.cells]
        outputs = []
        for step in range(steps):
            h = x[:, :, step, :]
            for layer_idx, cell in enumerate(self.cells):
                states[layer_idx] = cell(h, states[layer_idx], adj)
                h = states[layer_idx]
            outputs.append(h)
        x = torch.stack(outputs, dim=2).reshape(batch_size, num_nodes, steps * hidden_dim)
        x = self.readout(x).transpose(1, 2)
        x = self.output_proj(x)
        return x
