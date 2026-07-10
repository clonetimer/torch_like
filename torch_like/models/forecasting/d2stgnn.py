"""D2STGNN-style dynamic decoupled spatiotemporal graph forecaster."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class D2STGNNConfig(ForecastingConfig):
    graph_mode: str = "learned"
    prior_weight: float = 0.5
    gcn_depth: int = 2
    node_dim: int = 16
    diffusion_weight: float = 0.5


class DynamicGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, d_model: int, node_dim: int):
        super().__init__()
        self.src = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.dst = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.context = nn.Linear(d_model, node_dim)
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, N, D)
        context = torch.tanh(self.context(h.mean(dim=1)))
        src = self.src.unsqueeze(0) + context
        dst = self.dst.unsqueeze(0) + context
        scores = torch.relu(torch.einsum("bnd,bmd->bnm", src, dst))
        return torch.softmax(scores, dim=-1)


class BatchGraphPropagation(nn.Module):
    def __init__(self, d_model: int, gcn_depth: int, dropout: float, alpha: float = 0.1):
        super().__init__()
        self.gcn_depth = gcn_depth
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear((gcn_depth + 1) * d_model, d_model)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        states = [h]
        x0 = h
        x = h
        for _ in range(self.gcn_depth):
            x = self.alpha * x0 + (1.0 - self.alpha) * torch.einsum("blnd,bnm->blmd", x, adj)
            states.append(x)
        return self.dropout(self.proj(torch.cat(states, dim=-1)))


class D2STGNNBlock(nn.Module):
    def __init__(self, d_model: int, gcn_depth: int, dropout: float, diffusion_weight: float):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.diffusion = BatchGraphPropagation(d_model, gcn_depth, dropout)
        self.temporal = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 3), padding=(0, 1), groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
        )
        self.intrinsic = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        diffusion = self.diffusion(h, adj)
        temporal = self.temporal(h.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        intrinsic = self.intrinsic(h)
        mixed = self.diffusion_weight * diffusion + (1.0 - self.diffusion_weight) * intrinsic + temporal
        return self.norm(h + mixed)


class D2STGNN(ForecastingModel):
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
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
        gcn_depth: int = 2,
        node_dim: int = 16,
        diffusion_weight: float = 0.5,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, d_ff, embed, freq, activation, output_attention
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.graph_mode = graph_mode
        self.prior_weight = prior_weight
        self._config = D2STGNNConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            e_layers=e_layers,
            dropout=dropout,
            graph_mode=graph_mode,
            prior_weight=prior_weight,
            gcn_depth=gcn_depth,
            node_dim=node_dim,
            diffusion_weight=diffusion_weight,
        )

        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)

        self.input_proj = nn.Linear(1, d_model)
        self.graph_learner = DynamicGraphLearner(enc_in, d_model, node_dim)
        self.blocks = nn.ModuleList(
            [D2STGNNBlock(d_model, gcn_depth, dropout, diffusion_weight) for _ in range(e_layers)]
        )
        self.temporal_pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, pred_len))
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _normalize(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj.clamp_min(0)
        return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _resolve_adjacency(self, h: torch.Tensor) -> torch.Tensor:
        learned_adj = self.graph_learner(h)
        prior_adj = None
        if self.prior_adjacency is not None:
            prior = self._normalize(self.prior_adjacency.to(device=h.device, dtype=h.dtype))
            prior_adj = prior.unsqueeze(0).expand(h.size(0), -1, -1)
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

    def learned_adjacency(self, x_enc: torch.Tensor | None = None) -> torch.Tensor:
        if x_enc is None:
            x_enc = torch.zeros(1, self.seq_len, self.enc_in, device=self.input_proj.weight.device)
        h = self.input_proj(x_enc.unsqueeze(-1))
        return self.graph_learner(h).mean(dim=0)

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
            raise ValueError("D2STGNN expects x_enc with shape (batch, seq_len, num_nodes)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input features, got {x_enc.size(-1)}")

        h = self.input_proj(x_enc.unsqueeze(-1))
        adj = self._resolve_adjacency(h)
        for block in self.blocks:
            h = block(h, adj)
        node_state = h[:, -1]
        out = self.temporal_pool(node_state).transpose(1, 2)
        return self.output_proj(out)
