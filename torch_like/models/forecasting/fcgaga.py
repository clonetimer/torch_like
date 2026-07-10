"""FC-GAGA-inspired gated graph forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class FCGAGAGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.src = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.dst = nn.Parameter(torch.randn(num_nodes, emb_dim))
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def forward(self) -> torch.Tensor:
        scores = self.src @ self.dst.transpose(0, 1)
        return torch.softmax(scores, dim=-1)


class HardGraphGate(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.node_proj = nn.Linear(num_nodes, hidden_dim)
        self.context_proj = nn.Linear(num_nodes, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L), adjacency: (C, C)
        node_summary = x.mean(dim=-1)
        context = node_summary @ adjacency.transpose(0, 1)
        hidden = torch.tanh(self.node_proj(node_summary) + self.context_proj(context))
        logits = self.out_proj(hidden)
        gates = F.softmax(logits, dim=-1)
        return gates.unsqueeze(-1)


class FCGAGABlock(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.backcast = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
        )
        self.forecast = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.norm(x)
        backcast = self.backcast(residual)
        forecast = self.forecast(residual)
        return x - backcast, forecast


class FCGAGA(ForecastingModel):
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
        e_layers: int = 3,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        hidden_dim: int | None = None,
        adaptive_embedding_dim: int = 16,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        del dec_in, d_model, n_heads, d_layers, embed, freq, activation, output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.graph_mode = graph_mode
        self.prior_weight = prior_weight
        self.hidden_dim = hidden_dim or d_ff
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=64,
            n_heads=1,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
        )

        self.graph_learner = FCGAGAGraphLearner(enc_in, adaptive_embedding_dim)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)

        self.input_norm = nn.LayerNorm(enc_in)
        self.gate = HardGraphGate(enc_in, self.hidden_dim)
        self.blocks = nn.ModuleList([FCGAGABlock(seq_len, self.hidden_dim, dropout) for _ in range(e_layers)])
        self.head = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _resolve_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        learned_adj = self.graph_learner().to(device=device, dtype=dtype)
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
            raise ValueError("FCGAGA expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        adj = self._resolve_adjacency(x_enc.device, x_enc.dtype)
        x = self.input_norm(x_enc)
        x = x.transpose(1, 2)
        gate = self.gate(x, adj)
        x = x * gate + torch.matmul(adj, x)

        forecast = 0.0
        for block in self.blocks:
            x, block_forecast = block(x)
            forecast = forecast + block_forecast

        out = self.head(forecast).transpose(1, 2)
        return self.output_proj(out)
