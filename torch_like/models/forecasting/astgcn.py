"""ASTGCN-inspired attention-based spatial-temporal graph forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class ASTGCNGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.src = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.dst = nn.Parameter(torch.randn(num_nodes, emb_dim))
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def forward(self) -> torch.Tensor:
        logits = torch.relu(self.src @ self.dst.transpose(0, 1))
        return torch.softmax(logits, dim=-1)


class SpatialAttention(nn.Module):
    def __init__(self, seq_len: int, num_nodes: int):
        super().__init__()
        self.time_proj = nn.Linear(seq_len, seq_len)
        self.node_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F, T)
        scores = torch.einsum("bnft,bmft->bnm", x, x) / max(x.size(2) * x.size(3), 1)
        scores = scores + self.node_bias
        return torch.softmax(scores, dim=-1)


class TemporalAttention(nn.Module):
    def __init__(self, num_nodes: int, seq_len: int):
        super().__init__()
        self.node_proj = nn.Linear(num_nodes, num_nodes)
        self.time_bias = nn.Parameter(torch.zeros(seq_len, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F, T)
        scores = torch.einsum("bnft,bnfs->bts", x, x) / max(x.size(1) * x.size(2), 1)
        scores = scores + self.time_bias
        return torch.softmax(scores, dim=-1)


class ASTGCNBlock(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, seq_len: int, dropout: float):
        super().__init__()
        self.temporal_attention = TemporalAttention(num_nodes, seq_len)
        self.spatial_attention = SpatialAttention(seq_len, num_nodes)
        self.feature_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1))
        self.residual_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T)
        temporal_scores = self.temporal_attention(x.permute(0, 2, 1, 3))
        x_t = torch.einsum("bcnt,bts->bcns", x, temporal_scores)
        spatial_scores = self.spatial_attention(x_t.permute(0, 2, 1, 3))
        combined_adj = 0.5 * adj.unsqueeze(0) + 0.5 * spatial_scores
        x_s = torch.einsum("bcnt,bnm->bcmt", x_t, combined_adj)
        x_s = self.feature_proj(x_s)
        x_s = F.relu(self.temporal_conv(x_s))
        residual = self.residual_proj(x)
        return self.norm(self.dropout(x_s) + residual)


class ASTGCN(ForecastingModel):
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
        d_ff: int = 64,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
        adaptive_embedding_dim: int = 16,
    ):
        super().__init__()
        del dec_in, label_len, n_heads, d_layers, d_ff, embed, freq, activation, output_attention
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.graph_mode = graph_mode
        self.prior_weight = prior_weight
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=0,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=1,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_model,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
        )
        self.graph_learner = ASTGCNGraphLearner(enc_in, adaptive_embedding_dim)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)
        self.input_proj = nn.Conv2d(1, d_model, kernel_size=(1, 1))
        self.blocks = nn.ModuleList(
            [ASTGCNBlock(num_nodes=enc_in, in_channels=d_model, hidden_channels=d_model, seq_len=seq_len, dropout=dropout) for _ in range(e_layers)]
        )
        self.readout = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(d_model, pred_len, kernel_size=(1, seq_len)),
        )
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
            raise ValueError("ASTGCN expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        adj = self._resolve_adjacency(x_enc.device, x_enc.dtype)
        x = x_enc.transpose(1, 2).unsqueeze(1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, adj)
        x = self.readout(x).squeeze(-1)
        x = self.output_proj(x)
        return x
