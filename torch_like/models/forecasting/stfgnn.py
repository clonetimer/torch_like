"""STFGNN-inspired spatial-temporal fusion graph forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class STFGNNGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.spatial_src = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.spatial_dst = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.temporal_src = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.temporal_dst = nn.Parameter(torch.randn(num_nodes, emb_dim))
        for param in [self.spatial_src, self.spatial_dst, self.temporal_src, self.temporal_dst]:
            nn.init.xavier_uniform_(param)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = torch.softmax(F.relu(self.spatial_src @ self.spatial_dst.transpose(0, 1)), dim=-1)
        temporal = torch.softmax(F.relu(self.temporal_src @ self.temporal_dst.transpose(0, 1)), dim=-1)
        return spatial, temporal


class STFGNNBlock(nn.Module):
    def __init__(self, channels: int, seq_len: int, dropout: float):
        super().__init__()
        self.spatial_proj = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, spatial_adj: torch.Tensor, temporal_adj: torch.Tensor) -> torch.Tensor:
        spatial = torch.einsum("bcnt,nm->bcmt", x, spatial_adj)
        temporal = torch.einsum("bcnt,nm->bcmt", x, temporal_adj)
        spatial = self.spatial_proj(spatial)
        temporal = self.temporal_conv(temporal)
        fused = torch.cat([spatial, temporal], dim=1)
        gate = torch.sigmoid(self.gate(fused))
        out = gate * spatial + (1.0 - gate) * temporal
        return self.norm(x + self.dropout(out[..., : self.seq_len]))


class STFGNN(ForecastingModel):
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
        adaptive_embedding_dim: int = 16,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, d_ff, embed, freq, activation, output_attention
        self.seq_len = seq_len
        self.label_len = label_len
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
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=1,
            e_layers=e_layers,
            d_layers=1,
            d_ff=128,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
        )

        self.graph_learner = STFGNNGraphLearner(enc_in, adaptive_embedding_dim)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)

        self.input_proj = nn.Conv2d(1, d_model, kernel_size=(1, 1))
        self.blocks = nn.ModuleList([STFGNNBlock(d_model, seq_len, dropout) for _ in range(e_layers)])
        self.temporal_head = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _resolve_graphs(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        learned_spatial, learned_temporal = self.graph_learner()
        learned_spatial = learned_spatial.to(device=device, dtype=dtype)
        learned_temporal = learned_temporal.to(device=device, dtype=dtype)
        prior_adj = None if self.prior_adjacency is None else self.prior_adjacency.to(device=device, dtype=dtype)
        if self.graph_mode == "learned":
            return learned_spatial, learned_temporal
        if self.graph_mode == "prior":
            if prior_adj is None:
                raise ValueError("graph_mode='prior' requires adjacency")
            return prior_adj, prior_adj
        if self.graph_mode == "hybrid":
            if prior_adj is None:
                raise ValueError("graph_mode='hybrid' requires adjacency")
            spatial = self.prior_weight * prior_adj + (1.0 - self.prior_weight) * learned_spatial
            temporal = self.prior_weight * prior_adj + (1.0 - self.prior_weight) * learned_temporal
            return spatial, temporal
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
            raise ValueError("STFGNN expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        spatial_adj, temporal_adj = self._resolve_graphs(x_enc.device, x_enc.dtype)
        x = self.input_proj(x_enc.transpose(1, 2).unsqueeze(1))
        for block in self.blocks:
            x = block(x, spatial_adj, temporal_adj)
        x = self.temporal_head(x.mean(dim=1)).transpose(1, 2)
        return self.output_proj(x)
