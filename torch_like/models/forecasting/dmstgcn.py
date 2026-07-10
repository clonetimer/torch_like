"""DMSTGCN-inspired dynamic multi-faceted spatio-temporal forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class DMSTGCNGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.src = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.dst = nn.Parameter(torch.randn(num_nodes, emb_dim))
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def forward(self) -> torch.Tensor:
        logits = torch.relu(self.src @ self.dst.transpose(0, 1))
        return torch.softmax(logits, dim=-1)


class DynamicGraphGenerator(nn.Module):
    def __init__(self, num_nodes: int, channels: int, time_emb_dim: int):
        super().__init__()
        self.time_query = nn.Linear(channels, time_emb_dim)
        self.time_key = nn.Linear(channels, time_emb_dim)
        self.num_nodes = num_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T)
        summary = x.mean(dim=-1).transpose(1, 2)  # (B, N, C)
        query = self.time_query(summary)
        key = self.time_key(summary)
        scores = torch.einsum("bnd,bmd->bnm", query, key) / max(query.size(-1), 1)
        return torch.softmax(torch.relu(scores), dim=-1)


class FacetFusion(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, primary: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([primary, auxiliary], dim=1))
        return self.dropout(gate * auxiliary + (1.0 - gate) * primary)


class DMSTGCNBlock(nn.Module):
    def __init__(self, channels: int, skip_channels: int, dropout: float):
        super().__init__()
        self.primary_conv = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.aux_conv = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2))
        self.fusion = FacetFusion(channels, dropout)
        self.skip_proj = nn.Conv2d(channels, skip_channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, dynamic_adj: torch.Tensor, static_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        primary = F.relu(self.primary_conv(x))
        auxiliary = F.relu(self.aux_conv(x))
        primary = torch.einsum("bcnt,bnm->bcmt", primary, dynamic_adj)
        auxiliary = torch.einsum("bcnt,nm->bcmt", auxiliary, static_adj)
        fused = self.fusion(primary, auxiliary)
        out = self.norm(fused + x)
        return out, self.skip_proj(out[..., -1:])


class DMSTGCN(ForecastingModel):
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
        skip_channels: int = 32,
        end_channels: int = 64,
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
        self.graph_learner = DMSTGCNGraphLearner(enc_in, adaptive_embedding_dim)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)
        self.input_proj = nn.Conv2d(1, d_model, kernel_size=(1, 1))
        self.dynamic_graph = DynamicGraphGenerator(enc_in, d_model, adaptive_embedding_dim)
        self.blocks = nn.ModuleList([DMSTGCNBlock(d_model, skip_channels, dropout) for _ in range(e_layers)])
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len, kernel_size=(1, 1))
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _resolve_static_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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
            raise ValueError("DMSTGCN expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        static_adj = self._resolve_static_adjacency(x_enc.device, x_enc.dtype)
        x = x_enc.transpose(1, 2).unsqueeze(1)
        x = self.input_proj(x)
        dynamic_adj = self.dynamic_graph(x)
        skip = None
        for block in self.blocks:
            x, block_skip = block(x, dynamic_adj, static_adj)
            skip = block_skip if skip is None else skip + block_skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x).squeeze(-1)
        x = self.output_proj(x)
        return x
