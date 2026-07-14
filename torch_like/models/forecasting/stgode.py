"""STGODE: spatial-temporal graph ODE forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class STGODEGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.physical_src = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.physical_dst = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.semantic_src = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.semantic_dst = nn.Parameter(torch.randn(num_nodes, emb_dim))
        for param in [self.physical_src, self.physical_dst, self.semantic_src, self.semantic_dst]:
            nn.init.xavier_uniform_(param)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        physical = torch.softmax(F.relu(self.physical_src @ self.physical_dst.transpose(0, 1)), dim=-1)
        semantic = torch.softmax(F.relu(self.semantic_src @ self.semantic_dst.transpose(0, 1)), dim=-1)
        return physical, semantic


class ODEGraphFunction(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.graph_proj = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.temporal_proj = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, spatial_adj: torch.Tensor, semantic_adj: torch.Tensor) -> torch.Tensor:
        spatial = torch.einsum("bcnt,nm->bcmt", x, spatial_adj)
        semantic = torch.einsum("bcnt,nm->bcmt", x, semantic_adj)
        spatial = self.graph_proj(spatial)
        semantic = self.temporal_proj(semantic)
        fused = torch.cat([spatial, semantic], dim=1)
        return self.dropout(torch.tanh(self.fusion(fused)))


class ODEBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, num_steps: int = 3, step_size: float = 1.0):
        super().__init__()
        self.func = ODEGraphFunction(channels=channels, dropout=dropout)
        self.num_steps = num_steps
        self.step_size = step_size / max(1, num_steps)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, spatial_adj: torch.Tensor, semantic_adj: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(self.num_steps):
            h = h + self.step_size * self.func(h, spatial_adj, semantic_adj)
        return self.norm(x + h)


class STGODE(ForecastingModel):
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
        e_layers: int = 3,
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
        semantic_adjacency=None,
        prior_weight: float = 0.5,
        ode_steps: int = 3,
        ode_step_size: float = 1.0,
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
            d_ff=d_model,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="tanh",
            output_attention=False,
        )
        self.graph_learner = STGODEGraphLearner(enc_in, adaptive_embedding_dim)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)
        if semantic_adjacency is not None:
            semantic_adjacency = torch.as_tensor(semantic_adjacency, dtype=torch.float32)
            if semantic_adjacency.shape != (enc_in, enc_in):
                raise ValueError(
                    f"semantic_adjacency must have shape ({enc_in}, {enc_in}), got {tuple(semantic_adjacency.shape)}"
                )
            self.register_buffer("prior_semantic_adjacency", semantic_adjacency)
        else:
            self.register_buffer("prior_semantic_adjacency", None)

        self.input_proj = nn.Conv2d(1, d_model, kernel_size=(1, 1))
        self.start_temporal = nn.Conv2d(d_model, d_model, kernel_size=(1, 3), padding=(0, 1))
        self.blocks = nn.ModuleList(
            [
                ODEBlock(channels=d_model, dropout=dropout, num_steps=ode_steps, step_size=ode_step_size)
                for _ in range(e_layers)
            ]
        )
        self.temporal_head = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _resolve_graphs(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        learned_physical, learned_semantic = self.graph_learner()
        learned_physical = learned_physical.to(device=device, dtype=dtype)
        learned_semantic = learned_semantic.to(device=device, dtype=dtype)
        prior_adj = None if self.prior_adjacency is None else self.prior_adjacency.to(device=device, dtype=dtype)
        prior_sem = (
            None
            if self.prior_semantic_adjacency is None
            else self.prior_semantic_adjacency.to(device=device, dtype=dtype)
        )
        if self.graph_mode == "learned":
            return learned_physical, learned_semantic
        if self.graph_mode == "prior":
            if prior_adj is None:
                raise ValueError("graph_mode='prior' requires adjacency")
            if prior_sem is None:
                prior_sem = prior_adj
            return prior_adj, prior_sem
        if self.graph_mode == "hybrid":
            if prior_adj is None:
                raise ValueError("graph_mode='hybrid' requires adjacency")
            if prior_sem is None:
                prior_sem = prior_adj
            physical = self.prior_weight * prior_adj + (1.0 - self.prior_weight) * learned_physical
            semantic = self.prior_weight * prior_sem + (1.0 - self.prior_weight) * learned_semantic
            return physical, semantic
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
            raise ValueError("STGODE expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        physical_adj, semantic_adj = self._resolve_graphs(x_enc.device, x_enc.dtype)
        x = self.input_proj(x_enc.transpose(1, 2).unsqueeze(1))
        x = torch.tanh(self.start_temporal(x))
        for block in self.blocks:
            x = block(x, physical_adj, semantic_adj)
        x = self.temporal_head(x.mean(dim=1)).transpose(1, 2)
        return self.output_proj(x)
