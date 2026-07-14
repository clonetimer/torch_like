"""Graph WaveNet-inspired graph temporal forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class GraphWaveNetGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.src = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.dst = nn.Parameter(torch.randn(num_nodes, emb_dim))
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def forward(self) -> torch.Tensor:
        scores = F.relu(self.src @ self.dst.transpose(0, 1))
        return torch.softmax(scores, dim=-1)


class GraphWaveNetBlock(nn.Module):
    def __init__(self, channels: int, skip_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size), dilation=(1, dilation), padding=(0, padding)
        )
        self.gate_conv = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size), dilation=(1, dilation), padding=(0, padding)
        )
        self.residual_proj = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.skip_proj = nn.Conv2d(channels, skip_channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        filt = torch.tanh(self.filter_conv(x)[..., : x.size(-1)])
        gate = torch.sigmoid(self.gate_conv(x)[..., : x.size(-1)])
        x = filt * gate
        skip = self.skip_proj(x)
        x = self.dropout(self.residual_proj(x))
        x = self.norm(x + residual)
        return x, skip


class GraphWaveNet(ForecastingModel):
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
        e_layers: int = 4,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.3,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        kernel_size: int = 2,
        adaptive_embedding_dim: int = 16,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        del dec_in, n_heads, d_layers, embed, freq, activation, output_attention, d_model
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
            d_model=residual_channels,
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

        self.graph_learner = GraphWaveNetGraphLearner(enc_in, adaptive_embedding_dim)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)

        self.input_proj = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))
        self.blocks = nn.ModuleList(
            [
                GraphWaveNetBlock(
                    channels=residual_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(e_layers)
            ]
        )
        self.graph_proj = nn.Conv2d(residual_channels, residual_channels, kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len, kernel_size=(1, 1))
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

    def _graph_propagate(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("bcnt,nm->bcmt", x, adj)
        return self.graph_proj(x)

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
            raise ValueError("GraphWaveNet expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        adj = self._resolve_adjacency(x_enc.device, x_enc.dtype)
        x = x_enc.transpose(1, 2).unsqueeze(1)
        x = self.input_proj(x)
        skip = None
        for block in self.blocks:
            x, block_skip = block(x)
            x = x + self._graph_propagate(x, adj)
            skip = block_skip if skip is None else skip + block_skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x[..., -1:]))
        x = self.end_conv_2(x).squeeze(-1)
        x = self.output_proj(x)
        return x
