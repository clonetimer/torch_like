"""StemGNN-inspired forecasting model with latent graph spectral-temporal blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class GLUBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * torch.sigmoid(self.gate(x))


class SpectralTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.temporal_glu = GLUBlock(hidden_dim)
        self.spectral_glu = GLUBlock(hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.temporal_glu(x)
        freq = torch.fft.rfft(x, dim=1)
        freq_feat = torch.cat([freq.real, freq.imag], dim=-1)
        freq_feat = self.spectral_glu(freq_feat)
        real, imag = freq_feat.chunk(2, dim=-1)
        freq = torch.complex(real, imag)
        x = torch.fft.irfft(freq, n=x.size(1), dim=1)
        x = self.out_proj(torch.cat([x, residual], dim=-1))
        x = self.dropout(x)
        return self.norm(x + residual)


class LatentGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.src = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.dst = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def forward(self) -> torch.Tensor:
        logits = torch.relu(self.src @ self.dst.T - self.dst @ self.src.T)
        return torch.softmax(logits, dim=-1)


class StemGNN(ForecastingModel):
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
        dropout: float = 0.2,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        stack_cnt: int = 2,
        latent_dim: int = 32,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
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
            n_heads=4,
            e_layers=e_layers,
            d_layers=1,
            d_ff=128,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
        )

        self.input_proj = nn.Linear(1, d_model)
        self.graph_learner = LatentGraphLearner(enc_in, latent_dim)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (enc_in, enc_in):
                raise ValueError(f"adjacency must have shape ({enc_in}, {enc_in}), got {tuple(adjacency.shape)}")
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)
        self.blocks = nn.ModuleList([SpectralTemporalBlock(d_model, dropout) for _ in range(max(stack_cnt, e_layers))])
        self.readout = nn.Sequential(
            nn.Linear(seq_len * d_model, max(d_model, pred_len * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model, pred_len * 2), pred_len),
        )
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

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
            raise ValueError("StemGNN expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        adj = self._resolve_adjacency(x_enc.device, x_enc.dtype)
        x = x_enc.transpose(1, 2).unsqueeze(-1)
        x = self.input_proj(x)
        x = torch.einsum("nm,bmtd->bntd", adj, x)
        batch, nodes, steps, hidden = x.shape
        x = x.reshape(batch * nodes, steps, hidden)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(batch, nodes, steps * hidden)
        x = self.readout(x).transpose(1, 2)
        x = self.output_proj(x)
        return x
