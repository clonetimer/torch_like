"""MTGNN: multivariate time series forecasting with learned graph structure."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class NConv(nn.Module):
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()


class Linear2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MixProp(nn.Module):
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super().__init__()
        self.nconv = NConv()
        self.mlp = Linear2d((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=x.device, dtype=x.dtype)
        degree = adj.sum(1)
        norm_adj = adj / degree.unsqueeze(1).clamp_min(1e-6)
        h = x
        states = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1.0 - self.alpha) * self.nconv(h, norm_adj)
            states.append(h)
        out = torch.cat(states, dim=1)
        out = self.mlp(out)
        return F.dropout(out, self.dropout, training=self.training)


class DilatedInception(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation_factor: int = 2):
        super().__init__()
        self.kernel_set = [2, 3, 6, 7]
        per_branch = c_out // len(self.kernel_set)
        self.tconv = nn.ModuleList(
            [
                nn.Conv2d(c_in, per_branch, kernel_size=(1, kern), dilation=(1, dilation_factor))
                for kern in self.kernel_set
            ]
        )
        self.dilation_factor = dilation_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        required = self.dilation_factor * (max(self.kernel_set) - 1) + 1
        if x.size(3) < required:
            x = F.pad(x, (required - x.size(3), 0, 0, 0))
        outs = [conv(x) for conv in self.tconv]
        min_time = min(out.size(3) for out in outs)
        outs = [out[..., -min_time:] for out in outs]
        return torch.cat(outs, dim=1)


class GraphConstructor(nn.Module):
    def __init__(self, num_nodes: int, node_dim: int, alpha: float = 3.0, top_k: int | None = None):
        super().__init__()
        self.emb1 = nn.Embedding(num_nodes, node_dim)
        self.emb2 = nn.Embedding(num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.top_k = top_k if top_k is not None else num_nodes

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        nodevec1 = torch.tanh(self.alpha * self.lin1(self.emb1(idx)))
        nodevec2 = torch.tanh(self.alpha * self.lin2(self.emb2(idx)))
        scores = torch.relu(torch.tanh(self.alpha * (nodevec1 @ nodevec2.T - nodevec2 @ nodevec1.T)))
        if self.top_k < self.num_nodes:
            mask = torch.zeros_like(scores)
            _, indices = scores.topk(self.top_k, dim=1)
            mask.scatter_(1, indices, 1.0)
            scores = scores * mask
        return scores


class MTGNN(ForecastingModel):
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
        dropout: float = 0.3,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        gcn_depth: int = 2,
        subgraph_size: int | None = None,
        node_dim: int = 32,
        dilation_exponential: int = 2,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        prop_alpha: float = 0.05,
        layer_norm_affine: bool = True,
        graph_mode: str = "learned",
        adjacency=None,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        del dec_in, d_layers, n_heads
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.num_nodes = enc_in
        self.output_attention = output_attention
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
            n_heads=4,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=dropout,
            embed=embed,
            freq=freq,
            activation=activation,
            output_attention=output_attention,
        )

        self.idx = torch.arange(self.num_nodes)
        self.gc = GraphConstructor(self.num_nodes, node_dim=node_dim, alpha=3.0, top_k=subgraph_size)
        if adjacency is not None:
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            if adjacency.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"adjacency must have shape ({self.num_nodes}, {self.num_nodes}), got {tuple(adjacency.shape)}"
                )
            self.register_buffer("prior_adjacency", adjacency)
        else:
            self.register_buffer("prior_adjacency", None)
        self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norms = nn.ModuleList()

        receptive_field = 1
        dilation = 1
        for _ in range(e_layers):
            self.filter_convs.append(DilatedInception(residual_channels, conv_channels, dilation))
            self.gate_convs.append(DilatedInception(residual_channels, conv_channels, dilation))
            self.residual_convs.append(nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, 1)))
            self.gconv1.append(MixProp(conv_channels, residual_channels, gcn_depth, dropout, prop_alpha))
            self.gconv2.append(MixProp(conv_channels, residual_channels, gcn_depth, dropout, prop_alpha))
            self.norms.append(
                nn.LayerNorm([residual_channels, self.num_nodes, 1], elementwise_affine=layer_norm_affine)
            )
            receptive_field += (7 - 1) * dilation
            dilation *= dilation_exponential
        self.receptive_field = receptive_field

        self.skip0 = nn.Conv2d(1, skip_channels, kernel_size=(1, 1))
        self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len, kernel_size=(1, 1), bias=True)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def _apply_norm(self, x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(x[..., -1:].contiguous())

    def _resolve_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        learned_adj = self.gc(self.idx.to(device))
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
            raise ValueError("MTGNN expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(2) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input features, got {x_enc.size(2)}")

        x = x_enc.transpose(1, 2).unsqueeze(1)
        if self.seq_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - self.seq_len, 0, 0, 0))

        adj = self._resolve_adjacency(x.device, x.dtype)
        skip = self.skip0(x[..., -1:])
        x = self.start_conv(x)

        for filter_conv, gate_conv, residual_conv, skip_conv, gconv1, gconv2, norm in zip(
            self.filter_convs,
            self.gate_convs,
            self.residual_convs,
            self.skip_convs,
            self.gconv1,
            self.gconv2,
            self.norms,
            strict=True,
        ):
            residual = x
            filt = torch.tanh(filter_conv(x))
            gate = torch.sigmoid(gate_conv(x))
            x = filt * gate
            skip = skip + skip_conv(x[..., -1:])
            x = gconv1(x, adj) + gconv2(x, adj.transpose(0, 1))
            x = residual_conv(x)
            x = x + residual[..., -x.size(3) :]
            x = self._apply_norm(x, norm)

        x = self.skipE(x) + skip
        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x).squeeze(-1)
        x = self.output_proj(x)
        return x
