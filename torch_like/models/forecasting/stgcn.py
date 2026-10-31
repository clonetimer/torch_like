"""Fixed-graph STGCN with a direct multi-horizon forecasting adapter.

Architecture: Yu, Yin and Zhu, IJCAI 2018,
https://www.ijcai.org/proceedings/2018/505 . The BSD-2-Clause TensorFlow
reference is https://github.com/VeritasYin/STGCN_IJCAI-18 . This module is an
independent PyTorch implementation of the mathematical operators, not a copy
of reference code or a checkpoint-compatible reproduction.

The temporal GLU / Chebyshev graph ReLU / temporal ReLU sandwich is preserved.
Channel-only LayerNorm shares parameters across nodes, and the output head
predicts all horizons directly. Those are explicit framework-oriented changes
from the reference's node-specific affine normalization and one-step rollout.
Graph multiplication is dense; this implementation does not claim sparse
edge-linear complexity or published benchmark performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch
from torch import nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class STGCNConfig(ForecastingConfig):
    """Common forecasting dimensions plus the fixed graph and STGCN block sizes."""

    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    dropout: float = 0.1
    adjacency: Any = None
    bottleneck_channels: int = 16
    temporal_kernel_size: int = 3
    cheb_order: int = 3


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validated_adjacency(adjacency: Any, num_nodes: int) -> torch.Tensor:
    """Return an owned, detached CPU tensor without modifying the supplied graph."""

    if adjacency is None:
        raise ValueError("STGCN requires an explicit adjacency matrix; it does not learn a fallback graph")
    try:
        graph = torch.as_tensor(adjacency)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("adjacency must be a dense real numeric matrix") from exc
    if graph.layout != torch.strided or graph.is_complex():
        raise ValueError("adjacency must be a dense real numeric matrix")
    if graph.shape != (num_nodes, num_nodes):
        raise ValueError(f"adjacency must have shape ({num_nodes}, {num_nodes}), got {tuple(graph.shape)}")
    # Read the original values in float64: constructing a Python/NumPy graph
    # in the default dtype first could already erase very small positive edges.
    graph = torch.as_tensor(adjacency, device="cpu", dtype=torch.float64).detach().clone()
    if not torch.isfinite(graph).all():
        raise ValueError("adjacency must contain only finite values")
    if torch.any(graph < 0):
        raise ValueError("adjacency must be nonnegative")
    if not torch.equal(graph, graph.transpose(0, 1)):
        raise ValueError("adjacency must be symmetric; directed graphs are not supported")
    if not torch.isfinite(graph.sum(dim=1)).all():
        raise ValueError("adjacency row sums must be finite")
    return graph


def _scaled_normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Compute 2 D^(-1/2)(D-A)D^(-1/2) / lambda_max - I.

    Self-loop weights contribute to D and cancel from D-A. Isolated nodes
    have zero rows/columns in the normalized Laplacian. If the whole
    Laplacian is zero (edgeless or self-loop-only graph), lambda_max=2 is
    used as a safe convention, giving -I and no graph-based node mixing.
    The caller supplies a validated symmetric, nonnegative float64 graph.
    """

    degree = adjacency.sum(dim=1)
    inverse_sqrt = torch.zeros_like(degree)
    nonzero = degree > 0
    inverse_sqrt[nonzero] = degree[nonzero].rsqrt()
    laplacian = torch.diag(degree) - adjacency
    laplacian = inverse_sqrt[:, None] * laplacian * inverse_sqrt[None, :]
    largest = torch.linalg.eigvalsh(laplacian).amax()
    if largest <= torch.finfo(laplacian.dtype).eps:
        largest = laplacian.new_tensor(2.0)
    identity = torch.eye(adjacency.size(0), dtype=adjacency.dtype, device=adjacency.device)
    return 2.0 * laplacian / largest - identity


class _ChannelAlign(nn.Module):
    """Project a shrinking channel axis; zero-pad an expanding one."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.project = nn.Conv2d(in_channels, out_channels, 1) if in_channels > out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aligned = self.project(x)
        if aligned.size(1) < self.out_channels:
            zeros = aligned.new_zeros(
                aligned.size(0), self.out_channels - aligned.size(1), aligned.size(2), aligned.size(3)
            )
            aligned = torch.cat((aligned, zeros), dim=1)
        return aligned


class _TemporalConv(nn.Module):
    """Valid temporal convolution with a right-aligned residual path."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, gated: bool):
        super().__init__()
        self.kernel_size = kernel_size
        self.gated = gated
        self.align = _ChannelAlign(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * (2 if gated else 1), (1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.align(x)[..., self.kernel_size - 1 :]
        convolved = self.conv(x)
        if self.gated:
            value, gate = convolved.chunk(2, dim=1)
            return (value + residual) * torch.sigmoid(gate)
        return torch.relu(convolved + residual)


class _ChebyshevGraphConv(nn.Module):
    """Sum_k T_k(L_tilde) X Theta_k, with dense polynomial-feature recurrence."""

    def __init__(self, in_channels: int, out_channels: int, cheb_order: int):
        super().__init__()
        self.cheb_order = cheb_order
        self.weight = nn.Parameter(torch.empty(cheb_order, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weight.view(cheb_order * in_channels, out_channels))

    def forward(self, x: torch.Tensor, scaled_laplacian: torch.Tensor) -> torch.Tensor:
        previous = x
        output = torch.einsum("bcnt,co->bont", previous, self.weight[0])
        if self.cheb_order > 1:
            current = torch.einsum("nm,bcmt->bcnt", scaled_laplacian, x)
            output = output + torch.einsum("bcnt,co->bont", current, self.weight[1])
            for index in range(2, self.cheb_order):
                following = 2.0 * torch.einsum("nm,bcmt->bcnt", scaled_laplacian, current) - previous
                output = output + torch.einsum("bcnt,co->bont", following, self.weight[index])
                previous, current = current, following
        return output + self.bias[None, :, None, None]


class _NodeSharedLayerNorm(nn.Module):
    """Normalize channels independently at each node/time, with shared affine weights."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _STConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        bottleneck_channels: int,
        temporal_kernel_size: int,
        cheb_order: int,
        dropout: float,
    ):
        super().__init__()
        self.temporal_in = _TemporalConv(in_channels, hidden_channels, temporal_kernel_size, gated=True)
        self.graph = _ChebyshevGraphConv(hidden_channels, bottleneck_channels, cheb_order)
        self.graph_residual = _ChannelAlign(hidden_channels, bottleneck_channels)
        self.temporal_out = _TemporalConv(bottleneck_channels, hidden_channels, temporal_kernel_size, gated=False)
        self.norm = _NodeSharedLayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scaled_laplacian: torch.Tensor) -> torch.Tensor:
        x = self.temporal_in(x)
        x = torch.relu(self.graph(x, scaled_laplacian) + self.graph_residual(x))
        x = self.temporal_out(x)
        return self.dropout(self.norm(x))


class STGCN(ForecastingModel):
    """Forecast all nodes from a supplied undirected graph.

    Input is ``[batch, seq_len, enc_in]``: each feature is one graph node
    with one observed scalar. Output is ``[batch, pred_len, enc_in]`` and
    ``c_out`` must equal ``enc_in``. Adjacency must be exactly symmetric,
    finite and nonnegative; self-loops and isolated nodes are supported.
    Positive weights that underflow to zero in the model's default dtype
    are rejected. The spectral operator is derived from the stored graph,
    so ordinary dtype rounding is shared by config reconstruction.
    The graph is fixed, detached, cloned and stored in persistent buffers.

    ``e_layers`` counts spatio-temporal blocks. Two valid temporal
    convolutions per block require
    ``seq_len > 2 * e_layers * (temporal_kernel_size - 1)``. ``cheb_order``
    counts Chebyshev terms, including the identity term. ``cheb_order=1``
    is a valid graph-independent ablation. ``d_model`` must be at least two:
    channel-only normalization of a single hidden channel removes its signal.

    Decoder values, time markers and attention masks are accepted only for
    the common forecasting API and are unused. Attention/embedding/decoder
    compatibility options do not change the fixed STGCN activation design.
    No future target values are read. The node-shared normalization and
    direct multi-horizon head are framework adaptations, not official
    checkpoint or one-step-rollout parity.
    """

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
        adjacency: Any = None,
        bottleneck_channels: int = 16,
        temporal_kernel_size: int = 3,
        cheb_order: int = 3,
    ):
        super().__init__()
        del n_heads, d_layers, d_ff, embed, freq, activation, output_attention
        for name, value in (
            ("enc_in", enc_in),
            ("dec_in", dec_in),
            ("c_out", c_out),
            ("seq_len", seq_len),
            ("pred_len", pred_len),
            ("d_model", d_model),
            ("e_layers", e_layers),
            ("bottleneck_channels", bottleneck_channels),
            ("temporal_kernel_size", temporal_kernel_size),
            ("cheb_order", cheb_order),
        ):
            _positive_int(name, value)
        if d_model < 2:
            raise ValueError("d_model must be at least 2; single-channel LayerNorm would erase the input signal")
        if isinstance(label_len, bool) or not isinstance(label_len, Integral) or not 0 <= label_len <= seq_len:
            raise ValueError("label_len must be an integer between 0 and seq_len")
        if c_out != enc_in:
            raise ValueError("STGCN preserves graph nodes: c_out must equal enc_in")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1)")
        remaining_steps = seq_len - 2 * e_layers * (temporal_kernel_size - 1)
        if remaining_steps < 1:
            raise ValueError("seq_len must exceed 2 * e_layers * (temporal_kernel_size - 1)")

        graph = _validated_adjacency(adjacency, enc_in)
        default_dtype = torch.get_default_dtype()
        stored_graph = graph.to(dtype=default_dtype)
        if not torch.isfinite(stored_graph).all():
            raise ValueError("adjacency values must be representable in the model's default floating dtype")
        if torch.any((graph > 0) & (stored_graph == 0)):
            raise ValueError(
                "positive adjacency weights must not underflow to zero in the model's default floating dtype"
            )
        # Use one canonical graph for both the serialized config and operator.
        # Evaluate its normalization/eigendecomposition in float64 for stability.
        scaled_laplacian = _scaled_normalized_laplacian(stored_graph.to(dtype=torch.float64))
        self.register_buffer("adjacency", stored_graph)
        self.register_buffer("scaled_laplacian", scaled_laplacian.to(dtype=default_dtype))
        self.enc_in = enc_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.cheb_order = cheb_order
        self.blocks = nn.ModuleList(
            _STConvBlock(
                1 if index == 0 else d_model,
                d_model,
                bottleneck_channels,
                temporal_kernel_size,
                cheb_order,
                float(dropout),
            )
            for index in range(e_layers)
        )
        self.output_temporal = _TemporalConv(d_model, d_model, remaining_steps, gated=True)
        self.output_norm = _NodeSharedLayerNorm(d_model)
        self.readout = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Sigmoid(),
            nn.Conv2d(d_model, pred_len, 1),
        )

    def config(self) -> dict[str, Any]:
        """Return a JSON-compatible graph snapshot independent of constructor aliases."""

        config = super().config()
        config["adjacency"] = self.adjacency.detach().cpu().tolist()
        return config

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        if not isinstance(x_enc, torch.Tensor) or not x_enc.is_floating_point():
            raise TypeError("x_enc must be a real floating-point tensor")
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"STGCN expects x_enc with shape (batch, {self.seq_len}, {self.enc_in})")
        if x_enc.size(0) == 0:
            raise ValueError("x_enc must contain at least one batch item")
        if not torch.isfinite(x_enc).all():
            raise ValueError("x_enc must contain only finite values")
        x = x_enc.transpose(1, 2).unsqueeze(1)
        scaled_laplacian = self.scaled_laplacian.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, scaled_laplacian)
        x = self.output_norm(self.output_temporal(x))
        return self.readout(x).squeeze(-1)
