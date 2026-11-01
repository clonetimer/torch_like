"""Pyraformer: native four-level sparse attention and direct FC forecasting.

Liu et al., ICLR 2022, https://openreview.net/forum?id=0EXmFzUn5I.
Independent mathematical implementation informed by the Apache-2.0 reference
ant-research/Pyraformer@84af4dbd93b7b96975b5034f0dde412005260123.
This portable subset is not an official checkpoint or benchmark reproduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.models.forecasting.pyraformer_layers import _BottleneckCSCM, _pyramid_graph, _PyramidEncoderLayer


@dataclass
class PyraformerConfig(ForecastingConfig):
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    window_size: tuple[int, ...] = (4, 4, 4)
    inner_size: int = 3
    d_bottleneck: int = 16


class Pyraformer(ForecastingModel):
    """Forecast (B,H,C) from observed (B,L,C) with a four-level pyramid.

    Appends one zero raw-value token; circular value convolution, positions and
    time projection embed it. Bottleneck Conv/BN/ELU constructs three coarser
    levels. Sparse attention links same-level neighbors and parents/children.
    Last parents absorb ragged tails although construction convolutions drop
    them. The last fine node and its ancestors feed a bias-free direct FC head.

    ``window_size`` has exactly three integers >=2; ``inner_size`` is positive
    and odd. All coarse lengths must be positive. Default factors require L>=63.
    Training needs B*coarsest_length>1 for BatchNorm; batch-one eval is supported.
    ``c_out`` equals ``enc_in``. No per-window normalization is performed.

    Four continuous marker features are optional (zeros if absent). Encoder
    marks have shape (B,L,4); decoder marks (B,label_len+H,4). Only the first
    future mark at index label_len is used. All decoder values and unused marks
    are ignored; external masks and attention output requests are rejected.
    ``d_layers``, ``embed`` and ``freq`` are compatibility-only fields.

    Float32/float64 input/model dtype and device must match. Used inputs must
    be finite; AMP is disabled internally. Position tables are built at the
    configured length and model default precision, supporting odd widths and
    lengths beyond the reference's 5000-position table. No TVM, dataset IDs,
    attention decoder, alternative CSCM, probabilistic head or pretrained assets.
    """

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
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.05,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        window_size: tuple[int, ...] = (4, 4, 4),
        inner_size: int = 3,
        d_bottleneck: int = 16,
    ):
        super().__init__()
        del d_layers, embed, freq
        for name, value in (
            ("enc_in", enc_in),
            ("dec_in", dec_in),
            ("c_out", c_out),
            ("seq_len", seq_len),
            ("pred_len", pred_len),
            ("d_model", d_model),
            ("n_heads", n_heads),
            ("e_layers", e_layers),
            ("d_ff", d_ff),
            ("d_bottleneck", d_bottleneck),
            ("inner_size", inner_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        # Integral also admits fixed-width NumPy values. Normalize every used
        # architectural integer before arithmetic or layer construction (4*D
        # otherwise wraps to zero for np.int8/uint8 D=64).
        enc_in, dec_in, c_out = int(enc_in), int(dec_in), int(c_out)
        seq_len, pred_len = int(seq_len), int(pred_len)
        d_model, n_heads, e_layers = int(d_model), int(n_heads), int(e_layers)
        d_ff, d_bottleneck, inner_size = int(d_ff), int(d_bottleneck), int(inner_size)
        if d_model < 2 or d_model % n_heads:
            raise ValueError("d_model must be >=2 and divisible by n_heads")
        if isinstance(label_len, bool) or not isinstance(label_len, Integral) or not 0 <= label_len <= seq_len:
            raise ValueError("label_len must be an integer between 0 and seq_len")
        label_len = int(label_len)
        if c_out != enc_in:
            raise ValueError("Pyraformer requires c_out == enc_in")
        if not isinstance(window_size, (list, tuple)) or len(window_size) != 3:
            raise ValueError("window_size must contain exactly three integer factors >=2")
        if any(isinstance(f, bool) or not isinstance(f, Integral) or f < 2 for f in window_size):
            raise ValueError("window_size must contain exactly three integer factors >=2")
        # Integral includes fixed-width NumPy scalars; do geometry arithmetic
        # with Python integers so overflow cannot bypass the nonempty guard.
        factors = tuple(int(f) for f in window_size)
        if int(seq_len) + 1 < math.prod(factors):
            raise ValueError("seq_len + 1 must leave all coarse pyramid levels nonempty")
        if inner_size % 2 == 0:
            raise ValueError("inner_size must be odd")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0,1)")
        if activation != "gelu":
            raise ValueError("activation must be gelu for this Pyraformer subset")
        if output_attention is not False:
            raise ValueError("output_attention must be False; dense attention outputs are not supported")

        self.enc_in, self.c_out = int(enc_in), int(c_out)
        self.seq_len, self.label_len, self.pred_len = int(seq_len), int(label_len), int(pred_len)
        self.window_size = factors
        self.level_lengths, neighbors, valid, last = _pyramid_graph(self.seq_len + 1, self.window_size, int(inner_size))
        self.register_buffer("neighbors", neighbors)
        self.register_buffer("neighbor_valid", valid)
        self.register_buffer("last_ancestors", last)
        self.value_embedding = nn.Conv1d(enc_in, d_model, 3, padding=1, padding_mode="circular")
        nn.init.kaiming_normal_(self.value_embedding.weight)
        self.time_embedding = nn.Linear(4, d_model)
        positions = torch.arange(self.seq_len + 1, dtype=torch.float64).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float64) * (-math.log(10000.0) / d_model))
        angles = positions * frequencies
        table = torch.empty(self.seq_len + 1, d_model, dtype=torch.float64)
        table[:, 0::2] = angles.sin()
        table[:, 1::2] = angles[:, : d_model // 2].cos()
        self.register_buffer("position", table.to(dtype=torch.get_default_dtype()).unsqueeze(0))
        self.embedding_dropout = nn.Dropout(dropout)
        self.cscm = _BottleneckCSCM(d_model, d_bottleneck, self.window_size)
        self.encoder = nn.ModuleList([_PyramidEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.head = nn.Linear(4 * d_model, self.pred_len * self.c_out, bias=False)
        nn.init.xavier_normal_(self.head.weight)

    def config(self) -> dict:
        config = super().config()
        config["window_size"] = list(self.window_size)
        return config

    def _validate_observed(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Pyraformer expects a torch.Tensor input")
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Pyraformer expects shape (batch, {self.seq_len}, {self.enc_in})")
        if x.shape[0] == 0:
            raise ValueError("Pyraformer requires a non-empty batch")
        if x.dtype not in (torch.float32, torch.float64) or x.dtype != self.value_embedding.weight.dtype:
            raise TypeError("Pyraformer input/model dtype must match and be float32 or float64")
        if x.device != self.value_embedding.weight.device:
            raise ValueError("Pyraformer input/model device must match")
        if not torch.isfinite(x).all():
            raise ValueError("Pyraformer observations must be finite")

    def _markers(self, markers: torch.Tensor | None, x: torch.Tensor, *, future: bool) -> torch.Tensor:
        length = self.label_len + self.pred_len if future else self.seq_len
        name = "x_mark_dec" if future else "x_mark_enc"
        if markers is None:
            return x.new_zeros(x.shape[0], 1 if future else length, 4)
        if not isinstance(markers, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if markers.shape != (x.shape[0], length, 4):
            raise ValueError(f"{name} must have shape (batch, {length}, 4)")
        if markers.dtype != x.dtype:
            raise TypeError(f"{name} dtype must match observations")
        if markers.device != x.device:
            raise ValueError(f"{name} device must match observations")
        used = markers[:, self.label_len : self.label_len + 1] if future else markers
        if not torch.isfinite(used).all():
            raise ValueError(f"used {name} values must be finite")
        return used

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
        del x_dec
        if any(mask is not None for mask in (enc_self_mask, dec_self_mask, dec_enc_mask)):
            raise ValueError("Pyraformer does not support external masks")
        self._validate_observed(x_enc)
        history_marks = self._markers(x_mark_enc, x_enc, future=False)
        prediction_mark = self._markers(x_mark_dec, x_enc, future=True)
        with torch.autocast(device_type=x_enc.device.type, enabled=False):
            values = torch.cat((x_enc, x_enc.new_zeros(x_enc.shape[0], 1, self.enc_in)), dim=1)
            marks = torch.cat((history_marks, prediction_mark), dim=1)
            embedded = self.value_embedding(values.transpose(1, 2)).transpose(1, 2)
            embedded = self.embedding_dropout(embedded + self.position + self.time_embedding(marks))
            state = self.cscm(embedded)
            for layer in self.encoder:
                state = layer(state, self.neighbors, self.neighbor_valid)
            selected = state[:, self.last_ancestors, :].flatten(1)
            return self.head(selected).reshape(x_enc.shape[0], self.pred_len, self.c_out)
