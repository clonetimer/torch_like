"""GMAN with supplied spatial embeddings and explicit historical/future calendar.

Zheng, Fan, Wang and Qi, AAAI 2020: https://arxiv.org/abs/1911.08415v2.
Apache-2.0 mathematical reference: zhengchuanpan/GMAN at
45ed232f4ee6a7ee9dd248d07f4952b697ecf320. This independent native subset is
not a TensorFlow checkpoint or training-pipeline reproduction. No spatial
assets, node2vec implementation, pretrained weights or datasets are bundled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral, Real
from typing import Any

import torch
from torch import nn

from torch_like.models.forecasting.common import ForecastingBatch, ForecastingConfig, ForecastingModel
from torch_like.models.forecasting.gman_layers import _mlp, _STAttentionBlock, _STEmbedding, _TransformAttention


@dataclass
class GMANConfig(ForecastingConfig):
    """PeMS-like architectural defaults; spatial_embedding and calendar are required."""

    d_model: int = 64
    n_heads: int = 8
    e_layers: int = 1
    d_layers: int = 1
    d_ff: int = 64
    dropout: float = 0.0
    activation: str = "relu"
    spatial_embedding: Any = None
    time_slots_per_day: int = 288
    temporal_mask: bool = False
    use_batch_norm: bool = True
    bn_decay: float = 0.9
    dtype: str | torch.dtype | None = None


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must be a finite real number in [0,1)")
    return float(value)


def _floating_dtype(value: str | torch.dtype | None) -> torch.dtype:
    if value is None:
        value = torch.get_default_dtype()
    if isinstance(value, (str, torch.dtype)):
        if value in ("float32", torch.float32):
            return torch.float32
        if value in ("float64", torch.float64):
            return torch.float64
    raise ValueError("dtype must select float32 or float64, or None for a supported process default")


def _owned_spatial_embedding(value: Any, nodes: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    if value is None:
        raise ValueError("GMAN requires an explicit spatial_embedding; no node embeddings are synthesized")
    try:
        probe = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError("spatial_embedding must be a dense real numeric matrix") from exc
    if probe.layout != torch.strided or probe.is_complex() or probe.dtype == torch.bool:
        raise ValueError("spatial_embedding must be a dense real non-boolean numeric matrix")
    if probe.shape != (nodes, width):
        raise ValueError(f"spatial_embedding must have shape ({nodes}, {width})")
    try:
        # Convert the original input, not a float32 probe that may have rounded
        # Python-list values to zero before representability checks.
        original = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError("spatial_embedding must be a materialized real numeric matrix") from exc
    if not torch.isfinite(original).all():
        raise ValueError("spatial_embedding must contain only finite values")
    stored = original.to(dtype=dtype)
    if not torch.isfinite(stored).all():
        raise ValueError("spatial_embedding must be representable in the model's construction dtype")
    if torch.any((original != 0) & (stored == 0)):
        raise ValueError("nonzero spatial_embedding values must not underflow in the model's construction dtype")
    return stored


class GMAN(ForecastingModel):
    """Dense multi-attention forecasting with an explicit calendar-query bridge.

    Each feature of x_enc[B,seq_len,enc_in] is one node's scalar traffic value;
    c_out must equal enc_in. Supplied fixed spatial_embedding[enc_in,d_model]
    rows MUST follow the same node order. They are cloned, detached and persisted.
    The caller owns embedding provenance and train-only input/target scaling.

    Both calendars are mandatory raw integer [weekday,slot] IDs: x_mark_enc
    has shape[B,seq_len,2], x_mark_dec[B,label_len+pred_len,2]. Weekday is0..6,
    slot is0..time_slots_per_day-1. The decoder prefix and ALL decoder traffic
    values are ignored. Future calendar is known covariate input, not a target.
    No timezone, cadence, continuity or node-order inference is performed.

    e_layers counts blocks in BOTH encoder and decoder. Defaults maskFalse,
    dropout0,L1 are PeMS-like; maskTrue,dropout.1,L5 select METR architecture
    settings. Dropout occurs only before both output affine layers. d_layers,
    d_ff,embed,freq are compatibility fields, unused by this architecture.

    Population BN uses eps1e-3, fixed decay and initial variance1 (the latter
    differs from TF tensor-EMA zero). Training BN mixes times, so temporal_mask
    does not promise online prefix causality. Singleton reductions are legal.
    Float32/64 only, model/input dtype and device must match; AMP is disabled.
    dtype optionally selects float32/float64 without changing the process
    default. Config records the current dtype so float64-only SE coordinates
    survive reconstruction in a float32-default process. from_config extracts
    dataclass fields without deep-copying computed tensors before detachment.
    Use scaled inputs: finiteness checks do not guarantee stability at extreme
    magnitudes or roll back BN statistics after overflow. Spatial attention is
    dense, and graph-derived offline embeddings do not make the pipeline graph-free.
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
        n_heads: int = 8,
        e_layers: int = 1,
        d_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.0,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "relu",
        output_attention: bool = False,
        spatial_embedding: Any = None,
        time_slots_per_day: int = 288,
        temporal_mask: bool = False,
        use_batch_norm: bool = True,
        bn_decay: float = 0.9,
        dtype: str | torch.dtype | None = None,
    ):
        super().__init__()
        del d_layers, d_ff, embed, freq
        for name, value in (
            ("enc_in", enc_in),
            ("dec_in", dec_in),
            ("c_out", c_out),
            ("seq_len", seq_len),
            ("pred_len", pred_len),
            ("d_model", d_model),
            ("n_heads", n_heads),
            ("e_layers", e_layers),
            ("time_slots_per_day", time_slots_per_day),
        ):
            setattr(self, name, _positive_int(name, value))
        if isinstance(label_len, bool) or not isinstance(label_len, Integral) or not 0 <= label_len <= self.seq_len:
            raise ValueError("label_len must be an integer between 0 and seq_len")
        self.label_len = int(label_len)
        if self.c_out != self.enc_in:
            raise ValueError("GMAN preserves nodes: c_out must equal enc_in")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        for name, value in (("temporal_mask", temporal_mask), ("use_batch_norm", use_batch_norm)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        if activation != "relu":
            raise ValueError("activation must be relu for this GMAN subset")
        if output_attention is not False:
            raise ValueError("output_attention must be False; attention matrices are not public outputs")
        self.dropout = _probability("dropout", dropout)
        self.bn_decay = _probability("bn_decay", bn_decay)
        self.temporal_mask = temporal_mask
        self.use_batch_norm = use_batch_norm
        construction_dtype = _floating_dtype(dtype)
        self.register_buffer(
            "spatial_embedding",
            _owned_spatial_embedding(spatial_embedding, self.enc_in, self.d_model, construction_dtype),
        )
        opts = dict(batch_norm=use_batch_norm, bn_decay=self.bn_decay)
        width = self.d_model
        self.input_projection = _mlp(1, width, width, **opts)
        self.embedding = _STEmbedding(width, self.time_slots_per_day, **opts)
        self.encoder = nn.ModuleList(
            _STAttentionBlock(width, self.n_heads, causal=temporal_mask, **opts) for _ in range(self.e_layers)
        )
        self.transform = _TransformAttention(width, self.n_heads, **opts)
        self.decoder = nn.ModuleList(
            _STAttentionBlock(width, self.n_heads, causal=temporal_mask, **opts) for _ in range(self.e_layers)
        )
        self.readout = _mlp(width, width, 1, dropout=self.dropout, **opts)
        # No process-global dtype mutation. Ordinary initialized parameters are
        # converted just as by model.double(); persisted values are loaded later.
        self.to(dtype=construction_dtype)

    @classmethod
    def from_config(cls, config):
        """Let owned-input validation detach tensors before any recursive copying."""
        if is_dataclass(config) and not isinstance(config, type):
            config = {field.name: getattr(config, field.name) for field in fields(config)}
        return super().from_config(config)

    def config(self) -> dict[str, Any]:
        """Return the current owned embedding and canonical used numeric scalars."""
        config = super().config()
        config["spatial_embedding"] = self.spatial_embedding.detach().cpu().tolist()
        config["dtype"] = str(self.spatial_embedding.dtype).removeprefix("torch.")
        for name in (
            "enc_in",
            "dec_in",
            "c_out",
            "seq_len",
            "label_len",
            "pred_len",
            "d_model",
            "n_heads",
            "e_layers",
            "time_slots_per_day",
            "dropout",
            "bn_decay",
        ):
            config[name] = getattr(self, name)
        return config

    def _validate_observed(self, x_enc: torch.Tensor) -> None:
        if not isinstance(x_enc, torch.Tensor):
            raise TypeError("x_enc must be a torch.Tensor")
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"GMAN expects x_enc shape (batch, {self.seq_len}, {self.enc_in})")
        if x_enc.shape[0] == 0:
            raise ValueError("x_enc must contain a non-empty batch")
        if x_enc.dtype not in (torch.float32, torch.float64) or x_enc.dtype != self.spatial_embedding.dtype:
            raise TypeError("GMAN input/model dtype must match and be float32 or float64")
        if x_enc.device != self.spatial_embedding.device:
            raise ValueError("GMAN input/model device must match")
        if not torch.isfinite(x_enc).all():
            raise ValueError("GMAN observations must be finite")

    def _calendar(self, marks: torch.Tensor | None, x: torch.Tensor, *, future: bool) -> torch.Tensor:
        name = "x_mark_dec" if future else "x_mark_enc"
        length = self.label_len + self.pred_len if future else self.seq_len
        if marks is None:
            raise ValueError(f"{name} is required: supply raw integer [weekday,slot] calendar")
        if not isinstance(marks, torch.Tensor):
            raise TypeError(f"{name} calendar must be a torch.Tensor")
        if marks.shape != (x.shape[0], length, 2):
            raise ValueError(f"{name} must have shape (batch, {length}, 2), ordered [weekday,slot]")
        if marks.device != x.device:
            raise ValueError(f"{name} calendar device must match x_enc")
        if marks.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"{name} calendar must use integer IDs, not bool or normalized floating features")
        used = marks[:, self.label_len :] if future else marks
        # Promote BEFORE comparing with T: int8/uint8 scalar bounds can wrap.
        used = used.to(dtype=torch.long)
        if torch.any((used[..., 0] < 0) | (used[..., 0] > 6)):
            raise ValueError(f"{name} used calendar weekdays must be in [0,6]")
        if torch.any((used[..., 1] < 0) | (used[..., 1] >= self.time_slots_per_day)):
            raise ValueError(f"{name} used calendar slots must be in [0,time_slots_per_day)")
        return used

    def make_forecast_batch(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        x_target_enc: torch.Tensor | None = None,
        x_exog_enc: torch.Tensor | None = None,
        x_target_dec: torch.Tensor | None = None,
        x_exog_dec: torch.Tensor | None = None,
    ) -> ForecastingBatch:
        self._validate_observed(x_enc)
        self._calendar(x_mark_enc, x_enc, future=False)
        self._calendar(x_mark_dec, x_enc, future=True)
        return super().make_forecast_batch(
            x_enc, x_mark_enc, x_dec, x_mark_dec, x_target_enc, x_exog_enc, x_target_dec, x_exog_dec
        )

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
            raise ValueError("GMAN does not support external masks; use temporal_mask")
        self._validate_observed(x_enc)
        past = self._calendar(x_mark_enc, x_enc, future=False)
        future = self._calendar(x_mark_dec, x_enc, future=True)
        with torch.autocast(device_type=x_enc.device.type, enabled=False):
            ste = self.embedding(self.spatial_embedding, torch.cat((past, future), dim=1))
            ste_history, ste_future = ste[:, : self.seq_len], ste[:, self.seq_len :]
            hidden = self.input_projection(x_enc[..., None])
            for block in self.encoder:
                hidden = block(hidden, ste_history)
            hidden = self.transform(hidden, ste_history, ste_future)
            for block in self.decoder:
                hidden = block(hidden, ste_future)
            return self.readout(hidden).squeeze(-1)
