"""Native univariate DeepState with explicit covariates and Gaussian state inference.

Reference: Rangapuram et al., NeurIPS 2018; GluonTS DeepState/ISSM/LDS at
d60556ca71167a3ec1ea807b491dd982d1de2112 (Apache-2.0). This portable subset adapts
the objective, scaling, masking, dtype and calendar interfaces; it does not load
GluonTS checkpoints or reproduce its complete estimator/training pipeline.

Gluon Time Series
Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch
from torch import nn

from torch_like.models.forecasting.common import (
    ProbabilisticForecastingBatch,
    ProbabilisticForecastingConfig,
    ProbabilisticForecastingModel,
)
from torch_like.models.forecasting.deepstate_layers import (
    _ISSM,
    _kalman_filter,
    _marginals,
    _ResidualLSTM,
    _sample_paths,
)


@dataclass
class DeepStateConfig(ProbabilisticForecastingConfig):
    """Univariate Gaussian structural model; periods and categories are explicit."""

    d_model: int = 40
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0
    time_feature_dim: int = 4
    seasonal_periods: tuple[int, ...] = ()
    add_trend: bool = True
    cardinalities: tuple[int, ...] = ()
    embedding_dimensions: tuple[int, ...] = ()
    scaling: bool = True
    minimum_scale: float = 1e-10
    noise_std_bounds: tuple[float, float] = (1e-6, 1.0)
    innovation_bounds: tuple[float, float] = (1e-6, 0.01)
    prior_cov_bounds: tuple[float, float] = (1e-6, 1.0)
    dtype: str | torch.dtype | None = None


@dataclass
class DeepStateBatch(ProbabilisticForecastingBatch):
    """Explicit masks, seasonal phase indices and optional static category IDs.

    ``x_dec`` is optional and ignored. The final pred_len decoder covariate and
    seasonal rows describe the forecast horizon; their prefix is ignored.
    Missing values need explicit bool masks; omitted masks mean all observed.
    """

    x_mark_enc: torch.Tensor | None = None
    x_dec: torch.Tensor | None = None
    x_mark_dec: torch.Tensor | None = None
    observed_enc: torch.Tensor | None = None
    observed_dec: torch.Tensor | None = None
    seasonal_indices_enc: torch.Tensor | None = None
    seasonal_indices_dec: torch.Tensor | None = None
    feat_static_cat: torch.Tensor | None = None


def _integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _integer_list(name: str, values, minimum: int = 1) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of integers")
    return tuple(_integer(name, value, minimum) for value in values)


def _dtype(value) -> torch.dtype:
    if value is None:
        value = torch.get_default_dtype()
    if isinstance(value, (str, torch.dtype)):
        if value in ("float32", torch.float32):
            return torch.float32
        if value in ("float64", torch.float64):
            return torch.float64
    raise ValueError("dtype must select float32 or float64")


def _positive_real(name: str, value: float, dtype: torch.dtype) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real number")
    stored = torch.tensor(float(value), dtype=dtype)
    if not torch.isfinite(stored) or stored <= 0:
        raise ValueError(f"{name} must be representable in the model dtype")
    return float(value)


def _bounds(name: str, values, dtype: torch.dtype, squared: bool = False) -> tuple[float, float]:
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"{name} must contain two positive ordered bounds")
    lower, upper = (_positive_real(name, value, dtype) for value in values)
    if lower > upper:
        raise ValueError(f"{name} bounds must be ordered")
    if squared:
        stored = torch.tensor((lower, upper), dtype=dtype).square()
        if not torch.isfinite(stored).all() or (stored <= 0).any():
            raise ValueError(f"squared {name} must be positive and representable in the model dtype")
    return lower, upper


def _bounded(value: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    lower, upper = bounds
    return lower + (upper - lower) * value.sigmoid()


class DeepState(ProbabilisticForecastingModel):
    """Covariate-driven residual LSTM plus level/trend/seasonal Kalman inference.

    Each row is an independent univariate series. Known real covariates and
    static categories condition the network; target values and their scale do
    not enter it. Seasonal indices are caller-supplied phases, not inferred
    dates. Rank-one process noise is shared across structural components.

    ``predict_distribution`` exposes Gaussian marginal standard deviations;
    ``sample`` preserves cross-time dependence through a shared evolving latent
    state. ``fit_step`` uses conditional future joint NLL per observed label in
    original units. ``context_nll`` averages context NLL per nonempty series.
    Mean-absolute scaling uses only observed history, with scale1 for zero or
    missing-only rows. These objectives/scaling rules are explicit adaptations.

    Float32/64 dense model inputs must match model dtype/device. AMP is disabled;
    Gaussian algebra accumulates in float64 for numerical robustness, increasing
    compute/storage cost. Public outputs retain model dtype. Extreme magnitudes
    can still exceed numerical range and are rejected with a rescaling message.
    No posterior/session state persists between calls. n_heads/d_layers/d_ff are
    accepted compatibility fields; this recurrent model does not use them.
    Only dropout0 is supported; reference zoneout is not inter-layer dropout.
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 40,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.0,
        distribution: str = "gaussian",
        num_samples: int = 100,
        time_feature_dim: int = 4,
        seasonal_periods: tuple[int, ...] = (),
        add_trend: bool = True,
        cardinalities: tuple[int, ...] = (),
        embedding_dimensions: tuple[int, ...] = (),
        scaling: bool = True,
        minimum_scale: float = 1e-10,
        noise_std_bounds: tuple[float, float] = (1e-6, 1.0),
        innovation_bounds: tuple[float, float] = (1e-6, 0.01),
        prior_cov_bounds: tuple[float, float] = (1e-6, 1.0),
        dtype: str | torch.dtype | None = None,
    ):
        super().__init__()
        construction_dtype = _dtype(dtype)
        for name, value in (
            ("enc_in", enc_in),
            ("dec_in", dec_in),
            ("c_out", c_out),
            ("seq_len", seq_len),
            ("pred_len", pred_len),
            ("d_model", d_model),
            ("n_heads", n_heads),
            ("e_layers", e_layers),
            ("d_layers", d_layers),
            ("d_ff", d_ff),
            ("num_samples", num_samples),
        ):
            setattr(self, name, _integer(name, value))
        if self.enc_in != 1 or self.dec_in != 1 or self.c_out != 1:
            raise ValueError("DeepState requires enc_in == dec_in == c_out == 1 (independent univariate series)")
        self.label_len = _integer("label_len", label_len, 0)
        if self.label_len > self.seq_len:
            raise ValueError("label_len must be <= seq_len")
        self.time_feature_dim = _integer("time_feature_dim", time_feature_dim, 0)
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or dropout != 0:
            raise ValueError("DeepState supports dropout=0 only; inter-layer dropout is not reference zoneout")
        if distribution != "gaussian":
            raise ValueError("DeepState supports distribution='gaussian' only")
        if not isinstance(add_trend, bool) or not isinstance(scaling, bool):
            raise ValueError("add_trend and scaling must be bool")
        self.dropout, self.distribution, self.add_trend, self.scaling = 0.0, distribution, add_trend, scaling
        self.seasonal_periods = _integer_list("seasonal_periods", seasonal_periods, 2)
        self.cardinalities = _integer_list("cardinalities", cardinalities)
        self.embedding_dimensions = _integer_list("embedding_dimensions", embedding_dimensions)
        if len(self.cardinalities) != len(self.embedding_dimensions):
            raise ValueError("cardinalities and embedding_dimensions must have equal lengths")
        self.minimum_scale = _positive_real("minimum_scale", minimum_scale, construction_dtype)
        self.noise_std_bounds = _bounds("noise_std_bounds", noise_std_bounds, construction_dtype, squared=True)
        self.innovation_bounds = _bounds("innovation_bounds", innovation_bounds, construction_dtype, squared=True)
        self.prior_cov_bounds = _bounds("prior_cov_bounds", prior_cov_bounds, construction_dtype)
        self.structure = _ISSM(self.seasonal_periods, add_trend, construction_dtype)
        self.embeddings = nn.ModuleList(
            nn.Embedding(count, width)
            for count, width in zip(self.cardinalities, self.embedding_dimensions, strict=True)
        )
        input_size = max(1, self.time_feature_dim + sum(self.embedding_dimensions))
        self.rnn = _ResidualLSTM(input_size, self.d_model, self.e_layers)
        self.prior_mean_head = nn.Linear(self.d_model, self.structure.state_dim)
        self.prior_cov_head = nn.Linear(self.d_model, self.structure.state_dim)
        self.noise_head = nn.Linear(self.d_model, 1)
        self.innovation_head = nn.Linear(self.d_model, 1)
        self.residual_head = nn.Linear(self.d_model, 1)
        self.to(dtype=construction_dtype)

    def config(self) -> dict[str, Any]:
        """Owned JSON-compatible values, with current parameter/buffer precision."""
        result = super().config()
        for name in tuple(result):
            if name != "dtype":
                value = getattr(self, name)
                result[name] = list(value) if isinstance(value, tuple) else value
        result["dtype"] = str(self.structure.transition.dtype).removeprefix("torch.")
        return result

    def _tensor(self, value, name: str, shape: tuple[int, ...], kind: str = "float") -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} is required as a torch.Tensor")
        if value.layout != torch.strided or value.is_nested or value.device.type == "meta":
            raise ValueError(f"{name} must use a dense materialized strided tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        reference = self.structure.transition
        if reference.dtype not in (torch.float32, torch.float64):
            raise TypeError("DeepState model dtype must be float32 or float64")
        if value.device != reference.device:
            raise ValueError(f"{name} device must match the model")
        if kind == "float":
            if value.dtype != reference.dtype:
                raise TypeError(f"{name} dtype must match the float32/float64 model")
        elif kind == "bool":
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must be a bool tensor")
        elif value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"{name} must be an integer tensor (not bool)")
        return value

    @staticmethod
    def _finite(value: torch.Tensor, name: str) -> None:
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite at all active/observed positions")

    def _observations(self, values, observed, name: str, shape: tuple[int, ...]):
        values = self._tensor(values, name, shape)
        mask_name = "observed_enc" if name == "x_enc" else "observed_dec"
        if observed is None:
            observed = torch.ones_like(values, dtype=torch.bool)
        else:
            observed = self._tensor(observed, mask_name, shape, "bool")
        clean = torch.where(observed, values, torch.zeros_like(values))
        self._finite(clean, name)
        return clean, observed

    def _known_inputs(self, batch, b: int, future: bool = False):
        suffix = "dec" if future else "enc"
        length = self.label_len + self.pred_len if future else self.seq_len
        active = self.pred_len if future else self.seq_len
        features = getattr(batch, f"x_mark_{suffix}", None)
        if self.time_feature_dim or features is not None:
            features = self._tensor(features, f"x_mark_{suffix}", (b, length, self.time_feature_dim))
            features = features[:, -active:]
            self._finite(features, f"x_mark_{suffix}")
        indices = getattr(batch, f"seasonal_indices_{suffix}", None)
        if self.seasonal_periods or indices is not None:
            indices = self._tensor(
                indices, f"seasonal_indices_{suffix}", (b, length, len(self.seasonal_periods)), "int"
            )
            indices = indices[:, -active:]
            for column, period in enumerate(self.seasonal_periods):
                if (indices[..., column] < 0).any() or (indices[..., column].long() >= period).any():
                    raise ValueError(f"seasonal_indices_{suffix} column {column} must be in [0,{period})")
        return features, indices

    def _categories(self, batch, b: int):
        categories = getattr(batch, "feat_static_cat", None)
        if self.cardinalities or categories is not None:
            categories = self._tensor(categories, "feat_static_cat", (b, len(self.cardinalities)), "int")
            for column, cardinality in enumerate(self.cardinalities):
                if (categories[:, column] < 0).any() or (categories[:, column].long() >= cardinality).any():
                    raise ValueError(f"feat_static_cat column {column} must be in [0,{cardinality})")
        return categories

    def _rnn_input(self, features, categories, b: int, length: int):
        parts = [] if features is None else [features]
        if self.cardinalities:
            static = torch.cat(
                [embedding(categories[:, column].long()) for column, embedding in enumerate(self.embeddings)], -1
            )
            parts.append(static[:, None].expand(-1, length, -1))
        if not parts or sum(value.shape[-1] for value in parts) == 0:
            return self.structure.transition.new_ones(b, length, 1)
        return torch.cat(parts, -1)

    def _sequence_params(self, output, indices):
        return dict(
            emission=self.structure.emission(indices, output),
            transition=self.structure.transition,
            innovation=_bounded(self.innovation_head(output), self.innovation_bounds),
            noise_std=_bounded(self.noise_head(output), self.noise_std_bounds),
            residual=self.residual_head(output),
        )

    def _validate_batch(self, batch) -> int:
        if not isinstance(batch, ProbabilisticForecastingBatch):
            raise TypeError("DeepState expects DeepStateBatch or ProbabilisticForecastingBatch")
        for name in (
            "x_target_enc",
            "x_exog_enc",
            "x_target_dec",
            "x_exog_dec",
            "enc_self_mask",
            "dec_self_mask",
            "dec_enc_mask",
        ):
            value = getattr(batch, name, None)
            if value is not None and (not isinstance(value, torch.Tensor) or value.numel()):
                raise ValueError(f"DeepState does not support {name}; use explicit DeepStateBatch fields")
        x = batch.x_enc
        if not isinstance(x, torch.Tensor) or x.is_nested or x.ndim != 3 or x.shape[0] == 0:
            raise ValueError("x_enc must be a nonempty tensor of shape [batch,seq_len,1]")
        return x.shape[0]

    def _context(self, batch, *, future: bool):
        b = self._validate_batch(batch)
        clean, observed = self._observations(
            batch.x_enc, getattr(batch, "observed_enc", None), "x_enc", (b, self.seq_len, 1)
        )
        features, indices = self._known_inputs(batch, b)
        categories = self._categories(batch, b)
        future_known = None
        if future:
            future_known = self._known_inputs(batch, b, future=True)
            if batch.x_dec is not None:
                self._tensor(batch.x_dec, "x_dec", (b, self.label_len + self.pred_len, 1))
        # Double reduction prevents overflow in the sum of finite float32 values.
        clean = clean.double()
        if self.scaling:
            count = observed.sum(1, keepdim=True).clamp_min(1)
            absolute_mean = clean.abs().sum(1, keepdim=True) / count
            scale = torch.where(
                absolute_mean > 0, absolute_mean.clamp_min(self.minimum_scale), torch.ones_like(absolute_mean)
            )
        else:
            scale = clean.new_ones(b, 1, 1)
        self._finite(scale, "context scale; rescale inputs")
        output, states = self.rnn(self._rnn_input(features, categories, b, self.seq_len))
        mean = self.prior_mean_head(output[:, 0])
        cov = torch.diag_embed(_bounded(self.prior_cov_head(output[:, 0]), self.prior_cov_bounds))
        params = self._sequence_params(output, indices)
        log_prob, mean, cov = _kalman_filter(clean / scale, observed, mean, cov, **params)
        return dict(
            log_prob=log_prob,
            observed=observed,
            mean=mean,
            cov=cov,
            scale=scale,
            states=states,
            categories=categories,
            future_known=future_known,
            batch_size=b,
        )

    def _future_params(self, context):
        features, indices = context["future_known"]
        inputs = self._rnn_input(features, context["categories"], context["batch_size"], self.pred_len)
        output, _ = self.rnn(inputs, context["states"])
        return self._sequence_params(output, indices)

    def _public(self, value: torch.Tensor, *, positive: bool = False) -> torch.Tensor:
        value = value.to(dtype=self.structure.transition.dtype)
        if not torch.isfinite(value).all():
            raise ValueError("DeepState result exceeds numerical range; rescale inputs or use float64")
        if positive and (value <= 0).any():
            raise ValueError(
                "DeepState predictive scale must stay positive in the model dtype; rescale inputs or use float64"
            )
        return value

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
        *,
        observed_enc: torch.Tensor | None = None,
        seasonal_indices_enc: torch.Tensor | None = None,
        seasonal_indices_dec: torch.Tensor | None = None,
        feat_static_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = DeepStateBatch(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_self_mask=enc_self_mask,
            dec_self_mask=dec_self_mask,
            dec_enc_mask=dec_enc_mask,
            observed_enc=observed_enc,
            seasonal_indices_enc=seasonal_indices_enc,
            seasonal_indices_dec=seasonal_indices_dec,
            feat_static_cat=feat_static_cat,
        )
        return self.predict(batch)

    def predict(self, batch: DeepStateBatch) -> torch.Tensor:
        return self.predict_distribution(batch)["mean"]

    def predict_distribution(self, batch: DeepStateBatch) -> dict[str, Any]:
        with torch.autocast(device_type=self.structure.transition.device.type, enabled=False):
            context = self._context(batch, future=True)
            mean, variance = _marginals(context["mean"], context["cov"], **self._future_params(context))
            return dict(
                mean=self._public(mean * context["scale"]),
                scale=self._public(variance.sqrt() * context["scale"], positive=True),
                distribution="gaussian",
            )

    def sample(self, batch: DeepStateBatch, num_samples: int | None = None) -> torch.Tensor:
        count = self.num_samples if num_samples is None else _integer("num_samples", num_samples)
        with torch.autocast(device_type=self.structure.transition.device.type, enabled=False):
            context = self._context(batch, future=True)
            paths = _sample_paths(context["mean"], context["cov"], **self._future_params(context), num_samples=count)
            return self._public(paths * context["scale"].unsqueeze(0))

    def fit_step(self, batch: DeepStateBatch) -> torch.Tensor:
        with torch.autocast(device_type=self.structure.transition.device.type, enabled=False):
            context = self._context(batch, future=True)
            clean, observed = self._observations(
                batch.y_true, getattr(batch, "observed_dec", None), "y_true", (context["batch_size"], self.pred_len, 1)
            )
            count = observed.sum()
            if not count:
                raise ValueError("fit_step requires at least one observed future label")
            log_prob, _, _ = _kalman_filter(
                clean.double() / context["scale"],
                observed,
                context["mean"],
                context["cov"],
                **self._future_params(context),
            )
            jacobian = context["scale"].log().squeeze(-1)
            nll = torch.where(observed[..., 0], jacobian - log_prob, torch.zeros_like(log_prob))
            return self._public(nll.sum() / count)

    def context_nll(self, batch: DeepStateBatch) -> torch.Tensor:
        """Reference-style context objective, averaging only nonempty series."""
        with torch.autocast(device_type=self.structure.transition.device.type, enabled=False):
            context = self._context(batch, future=False)
            observed = context["observed"][..., 0]
            counts = observed.sum(1)
            nonempty = counts > 0
            if not nonempty.any():
                raise ValueError("context_nll requires at least one observed context value")
            jacobian = context["scale"].log().squeeze(-1)
            nll = torch.where(observed, jacobian - context["log_prob"], torch.zeros_like(context["log_prob"]))
            return self._public((nll.sum(1) / counts.clamp_min(1))[nonempty].mean())

    def make_probabilistic_forecast_batch(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        y_true: torch.Tensor | None = None,
        x_target_enc: torch.Tensor | None = None,
        x_exog_enc: torch.Tensor | None = None,
        x_target_dec: torch.Tensor | None = None,
        x_exog_dec: torch.Tensor | None = None,
        *,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
        observed_enc: torch.Tensor | None = None,
        observed_dec: torch.Tensor | None = None,
        seasonal_indices_enc: torch.Tensor | None = None,
        seasonal_indices_dec: torch.Tensor | None = None,
        feat_static_cat: torch.Tensor | None = None,
    ) -> DeepStateBatch:
        """Preserve explicit inputs without fabricating covariates/calendars/masks."""
        return DeepStateBatch(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            y_true,
            x_target_enc,
            x_exog_enc,
            x_target_dec,
            x_exog_dec,
            enc_self_mask,
            dec_self_mask,
            dec_enc_mask,
            observed_enc,
            observed_dec,
            seasonal_indices_enc,
            seasonal_indices_dec,
            feat_static_cat,
        )
