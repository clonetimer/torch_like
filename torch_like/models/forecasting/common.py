"""Common forecasting configs and model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torch_like.models.base import BaseModel


@dataclass
class ForecastingConfig:
    enc_in: int
    dec_in: int
    c_out: int
    seq_len: int
    label_len: int
    pred_len: int
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.05
    embed: str = "fixed"
    freq: str = "h"
    activation: str = "gelu"
    output_attention: bool = False


@dataclass
class ForecastingBatch:
    x_enc: torch.Tensor
    x_mark_enc: Optional[torch.Tensor]
    x_dec: torch.Tensor
    x_mark_dec: Optional[torch.Tensor]
    x_target_enc: Optional[torch.Tensor] = None
    x_exog_enc: Optional[torch.Tensor] = None
    x_target_dec: Optional[torch.Tensor] = None
    x_exog_dec: Optional[torch.Tensor] = None
    enc_self_mask: Optional[torch.Tensor] = None
    dec_self_mask: Optional[torch.Tensor] = None
    dec_enc_mask: Optional[torch.Tensor] = None


@dataclass
class ProbabilisticForecastingConfig:
    enc_in: int
    dec_in: int
    c_out: int
    seq_len: int
    label_len: int
    pred_len: int
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.05
    distribution: str = "gaussian"
    num_samples: int = 100


@dataclass
class ProbabilisticForecastingBatch:
    x_enc: torch.Tensor
    x_mark_enc: Optional[torch.Tensor]
    x_dec: torch.Tensor
    x_mark_dec: Optional[torch.Tensor]
    y_true: Optional[torch.Tensor] = None
    x_target_enc: Optional[torch.Tensor] = None
    x_exog_enc: Optional[torch.Tensor] = None
    x_target_dec: Optional[torch.Tensor] = None
    x_exog_dec: Optional[torch.Tensor] = None
    enc_self_mask: Optional[torch.Tensor] = None
    dec_self_mask: Optional[torch.Tensor] = None
    dec_enc_mask: Optional[torch.Tensor] = None


def _make_default_decoder_input(
    model: BaseModel,
    x_enc: torch.Tensor,
    x_target_enc: Optional[torch.Tensor],
) -> torch.Tensor:
    label_len = getattr(model, "label_len")
    pred_len = getattr(model, "pred_len")
    dec_in = int(model.config().get("dec_in", x_enc.size(-1)))
    source = x_target_enc if x_target_enc is not None and x_target_enc.size(-1) == dec_in else x_enc
    if source.size(-1) > dec_in:
        source = source[..., -dec_in:]
    elif source.size(-1) < dec_in:
        padding = source.new_zeros(*source.shape[:-1], dec_in - source.size(-1))
        source = torch.cat([source, padding], dim=-1)
    known = source[:, -label_len:, :] if label_len else source[:, :0, :]
    future = torch.zeros(source.size(0), pred_len, dec_in, device=source.device, dtype=source.dtype)
    return torch.cat([known, future], dim=1)


class ForecastingModel(BaseModel):
    """Base class for forecasting models with a standard predict entrypoint."""

    def predict(self, batch: ForecastingBatch) -> torch.Tensor:
        return self(
            batch.x_enc,
            batch.x_mark_enc,
            batch.x_dec,
            batch.x_mark_dec,
            enc_self_mask=batch.enc_self_mask,
            dec_self_mask=batch.dec_self_mask,
            dec_enc_mask=batch.dec_enc_mask,
        )

    def make_forecast_batch(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        x_target_enc: Optional[torch.Tensor] = None,
        x_exog_enc: Optional[torch.Tensor] = None,
        x_target_dec: Optional[torch.Tensor] = None,
        x_exog_dec: Optional[torch.Tensor] = None,
    ) -> ForecastingBatch:
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.size(0), x_enc.size(1), 4, device=x_enc.device, dtype=x_enc.dtype)
        if x_dec is None:
            x_dec = _make_default_decoder_input(self, x_enc, x_target_enc)
        if x_mark_dec is None:
            x_mark_dec = torch.zeros(x_dec.size(0), x_dec.size(1), 4, device=x_dec.device, dtype=x_dec.dtype)
        return ForecastingBatch(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            x_target_enc=x_target_enc,
            x_exog_enc=x_exog_enc,
            x_target_dec=x_target_dec,
            x_exog_dec=x_exog_dec,
        )


class ProbabilisticForecastingModel(BaseModel):
    """Base class for probabilistic forecasting models."""

    def predict(self, batch: ProbabilisticForecastingBatch) -> torch.Tensor:
        """Return a deterministic summary prediction, e.g. mean or median."""
        return self(
            batch.x_enc,
            batch.x_mark_enc,
            batch.x_dec,
            batch.x_mark_dec,
            enc_self_mask=batch.enc_self_mask,
            dec_self_mask=batch.dec_self_mask,
            dec_enc_mask=batch.dec_enc_mask,
        )

    def predict_distribution(self, batch: ProbabilisticForecastingBatch):
        raise NotImplementedError("Subclasses should return distribution parameters or a distribution object")

    def sample(self, batch: ProbabilisticForecastingBatch, num_samples: int = 100) -> torch.Tensor:
        raise NotImplementedError("Subclasses should generate samples with shape (num_samples, batch, pred_len, c_out)")

    def fit_step(self, batch: ProbabilisticForecastingBatch) -> torch.Tensor:
        if batch.y_true is None:
            raise ValueError("ProbabilisticForecastingBatch.y_true is required for fit_step")
        raise NotImplementedError("Subclasses should implement likelihood or sample-based training objectives")

    def make_probabilistic_forecast_batch(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        y_true: Optional[torch.Tensor] = None,
        x_target_enc: Optional[torch.Tensor] = None,
        x_exog_enc: Optional[torch.Tensor] = None,
        x_target_dec: Optional[torch.Tensor] = None,
        x_exog_dec: Optional[torch.Tensor] = None,
    ) -> ProbabilisticForecastingBatch:
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.size(0), x_enc.size(1), 4, device=x_enc.device, dtype=x_enc.dtype)
        if x_dec is None:
            x_dec = _make_default_decoder_input(self, x_enc, x_target_enc)
        if x_mark_dec is None:
            x_mark_dec = torch.zeros(x_dec.size(0), x_dec.size(1), 4, device=x_dec.device, dtype=x_dec.dtype)
        return ProbabilisticForecastingBatch(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            y_true=y_true,
            x_target_enc=x_target_enc,
            x_exog_enc=x_exog_enc,
            x_target_dec=x_target_dec,
            x_exog_dec=x_exog_dec,
        )
