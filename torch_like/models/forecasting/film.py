"""FiLM: fixed Legendre memory with frequency enhancement and multiscale fusion.

Paper: Zhou et al., NeurIPS 2022, https://arxiv.org/abs/2205.08897.
Official MIT reference: tianzhou2011/FiLM at
2794355ff6258743a29715263414283782910521. This independently implemented
portable subset is not an official-checkpoint or benchmark-reproduction claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.models.forecasting.film_layers import _FrequencyEnhancedLayer, _LegendreProjection


@dataclass
class FiLMConfig(ForecastingConfig):
    memory_order: int = 64
    modes: int = 16
    scales: tuple[int, ...] = (1, 2, 4)
    use_norm: bool = True
    eps: float = 1e-5


class FiLM(ForecastingModel):
    """Forecast (B,H,C) from observed (B,L,C) with channel-independent memory.

    Every scale s uses the last min(L,s*H) observations, bilinear LMU dynamics
    with dt=1/(s*H), coefficient-frequency mixing and Legendre reconstruction.
    Like the pinned reference, the filtered coefficient index is min(H-1,T-1),
    NOT always the last index. A learned scalar Linear(S,1) fuses the scales.

    ``modes`` counts contiguous low rFFT bins including DC. The effective count
    is min(modes,max(1,min(L,H)//2)); retaining DC for H=1 is an explicit
    extension of the reference. Memory order defaults to 64 rather than 256.
    Optional low-rank/random-mode variants and learned normalization affine
    are not included. Dense recurrence costs O(B*C*T*N^2) per scale and stores
    the full O(B*C*N*T) coefficient history; this is not a parallel scan.

    ``c_out`` must equal ``enc_in``. Decoder values, time markers, masks and
    standard attention/embedding settings are accepted but unused. No masked
    observations or future covariates are modeled. Population normalization
    uses detached observed-window statistics only. Float32/float64 model and
    input dtypes/devices must match; lower-precision AMP is disabled internally.
    Nonfinite normalization statistics/normalized values raise a numerical-range
    ValueError before they can produce invalid forecasts or training gradients.
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.05,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        memory_order: int = 64,
        modes: int = 16,
        scales: tuple[int, ...] = (1, 2, 4),
        use_norm: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        del d_model, n_heads, e_layers, d_layers, d_ff, dropout, embed, freq, activation, output_attention
        for name, value in (
            ("enc_in", enc_in),
            ("dec_in", dec_in),
            ("c_out", c_out),
            ("pred_len", pred_len),
            ("memory_order", memory_order),
            ("modes", modes),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seq_len, bool) or not isinstance(seq_len, Integral) or seq_len < 2:
            raise ValueError("seq_len must be an integer greater than or equal to 2")
        if isinstance(label_len, bool) or not isinstance(label_len, Integral) or not 0 <= label_len <= seq_len:
            raise ValueError("label_len must be an integer between 0 and seq_len")
        if c_out != enc_in:
            raise ValueError("FiLM requires c_out == enc_in for channel-independent forecasting")
        if not isinstance(scales, (tuple, list)) or not scales:
            raise ValueError("scales must be a non-empty tuple or list of unique positive integers")
        if any(isinstance(s, bool) or not isinstance(s, Integral) or s <= 0 for s in scales):
            raise ValueError("scales must contain unique positive integers")
        if len(set(scales)) != len(scales):
            raise ValueError("scales must contain unique positive integers")
        if not isinstance(use_norm, bool):
            raise ValueError("use_norm must be a bool")
        if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive real number")
        eps32 = torch.tensor(float(eps), dtype=torch.float32)
        if not torch.isfinite(eps32) or eps32 <= 0:
            raise ValueError("eps must be representable as a finite nonzero float32")

        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.memory_order = int(memory_order)
        self.modes = int(modes)
        self.scales = tuple(int(s) for s in scales)
        self.use_norm = use_norm
        self.eps = float(eps)
        self.retained_modes = min(self.modes, max(1, min(self.seq_len, self.pred_len) // 2))
        self.projections = nn.ModuleList(
            [_LegendreProjection(self.memory_order, s * self.pred_len, self.pred_len) for s in self.scales]
        )
        self.filters = nn.ModuleList(
            [_FrequencyEnhancedLayer(self.memory_order, self.retained_modes) for _ in self.scales]
        )
        self.scale_fusion = nn.Linear(len(self.scales), 1)
        nn.init.constant_(self.scale_fusion.weight, 1 / len(self.scales))
        nn.init.zeros_(self.scale_fusion.bias)

    def config(self) -> dict:
        """Return JSON-stable configuration, canonicalizing scales to a list."""
        config = super().config()
        config["scales"] = list(self.scales)
        return config

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("FiLM expects a torch.Tensor input")
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"FiLM expects shape (batch, {self.seq_len}, {self.enc_in})")
        if x.shape[0] == 0:
            raise ValueError("FiLM requires a non-empty batch")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("FiLM computation supports float32 or float64 inputs")
        if x.dtype != self.scale_fusion.weight.dtype:
            raise TypeError("FiLM input dtype must match model dtype")
        if x.device != self.scale_fusion.weight.device:
            raise ValueError("FiLM input and model must be on the same device")
        if not torch.isfinite(x).all():
            raise ValueError("FiLM input must contain only finite values")

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
        self._validate_input(x_enc)
        with torch.autocast(device_type=x_enc.device.type, enabled=False):
            history = x_enc
            if self.use_norm:
                mean = history.mean(dim=1, keepdim=True).detach()
                scale = torch.sqrt(history.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
                if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
                    raise ValueError(
                        "FiLM normalization exceeds the input dtype numerical range; rescale data or use float64"
                    )
                history = (history - mean) / scale
                if not torch.isfinite(history).all():
                    raise ValueError(
                        "FiLM normalization exceeds the input dtype numerical range; rescale data or use float64"
                    )
            branches = []
            for factor, projection, spectral_filter in zip(self.scales, self.projections, self.filters, strict=True):
                observed = history[:, -(factor * self.pred_len) :, :].transpose(1, 2)
                filtered = spectral_filter(projection(observed))
                coefficients = filtered[..., min(self.pred_len - 1, filtered.size(-1) - 1)]
                branches.append(projection.reconstruct(coefficients))
            forecast = self.scale_fusion(torch.stack(branches, dim=-1)).squeeze(-1).transpose(1, 2)
            if self.use_norm:
                forecast = forecast * scale + mean
            return forecast
