"""FITS: low-pass complex-frequency interpolation for forecasting.

Independent, paper-aligned implementation of Xu et al., ICLR 2024:
https://arxiv.org/abs/2307.03756
Official reference: https://github.com/VEWOXIC/FITS (Apache-2.0).

The learnable operator is genuinely complex-linear, represented by paired real
parameters so normal ``Module.double()`` and optimizer behavior remain intact.
This implementation uses population variance for instance normalization, an
explicit inverse-FFT length, and a bounded output frequency count. It is not an
official-checkpoint adapter or a claim of reproduced benchmark performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


@dataclass
class FITSConfig(ForecastingConfig):
    cut_freq: int | None = None
    individual: bool = False
    use_norm: bool = True
    eps: float = 1e-5
    bias: bool = True


class FITS(ForecastingModel):
    """Forecast each input channel using a low-pass complex linear map.

    ``cut_freq`` is a count of retained one-sided rFFT bins, including DC, not
    a physical frequency in Hz. It defaults to ``min(32, seq_len // 2 + 1)``.
    The output count is ``min(cut_freq * (seq_len + pred_len) // seq_len,
    (seq_len + pred_len) // 2 + 1)``; the cap keeps maximal cutoffs legal.

    Channels never interact. ``individual=False`` shares the frequency map
    across channels; ``True`` learns one per channel. Therefore ``c_out`` must
    equal ``enc_in``. Standard attention, embedding, dropout and decoder
    arguments are accepted for ForecastingConfig compatibility but unused.

    ``forward`` returns only the future horizon. ``reconstruct_and_forecast``
    also exposes the reconstructed history for external backcast supervision.
    Only the observed input is consumed; no target or future decoder values
    enter the prediction. FFT computation supports float32/float64, with
    matching model/input dtypes, and is protected from lower-precision AMP.
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
        e_layers: int = 1,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.0,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        cut_freq: int | None = None,
        individual: bool = False,
        use_norm: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        del d_model, n_heads, e_layers, d_layers, d_ff, dropout, embed, freq, activation, output_attention
        for name, value in (("enc_in", enc_in), ("dec_in", dec_in), ("c_out", c_out), ("pred_len", pred_len)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seq_len, bool) or not isinstance(seq_len, Integral) or seq_len < 2:
            raise ValueError("seq_len must be an integer greater than or equal to 2")
        if isinstance(label_len, bool) or not isinstance(label_len, Integral) or not 0 <= label_len <= seq_len:
            raise ValueError("label_len must be an integer between 0 and seq_len")
        if c_out != enc_in:
            raise ValueError("FITS requires c_out == enc_in to preserve channel-independent forecast semantics")
        for name, value in (("individual", individual), ("use_norm", use_norm), ("bias", bias)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive real number")

        input_bins = seq_len // 2 + 1
        if cut_freq is None:
            cut_freq = min(32, input_bins)
        if isinstance(cut_freq, bool) or not isinstance(cut_freq, Integral) or not 1 <= cut_freq <= input_bins:
            raise ValueError(f"cut_freq must be an integer in [1, {input_bins}], including the DC bin")

        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.cut_freq = int(cut_freq)
        self.individual = individual
        self.use_norm = use_norm
        self.eps = float(eps)
        self.total_len = self.seq_len + self.pred_len
        self.length_ratio = self.total_len / self.seq_len
        self.out_freq = min(self.cut_freq * self.total_len // self.seq_len, self.total_len // 2 + 1)

        weight_shape = (self.out_freq, self.cut_freq)
        bias_shape = (self.out_freq,)
        if individual:
            weight_shape = (self.enc_in, *weight_shape)
            bias_shape = (self.enc_in, *bias_shape)
        self.weight_real = nn.Parameter(torch.empty(weight_shape))
        self.weight_imag = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias_real = nn.Parameter(torch.empty(bias_shape))
            self.bias_imag = nn.Parameter(torch.empty(bias_shape))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize both parts of the complex map with zero complex bias."""
        bound = 1 / math.sqrt(2 * self.cut_freq)
        nn.init.uniform_(self.weight_real, -bound, bound)
        nn.init.uniform_(self.weight_imag, -bound, bound)
        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("FITS expects a torch.Tensor input")
        if x.ndim != 3:
            raise ValueError("FITS expects x_enc with shape (batch, seq_len, enc_in)")
        if x.shape[1] != self.seq_len or x.shape[2] != self.enc_in:
            raise ValueError(f"FITS expects sequence length {self.seq_len} and {self.enc_in} input features")
        if x.shape[0] == 0:
            raise ValueError("FITS requires a non-empty batch")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("FITS FFT computation supports float32 or float64 inputs")
        if x.dtype != self.weight_real.dtype:
            raise TypeError("FITS input dtype must match model dtype; move the model and input together")
        if x.device != self.weight_real.device:
            raise ValueError("FITS input and model must be on the same device")
        if not torch.isfinite(x).all():
            raise ValueError("FITS input must contain only finite values")

    def _interpolate(self, spectrum: torch.Tensor) -> torch.Tensor:
        # (a + ib)(u + iv) = (au - bv) + i(av + bu). Omitting either
        # cross-term would not implement FITS's phase-sensitive complex map.
        if self.individual:
            real = torch.einsum("bci,coi->bco", spectrum.real, self.weight_real) - torch.einsum(
                "bci,coi->bco", spectrum.imag, self.weight_imag
            )
            imag = torch.einsum("bci,coi->bco", spectrum.real, self.weight_imag) + torch.einsum(
                "bci,coi->bco", spectrum.imag, self.weight_real
            )
        else:
            real = F.linear(spectrum.real, self.weight_real) - F.linear(spectrum.imag, self.weight_imag)
            imag = F.linear(spectrum.real, self.weight_imag) + F.linear(spectrum.imag, self.weight_real)
        if self.bias_real is not None:
            real = real + self.bias_real
            imag = imag + self.bias_imag
        return torch.complex(real, imag)

    def reconstruct_and_forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Return reconstructed history and future, shaped ``(B, L + H, C)``.

        Normalization statistics come from this input window only. Population
        variance (rather than the reference code's sample variance) makes
        short/constant-window behavior explicit. No running statistics or
        external state are fitted during inference.
        """
        self._validate_input(x_enc)
        with torch.autocast(device_type=x_enc.device.type, enabled=False):
            history = x_enc
            mean = scale = None
            if self.use_norm:
                mean = history.mean(dim=1, keepdim=True)
                scale = torch.sqrt(history.var(dim=1, keepdim=True, unbiased=False) + self.eps)
                history = (history - mean) / scale

            spectrum = torch.fft.rfft(history.transpose(1, 2), dim=-1)[..., : self.cut_freq]
            low = self._interpolate(spectrum)
            missing_bins = self.total_len // 2 + 1 - self.out_freq
            spectrum_out = torch.cat([low, low.new_zeros(*low.shape[:-1], missing_bins)], dim=-1)
            extended = torch.fft.irfft(spectrum_out, n=self.total_len, dim=-1).transpose(1, 2)
            extended = extended * self.length_ratio
            if self.use_norm:
                extended = extended * scale + mean
            return extended

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
        return self.reconstruct_and_forecast(x_enc)[:, -self.pred_len :, :]
