"""Private native Legendre-memory and frequency-enhancement operators for FiLM.

Equations follow Zhou et al. (NeurIPS 2022), https://arxiv.org/abs/2205.08897.
Reference: tianzhou2011/FiLM, MIT, commit 2794355ff6258743a29715263414283782910521.
No SciPy runtime, custom FFT extension, or reference training code is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LegendreProjection(nn.Module):
    """Fixed bilinear LMU recurrence, with a horizon-only Legendre basis.

    Validated model settings supply positive order/window and H <= window.
    Setup is calculated in float64, then buffers use the default floating dtype.
    Forward maps (B,C,T) to (B,C,N,T), without carrying state between calls.
    """

    def __init__(self, order: int, window: int, pred_len: int):
        super().__init__()
        self.order = order
        rows = torch.arange(order, dtype=torch.float64)[:, None]
        cols = torch.arange(order, dtype=torch.float64)[None, :]
        signs = torch.where((rows - cols + 1).remainder(2) == 0, 1.0, -1.0)
        continuous_a = (2 * rows + 1) * torch.where(rows < cols, -1.0, signs)
        index = torch.arange(order, dtype=torch.float64)
        continuous_b = (2 * index + 1) * torch.where(index.remainder(2) == 0, 1.0, -1.0)
        identity = torch.eye(order, dtype=torch.float64)
        left = identity - continuous_a / (2 * window)
        discrete_a = torch.linalg.solve(left, identity + continuous_a / (2 * window))
        discrete_b = torch.linalg.solve(left, continuous_b / window)

        # Only the final H grid rows are consumed. Do not include t=1.
        z = 1 - 2 * torch.arange(window - pred_len, window, dtype=torch.float64) / window
        polynomials = [torch.ones_like(z)]
        if order > 1:
            polynomials.append(z)
        for degree in range(2, order):
            polynomials.append(((2 * degree - 1) * z * polynomials[-1] - (degree - 1) * polynomials[-2]) / degree)
        dtype = torch.get_default_dtype()
        self.register_buffer("A", discrete_a.to(dtype=dtype))
        self.register_buffer("B", discrete_b.to(dtype=dtype))
        self.register_buffer("basis", torch.stack(polynomials, dim=-1).to(dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x.new_zeros(*x.shape[:-1], self.order)
        states = []
        for sample in x.unbind(dim=-1):
            state = F.linear(state, self.A) + sample.unsqueeze(-1) * self.B
            states.append(state)
        return torch.stack(states, dim=-1)

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        return F.linear(coefficients, self.basis)


class _FrequencyEnhancedLayer(nn.Module):
    """Low-pass complex mixing of memory coefficients, independently per channel.

    The time-axis rFFT is mixed as input-order x output-order x frequency.
    Paired real parameters preserve normal Module.double() and optimizer rules.
    DC (and even-length Nyquist) imaginary parts cannot affect a real irFFT.
    """

    def __init__(self, order: int, modes: int):
        super().__init__()
        self.modes = modes
        self.weight_real = nn.Parameter(torch.empty(order, order, modes))
        self.weight_imag = nn.Parameter(torch.empty(order, order, modes))
        bound = 1 / (order * order)
        nn.init.uniform_(self.weight_real, -bound, bound)
        nn.init.uniform_(self.weight_imag, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1)
        count = min(self.modes, spectrum.size(-1))
        low = spectrum[..., :count]
        real_weight = self.weight_real[..., :count]
        imag_weight = self.weight_imag[..., :count]
        real = torch.einsum("bcim,iom->bcom", low.real, real_weight) - torch.einsum(
            "bcim,iom->bcom", low.imag, imag_weight
        )
        imag = torch.einsum("bcim,iom->bcom", low.real, imag_weight) + torch.einsum(
            "bcim,iom->bcom", low.imag, real_weight
        )
        output = torch.cat(
            [torch.complex(real, imag), spectrum.new_zeros(*spectrum.shape[:-1], spectrum.size(-1) - count)],
            dim=-1,
        )
        return torch.fft.irfft(output, n=x.size(-1), dim=-1)
