"""CBAM-enhanced temporal convolution adapter for anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.detection.common import DetectionConfig, DetectionModel
from torch_like.nn.cbam import CBAM1d


@dataclass
class CBAMTCNDetectorConfig(DetectionConfig):
    e_layers: int = 2
    kernel_size: int = 5
    cbam_reduction: int = 8
    cbam_kernel_size: int = 7
    use_cbam: bool = True


class _CBAMTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        reduction: int,
        cbam_kernel: int,
        use_cbam: bool,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.padding = padding
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.cbam = CBAM1d(channels, reduction=reduction, kernel_size=cbam_kernel) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        if self.padding:
            x = x[..., : -self.padding]
        x = self.dropout(self.activation(self.norm(x)))
        return residual + self.cbam(x)


class CBAMTCNDetector(DetectionModel):
    """Predict the final sample from preceding context with a CBAM-TCN encoder."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        dropout: float = 0.1,
        threshold: float = 0.5,
        e_layers: int = 2,
        kernel_size: int = 5,
        cbam_reduction: int = 8,
        cbam_kernel_size: int = 7,
        use_cbam: bool = True,
    ):
        super().__init__()
        del threshold
        if input_dim <= 0 or d_model <= 0 or e_layers <= 0:
            raise ValueError("input_dim, d_model, and e_layers must be positive")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self._config = CBAMTCNDetectorConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            e_layers=e_layers,
            kernel_size=kernel_size,
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size,
            use_cbam=use_cbam,
        )
        self.input_projection = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                _CBAMTemporalBlock(
                    d_model,
                    kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                    reduction=cbam_reduction,
                    cbam_kernel=cbam_kernel_size,
                    use_cbam=use_cbam,
                )
                for layer in range(e_layers)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, input_dim))

    def forward(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        del x_mark
        if x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(f"CBAMTCNDetector expects x with shape (batch, {self.seq_len}, {self.input_dim})")
        context = x[:, :-1].transpose(1, 2)
        encoded = self.blocks(self.input_projection(context))
        return self.output(encoded[:, :, -1])


__all__ = ["CBAMTCNDetector", "CBAMTCNDetectorConfig"]
