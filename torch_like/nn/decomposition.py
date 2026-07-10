"""Series decomposition block — core component of Autoformer.

SeriesDecomp uses moving average to decompose a time series into
seasonal (residual) and trend (moving average) components:

    seasonal = x - moving_avg(x)
    trend    = moving_avg(x)

Reference:
    Wu et al., "Autoformer: Decomposition Transformers with
    Auto-Correlation for Long-Term Series Forecasting" (NeurIPS 2021)
"""

import torch.nn as nn


class SeriesDecomp(nn.Module):
    """Decompose time series into trend and seasonal via moving average.

    Args:
        kernel_size: moving average window size (odd recommended).
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )

    def forward(self, x):
        """Decompose x into trend and seasonal.

        Args:
            x: (B, L, D) — input sequence.

        Returns:
            seasonal: (B, L, D) — residual component.
            trend:    (B, L, D) — moving average component.
        """
        # AvgPool1d expects (B, D, L)
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend
