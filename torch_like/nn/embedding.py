"""Data embedding for time series: value + position + temporal.

Three components combined: scalar projection, learnable positional encoding,
and temporal feature embedding.

Reference: Informer (Zhou et al., AAAI 2021), Section 3.1.
"""

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding (not sinusoidal)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)

    def forward(self, x):
        # x: (L, B, d_model) — time-first format
        return x + self.pe[: x.size(0)]


class TokenEmbedding(nn.Module):
    """Conv1d projection for scalar -> d_model."""

    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        # x: (B, L, C)
        x = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    """Embed temporal features into d_model dims via linear projection.

    Accepts variable-length feature vectors (e.g., 4 for hourly, 5 for second-level).
    """

    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super().__init__()
        # Use a generic linear projection instead of per-feature embeddings
        # so the module works with any number of time features
        self.linear = nn.Linear(1, d_model, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        # x: (B, L, F) — F = number of time features (varies by freq)
        # Sum projections across feature dimension
        out = 0.0  # scalar accumulator
        for i in range(x.shape[-1]):
            out = out + self.linear(x[:, :, i : i + 1])
        return out


class DataEmbedding(nn.Module):
    """Complete data embedding: TokenEmbedding + PositionalEmbedding + TemporalEmbedding.

    Args:
        c_in: number of input features.
        d_model: model dimension.
        dropout: dropout rate.
    """

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model, embed_type, freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """Forward pass.

        Args:
            x: (B, L, C) input sequence.
            x_mark: (B, L, 4) temporal features; can be None.

        Returns:
            (B, L, d_model)
        """
        if x_mark is None:
            x_mark = torch.zeros(x.size(0), x.size(1), 4, device=x.device)
        tok = self.token_embedding(x)
        pos = self.position_embedding(tok.transpose(0, 1)).transpose(0, 1)
        temp = self.temporal_embedding(x_mark)
        return self.dropout(pos + temp)
