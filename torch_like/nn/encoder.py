"""Encoder with self-attention distilling for long sequences.

The Informer encoder stacks multiple attention layers and periodically
distills the sequence via 1D convolution + max pooling, progressively
halving the sequence length while retaining dominant features.

Reference: Informer, Section 3.2 (Self-attention Distilling).
"""

import torch.nn as nn


class ConvLayer(nn.Module):
    """1D convolutional distilling layer: Conv -> ELU -> MaxPool(stride=2).

    Halves the sequence length while preserving salient features.
    """

    def __init__(self, d_model):
        super().__init__()
        self.down_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (B, L, d_model)
        x = self.down_conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    """Single encoder layer: ProbSparse self-attention + FFN.

    Handles multi-head QKV projections internally; the hosted attention
    module receives (B, L, H, D) tensors.
    """

    def __init__(self, attention, d_model, n_heads, d_ff=None,
                 dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.n_heads = n_heads

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        H = self.n_heads

        # Multi-head QKV projections
        q = self.w_q(x).view(B, L, H, -1)
        k = self.w_k(x).view(B, L, H, -1)
        v = self.w_v(x).view(B, L, H, -1)

        # Self-attention with residual
        attn_out, _ = self.attention(q, k, v, attn_mask=attn_mask)
        attn_out = self.out_proj(attn_out.reshape(B, L, -1))
        x = x + self.dropout(attn_out)
        y = self.norm1(x)

        # FFN with residual
        ffn_out = self.conv2(self.activation(self.conv1(y.transpose(-1, 1)))).transpose(-1, 1)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    """Informer encoder with progressive self-attention distilling.

    Args:
        attn_layers: list of EncoderLayer instances.
        conv_layers: list of ConvLayer instances (for distilling).
        norm_layer: optional final LayerNorm.
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)

            x = self.attn_layers[-1](x, attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
