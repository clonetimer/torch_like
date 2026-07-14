"""Decoder with masked self-attention and cross-attention.

The Informer decoder uses a generative inference style: it takes a start
token (from the end of encoder input) concatenated with zeros as placeholder
for the prediction horizon, and predicts all future steps at once.

Reference: Informer, Section 3.3 (Generative Style Decoder).
"""

import torch.nn as nn


class DecoderLayer(nn.Module):
    """Single decoder layer: masked self-attn + cross-attn + FFN.

    Handles multi-head QKV projections internally.
    """

    def __init__(self, self_attention, cross_attention, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        # Self-attention QKV projections
        self.self_w_q = nn.Linear(d_model, d_model, bias=False)
        self.self_w_k = nn.Linear(d_model, d_model, bias=False)
        self.self_w_v = nn.Linear(d_model, d_model, bias=False)
        self.self_out_proj = nn.Linear(d_model, d_model, bias=False)

        # Cross-attention projections
        self.cross_w_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_w_k = nn.Linear(d_model, d_model, bias=False)
        self.cross_w_v = nn.Linear(d_model, d_model, bias=False)
        self.cross_out_proj = nn.Linear(d_model, d_model, bias=False)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.n_heads = n_heads

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        B, L, _ = x.shape
        Lc = cross.shape[1]
        H = self.n_heads

        # Self-attention (with triangular mask for auto-regressive)
        q_s = self.self_w_q(x).view(B, L, H, -1)
        k_s = self.self_w_k(x).view(B, L, H, -1)
        v_s = self.self_w_v(x).view(B, L, H, -1)
        self_out, _ = self.self_attention(q_s, k_s, v_s, attn_mask=x_mask)
        x = x + self.dropout(self.self_out_proj(self_out.reshape(B, L, -1)))
        x = self.norm1(x)

        # Cross-attention with encoder output
        q_c = self.cross_w_q(x).view(B, L, H, -1)
        k_c = self.cross_w_k(cross).view(B, Lc, H, -1)
        v_c = self.cross_w_v(cross).view(B, Lc, H, -1)
        cross_out, _ = self.cross_attention(q_c, k_c, v_c, attn_mask=cross_mask)
        x = x + self.dropout(self.cross_out_proj(cross_out.reshape(B, L, -1)))
        y = self.norm2(x)

        # FFN
        ffn_out = self.conv2(self.activation(self.conv1(y.transpose(-1, 1)))).transpose(-1, 1)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    """Informer decoder: stacked DecoderLayers with final projection.

    Args:
        layers: list of DecoderLayer instances.
        norm_layer: optional final LayerNorm.
        projection: optional final Linear projection.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x
