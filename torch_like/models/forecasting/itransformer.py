"""iTransformer: inverted Transformer for multivariate time series forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel


class ITransformer(ForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
    ):
        super().__init__()
        del dec_in, d_layers, embed, freq, output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation=activation,
            output_attention=False,
        )

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.variate_pos_embed = nn.Parameter(torch.zeros(1, enc_in, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)
        self.output_proj = nn.Identity() if c_out == enc_in else nn.Linear(enc_in, c_out)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        if x_enc.dim() != 3:
            raise ValueError("ITransformer expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        # Inverted tokenization: each variate becomes one token, time dimension becomes token feature.
        x = x_enc.transpose(1, 2)  # (B, C, L)
        x = self.value_embedding(x) + self.variate_pos_embed  # (B, C, D)
        x = self.dropout(x)
        x = self.encoder(x)  # attention over variates
        x = self.head(x).transpose(1, 2)  # (B, pred_len, C)
        x = self.output_proj(x)
        return x
