"""Informer: ProbSparse self-attention transformer for long-sequence forecasting."""

import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.nn.attention import FullAttention, ProbSparseAttention
from torch_like.nn.decoder import Decoder, DecoderLayer
from torch_like.nn.embedding import DataEmbedding
from torch_like.nn.encoder import ConvLayer, Encoder, EncoderLayer


class Informer(ForecastingModel):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        pred_len,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=2048,
        factor=5,
        dropout=0.05,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self._config = ForecastingConfig(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            embed=embed,
            freq=freq,
            activation=activation,
            output_attention=output_attention,
        )

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        AttnCls = ProbSparseAttention if attn == "prob" else FullAttention

        enc_attn_layers, enc_conv_layers = [], []
        self_attn_args = dict(
            mask_flag=False, factor=factor, attention_dropout=dropout, output_attention=output_attention
        )
        cross_attn_args = dict(mask_flag=False, attention_dropout=dropout, output_attention=output_attention)

        for i in range(e_layers):
            enc_attn_layers.append(
                EncoderLayer(AttnCls(**self_attn_args), d_model, n_heads, d_ff, dropout=dropout, activation=activation)
            )
            if distil and i < e_layers - 1:
                enc_conv_layers.append(ConvLayer(d_model))

        self.encoder = Encoder(
            enc_attn_layers, conv_layers=enc_conv_layers if distil else None, norm_layer=nn.LayerNorm(d_model)
        )

        dec_layers = []
        for _ in range(d_layers):
            dec_layers.append(
                DecoderLayer(
                    self_attention=AttnCls(**self_attn_args),
                    cross_attention=FullAttention(**cross_attn_args),
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.decoder = Decoder(
            dec_layers, norm_layer=nn.LayerNorm(d_model), projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        return dec_out[:, -self.pred_len :, :]
