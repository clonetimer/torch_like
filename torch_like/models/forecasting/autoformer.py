"""Autoformer: Decomposition Transformers with Auto-Correlation."""

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.nn.embedding import DataEmbedding
from torch_like.nn.attention import AutoCorrelation, AutoCorrelationLayer
from torch_like.nn.decomposition import SeriesDecomp


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_ff=None, moving_avg=25, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x, _ = self.decomp1(x)
        ffn_out = self.conv2(self.activation(self.conv1(x.transpose(-1, 1)))).transpose(-1, 1)
        x = x + self.dropout(ffn_out)
        x, _ = self.decomp2(x)
        return x


class AutoformerEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    def forward(self, x, attn_mask=None):
        for layer in self.attn_layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x) if self.norm is not None else x


class AutoformerDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, n_heads, d_ff=None, moving_avg=25, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attn_out, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(attn_out)
        x, trend1 = self.decomp1(x)
        cross_out, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(cross_out)
        x, trend2 = self.decomp2(x)
        ffn_out = self.conv2(self.activation(self.conv1(x.transpose(-1, 1)))).transpose(-1, 1)
        x = x + self.dropout(ffn_out)
        x, trend3 = self.decomp3(x)
        return x, trend1 + trend2 + trend3


class AutoformerDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class Autoformer(ForecastingModel):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=25, factor=1, dropout=0.05, embed="fixed", freq="h", activation="gelu", output_attention=False):
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
        self.decomp = SeriesDecomp(moving_avg)
        enc_attn_layers = []
        for _ in range(e_layers):
            enc_attn_layers.append(AutoformerEncoderLayer(AutoCorrelationLayer(AutoCorrelation(mask_flag=False, factor=factor, attention_dropout=dropout), d_model, n_heads), d_model=d_model, n_heads=n_heads, d_ff=d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation))
        self.encoder = AutoformerEncoder(enc_attn_layers, norm_layer=nn.LayerNorm(d_model))
        dec_layers = []
        for _ in range(d_layers):
            dec_layers.append(AutoformerDecoderLayer(self_attention=AutoCorrelationLayer(AutoCorrelation(mask_flag=True, factor=factor, attention_dropout=dropout), d_model, n_heads), cross_attention=AutoCorrelationLayer(AutoCorrelation(mask_flag=False, factor=factor, attention_dropout=dropout), d_model, n_heads), d_model=d_model, n_heads=n_heads, d_ff=d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation))
        self.decoder = AutoformerDecoder(dec_layers, norm_layer=nn.LayerNorm(d_model), projection=None)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self._param_count = sum(p.numel() for p in self.parameters())
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, trend_init = self.decomp(enc_out)
        mean = enc_out.mean(dim=1, keepdim=True).detach().repeat(1, self.pred_len, 1)
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out, trend = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        dec_out = self.projection(dec_out + trend)
        return dec_out[:, -self.pred_len :, :]
