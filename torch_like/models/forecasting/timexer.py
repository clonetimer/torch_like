"""TimeXer-inspired forecasting model with endogenous-exogenous dual input."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch_like.models.forecasting.common import ForecastingBatch, ForecastingConfig, ForecastingModel


class TxPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


class TimeXerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        self_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(self_out))

        glb = x[:, -1:, :]
        cross_out, _ = self.cross_attn(glb, cross, cross, need_weights=False)
        glb = self.norm2(glb + self.dropout(cross_out))
        x = torch.cat([x[:, :-1, :], glb], dim=1)

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class FlattenHead(nn.Module):
    def __init__(self, patch_num: int, d_model: int, pred_len: int, dropout: float):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear((patch_num + 1) * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(self.flatten(x)))


class TimeXer(ForecastingModel):
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
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        patch_len: int = 16,
        use_norm: bool = True,
    ):
        super().__init__()
        del dec_in, d_layers, embed, freq, output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.patch_len = patch_len
        self.use_norm = use_norm
        if seq_len < patch_len:
            raise ValueError("seq_len must be >= patch_len")
        self.patch_num = seq_len // patch_len
        if self.patch_num < 1:
            raise ValueError("Invalid patch_num")
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

        self.target_patch_embed = nn.Linear(patch_len, d_model, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.target_pos_embed = TxPositionalEmbedding(d_model, max_len=self.patch_num + 1)
        self.exog_embed = nn.LazyLinear(d_model)
        self.encoder_layers = nn.ModuleList(
            [TimeXerEncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)]
        )
        self.head = FlattenHead(self.patch_num, d_model, pred_len, dropout)

    def _prepare_target_tokens(self, x_target_enc: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch, _, target_dim = x_target_enc.shape
        x = x_target_enc.transpose(1, 2)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = self.target_patch_embed(x)
        glb = self.global_token.expand(batch, target_dim, 1, -1)
        pos = self.target_pos_embed(x.size(2) + 1).unsqueeze(1)
        x = torch.cat([x, glb], dim=2) + pos
        x = x.reshape(batch * target_dim, x.size(2), x.size(3))
        return x, target_dim

    def _prepare_exog_tokens(
        self,
        x_exog_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        target_dim: int,
    ) -> torch.Tensor:
        if x_mark_enc is not None and x_mark_enc.numel() > 0:
            x = torch.cat([x_exog_enc, x_mark_enc], dim=-1)
        else:
            x = x_exog_enc
        x = self.exog_embed(x).transpose(1, 2)
        x = x.repeat_interleave(target_dim, dim=0)
        return x

    def predict(self, batch: ForecastingBatch) -> torch.Tensor:
        x_target_enc = batch.x_target_enc
        if x_target_enc is None or x_target_enc.numel() == 0:
            x_target_enc = batch.x_enc[..., -self.c_out :]
        x_exog_enc = batch.x_exog_enc
        if x_exog_enc is None or x_exog_enc.numel() == 0:
            exog_parts = [batch.x_enc]
            if batch.x_mark_enc is not None:
                exog_parts.append(batch.x_mark_enc)
            x_exog_enc = torch.cat(exog_parts, dim=-1)
            x_mark_enc = None
        else:
            x_mark_enc = batch.x_mark_enc

        means = None
        stdev = None
        if self.use_norm:
            means = x_target_enc.mean(1, keepdim=True).detach()
            x_target_enc = x_target_enc - means
            stdev = torch.sqrt(torch.var(x_target_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_target_enc = x_target_enc / stdev

        target_tokens, target_dim = self._prepare_target_tokens(x_target_enc)
        exog_tokens = self._prepare_exog_tokens(x_exog_enc, x_mark_enc, target_dim)
        for layer in self.encoder_layers:
            target_tokens = layer(target_tokens, exog_tokens)
        target_tokens = target_tokens.reshape(-1, target_dim, target_tokens.size(1), target_tokens.size(2)).permute(
            0, 1, 3, 2
        )
        out = self.head(target_tokens).transpose(1, 2)

        if self.use_norm and means is not None and stdev is not None:
            out = out * stdev[:, 0, :target_dim].unsqueeze(1).repeat(1, self.pred_len, 1)
            out = out + means[:, 0, :target_dim].unsqueeze(1).repeat(1, self.pred_len, 1)
        return out

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
        del x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        batch = ForecastingBatch(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=torch.empty(0, device=x_enc.device),
            x_mark_dec=None,
        )
        return self.predict(batch)
