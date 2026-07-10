"""TimeMixer: decomposable multiscale mixing for forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import ForecastingConfig, ForecastingModel
from torch_like.nn.decomposition import SeriesDecomp


class MixerMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len: int, down_sampling_layers: int, moving_avg: int, d_ff: int, dropout: float):
        super().__init__()
        self.decomp = SeriesDecomp(moving_avg)
        self.down_sampling_layers = down_sampling_layers
        self.season_mixers = nn.ModuleList()
        self.trend_mixers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(down_sampling_layers + 1):
            length = max(1, seq_len // (2**i))
            self.season_mixers.append(MixerMLP(length, d_ff, dropout))
            self.trend_mixers.append(MixerMLP(length, d_ff, dropout))
            self.norms.append(nn.LayerNorm(length))

    def _build_multiscale(self, x: torch.Tensor):
        scales = [x]
        current = x.transpose(1, 2)
        for _ in range(self.down_sampling_layers):
            current = F.avg_pool1d(current, kernel_size=2, stride=2, ceil_mode=True)
            scales.append(current.transpose(1, 2))
        return scales

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        scales = self._build_multiscale(x)
        outputs = []
        prev_season = None
        prev_trend = None
        for idx, scale in enumerate(scales):
            season, trend = self.decomp(scale)
            season = season.transpose(1, 2)
            trend = trend.transpose(1, 2)
            if prev_season is not None:
                prev_up = F.interpolate(prev_season, size=season.size(-1), mode="linear", align_corners=False)
                season = season + prev_up
            if prev_trend is not None:
                prev_up = F.interpolate(prev_trend, size=trend.size(-1), mode="linear", align_corners=False)
                trend = trend + prev_up
            season = self.norms[idx](season + self.season_mixers[idx](season))
            trend = self.norms[idx](trend + self.trend_mixers[idx](trend))
            fused = (season + trend).transpose(1, 2)
            outputs.append(fused)
            prev_season = season
            prev_trend = trend
        return outputs


class FutureMultiPredictorMixing(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, down_sampling_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.predictors = nn.ModuleList()
        self.weights = nn.Parameter(torch.ones(down_sampling_layers + 1))
        for i in range(down_sampling_layers + 1):
            length = max(1, seq_len // (2**i))
            self.predictors.append(
                nn.Sequential(
                    nn.Linear(length, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, pred_len),
                )
            )

    def forward(self, multiscale_features: list[torch.Tensor]) -> torch.Tensor:
        preds = []
        for feature, predictor in zip(multiscale_features, self.predictors):
            preds.append(predictor(feature.transpose(1, 2)).transpose(1, 2))
        stacked = torch.stack(preds, dim=0)
        weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (weights * stacked).sum(dim=0)


class TimeMixer(ForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        moving_avg: int = 25,
        down_sampling_layers: int = 2,
    ):
        super().__init__()
        del dec_in, d_model, n_heads, e_layers, d_layers, embed, freq, activation, output_attention
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
            d_model=64,
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=d_ff,
            dropout=dropout,
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
        )
        self.normalize = nn.LayerNorm(enc_in)
        self.pdm = PastDecomposableMixing(seq_len, down_sampling_layers, moving_avg, d_ff, dropout)
        self.fmm = FutureMultiPredictorMixing(seq_len, pred_len, down_sampling_layers, d_ff, dropout)
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
            raise ValueError("TimeMixer expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} features, got {x_enc.size(-1)}")

        x = self.normalize(x_enc)
        multiscale = self.pdm(x)
        out = self.fmm(multiscale)
        out = self.output_proj(out)
        return out
