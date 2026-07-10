"""RUL regression adapters built on forecasting backbones."""

import torch.nn as nn

from torch_like.models.rul import RULConfig, RULModel

from .autoformer import Autoformer
from .informer import Informer
from .logtrans import LogTrans


class InformerRUL(RULModel, Informer):
    def __init__(
        self,
        enc_in=2,
        seq_len=2048,
        d_model=256,
        n_heads=8,
        e_layers=2,
        d_ff=1024,
        factor=5,
        dropout=0.1,
        distil=False,
        **kwargs,
    ):
        super().__init__(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=enc_in,
            seq_len=seq_len,
            label_len=1,
            pred_len=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=0,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            distil=distil,
            **kwargs,
        )
        self._config = RULConfig(
            enc_in=enc_in,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self._rul_pool = nn.AdaptiveAvgPool1d(1)
        self._rul_head = nn.Linear(d_model, 1)
        self._param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x, x_mark=None):
        enc_out = self.enc_embedding(x, x_mark)
        enc_out = self.encoder(enc_out)
        enc_out = enc_out.transpose(1, 2)
        enc_out = self._rul_pool(enc_out).squeeze(-1)
        return self._rul_head(enc_out).squeeze(-1)


class AutoformerRUL(RULModel, Autoformer):
    def __init__(
        self,
        enc_in=2,
        seq_len=2048,
        d_model=256,
        n_heads=8,
        e_layers=2,
        d_ff=1024,
        factor=1,
        moving_avg=25,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=enc_in,
            seq_len=seq_len,
            label_len=1,
            pred_len=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=0,
            d_ff=d_ff,
            factor=factor,
            moving_avg=moving_avg,
            dropout=dropout,
            **kwargs,
        )
        self._config = RULConfig(
            enc_in=enc_in,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self._rul_pool = nn.AdaptiveAvgPool1d(1)
        self._rul_head = nn.Linear(d_model, 1)
        self._param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x, x_mark=None):
        enc_out = self.enc_embedding(x, x_mark)
        enc_out = self.encoder(enc_out)
        enc_out = enc_out.transpose(1, 2)
        enc_out = self._rul_pool(enc_out).squeeze(-1)
        return self._rul_head(enc_out).squeeze(-1)


class LogTransRUL(RULModel, LogTrans):
    def __init__(
        self,
        enc_in=2,
        seq_len=2048,
        d_model=256,
        n_heads=8,
        e_layers=2,
        d_ff=1024,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=enc_in,
            seq_len=seq_len,
            label_len=1,
            pred_len=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=0,
            d_ff=d_ff,
            dropout=dropout,
            **kwargs,
        )
        self._config = RULConfig(
            enc_in=enc_in,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self._rul_pool = nn.AdaptiveAvgPool1d(1)
        self._rul_head = nn.Linear(d_model, 1)
        self._param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x, x_mark=None):
        enc_out = self.enc_embedding(x, x_mark)
        enc_out = self.encoder(enc_out)
        enc_out = enc_out.transpose(1, 2)
        enc_out = self._rul_pool(enc_out).squeeze(-1)
        return self._rul_head(enc_out).squeeze(-1)
