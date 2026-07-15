"""RTDetector anomaly detection model.

Implementation follows the official RTdetector repository structure while
adapting the interface to torch_like's detection task API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.detection.common import DetectionBatch, DetectionModel


@dataclass
class RTDetectorConfig:
    input_dim: int
    seq_len: int = 10
    ff_dim: int = 16
    projector_hidden_dim: int = 16
    projector_hidden_layers: int = 1
    dropout: float = 0.1


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(0)])


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, is_causal: bool = False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout2(src2)


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return tgt + self.dropout3(tgt2)


class _Projector(nn.Module):
    def __init__(
        self, enc_in: int, seq_len: int, hidden_dim: int, hidden_layers: int, output_dim: int, kernel_size: int = 3
    ):
        super().__init__()
        padding = kernel_size // 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        layers = [nn.Linear(2 * enc_in, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1)
        x = x.view(batch_size, -1)
        return self.backbone(x)


class RTDetector(DetectionModel):
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 10,
        ff_dim: int = 16,
        projector_hidden_dim: int = 16,
        projector_hidden_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2 for RTDetector")
        self._config = RTDetectorConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            ff_dim=ff_dim,
            projector_hidden_dim=projector_hidden_dim,
            projector_hidden_layers=projector_hidden_layers,
            dropout=dropout,
        )
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.model_dim = 2 * input_dim
        self.pos_encoder = _PositionalEncoding(self.model_dim, dropout, seq_len)
        encoder_layer = _TransformerEncoderLayer(
            self.model_dim, nhead=input_dim, dim_feedforward=ff_dim, dropout=dropout
        )
        decoder_layer1 = _TransformerDecoderLayer(
            self.model_dim, nhead=input_dim, dim_feedforward=ff_dim, dropout=dropout
        )
        decoder_layer2 = _TransformerDecoderLayer(
            self.model_dim, nhead=input_dim, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer1, num_layers=1)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2, num_layers=1)
        self.reconstruction_head = nn.Sequential(nn.Linear(self.model_dim, input_dim), nn.Sigmoid())
        self.tau_learner = _Projector(
            enc_in=self.model_dim,
            seq_len=seq_len,
            hidden_dim=projector_hidden_dim,
            hidden_layers=projector_hidden_layers,
            output_dim=1,
        )
        self.delta_learner = _Projector(
            enc_in=self.model_dim,
            seq_len=seq_len,
            hidden_dim=projector_hidden_dim,
            hidden_layers=projector_hidden_layers,
            output_dim=seq_len,
        )
        self._std_enc = None
        self._mean_enc = None

    def _encode(self, src: torch.Tensor, conditioning: torch.Tensor, tgt: torch.Tensor):
        src = torch.cat((src, conditioning), dim=2)
        src = src.permute(1, 0, 2)
        mean_enc = src.mean(dim=1, keepdim=True).detach()
        src = src - mean_enc
        std_enc = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        self._std_enc = std_enc
        self._mean_enc = mean_enc
        src = src / std_enc
        tau = self.tau_learner(src, std_enc).exp()
        delta = self.delta_learner(src, mean_enc)
        tau = tau.unsqueeze(1)
        delta = delta.unsqueeze(1)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        memory = memory.permute(1, 0, 2)
        delta = delta.permute(0, 2, 1)
        memory = memory * tau + delta
        memory = memory.permute(1, 0, 2)
        tgt = tgt.repeat(1, 1, 2)
        result = delta.mean(dim=1, keepdim=True)
        tgt = tgt.permute(1, 0, 2)
        tgt = tgt * tau + result
        return tgt.permute(1, 0, 2), memory

    def _decode_phase(
        self, decoder: nn.Module, src: torch.Tensor, conditioning: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        decoded = decoder(*self._encode(src, conditioning, tgt)).squeeze(0)
        decoded = decoded * self._std_enc.squeeze(1) + self._mean_enc.squeeze(1)
        decoded = decoded.unsqueeze(0)
        return self.reconstruction_head(decoded)

    def forward(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        src = x.permute(1, 0, 2)
        tgt = src[-1:, :, :]
        phase1 = self._decode_phase(self.transformer_decoder1, src, torch.zeros_like(src), tgt)
        conditioning = (phase1 - src) ** 2
        phase2 = self._decode_phase(self.transformer_decoder2, src, conditioning, tgt)
        return phase2.squeeze(0)

    def reconstruct(self, batch: DetectionBatch):
        src = batch.x.permute(1, 0, 2)
        tgt = src[-1:, :, :]
        phase1 = self._decode_phase(self.transformer_decoder1, src, torch.zeros_like(src), tgt)
        conditioning = (phase1 - src) ** 2
        phase2 = self._decode_phase(self.transformer_decoder2, src, conditioning, tgt)
        return phase1.squeeze(0), phase2.squeeze(0)

    def anomaly_score(self, batch: DetectionBatch, reduction: str = "none") -> torch.Tensor:
        return self.score_batch(batch, reduction=reduction)

    def fit_step(self, batch: DetectionBatch):
        _, phase2 = self.reconstruct(batch)
        target = batch.x[:, -1, :]
        return torch.mean((phase2 - target) ** 2)
