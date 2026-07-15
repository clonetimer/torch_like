"""Wav2Vec 2.0-style raw waveform encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.audio.common import AudioConfig, AudioModel


@dataclass
class Wav2Vec2Config(AudioConfig):
    feature_channels: tuple[int, ...] = (64, 128)
    kernel_sizes: tuple[int, ...] = (10, 3)
    strides: tuple[int, ...] = (5, 2)
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512


class Wav2Vec2(AudioModel):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1000,
        d_model: int = 256,
        dropout: float = 0.1,
        feature_channels: tuple[int, ...] = (64, 128),
        kernel_sizes: tuple[int, ...] = (10, 3),
        strides: tuple[int, ...] = (5, 2),
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
    ):
        super().__init__()
        if not (len(feature_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("feature_channels, kernel_sizes, and strides must have the same length")
        if min(input_dim, output_dim, d_model, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("Wav2Vec2 dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._config = Wav2Vec2Config(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            feature_channels=tuple(feature_channels),
            kernel_sizes=tuple(kernel_sizes),
            strides=tuple(strides),
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
        )
        conv_layers: list[nn.Module] = []
        current = input_dim
        for channels, kernel, stride in zip(feature_channels, kernel_sizes, strides, strict=True):
            conv_layers.extend(
                [
                    nn.Conv1d(current, channels, kernel_size=kernel, stride=stride),
                    nn.GroupNorm(1, channels),
                    nn.GELU(),
                ]
            )
            current = channels
        self.feature_encoder = nn.Sequential(*conv_layers)
        self.feature_projection = nn.Linear(current, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.output_projection = nn.Linear(d_model, output_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != self.input_dim:
            raise ValueError(f"Wav2Vec2 expects x with shape (batch, time) or (batch, {self.input_dim}, time)")
        features = self.feature_encoder(x.float()).transpose(1, 2)
        return self.feature_projection(features)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        del lengths
        return self.encoder(self.extract_features(x))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.output_projection(self.encode(x, lengths=lengths))
