"""Audio Spectrogram Transformer-style classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.audio.common import AudioConfig, AudioModel


@dataclass
class AudioSpectrogramTransformerConfig(AudioConfig):
    freq_bins: int = 128
    time_frames: int = 128
    patch_size: tuple[int, int] = (16, 16)
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512


class AudioSpectrogramTransformer(AudioModel):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 527,
        d_model: int = 256,
        dropout: float = 0.1,
        freq_bins: int = 128,
        time_frames: int = 128,
        patch_size: tuple[int, int] = (16, 16),
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
    ):
        super().__init__()
        if min(input_dim, output_dim, d_model, freq_bins, time_frames, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("AudioSpectrogramTransformer dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        patch_f, patch_t = patch_size
        if freq_bins % patch_f != 0 or time_frames % patch_t != 0:
            raise ValueError("freq_bins and time_frames must be divisible by patch_size")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.patch_size = patch_size
        self._config = AudioSpectrogramTransformerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            freq_bins=freq_bins,
            time_frames=time_frames,
            patch_size=patch_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
        )
        self.patch_embed = nn.Conv2d(input_dim, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (freq_bins // patch_f) * (time_frames // patch_t)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1:] != (self.input_dim, self.freq_bins, self.time_frames):
            raise ValueError(
                f"AudioSpectrogramTransformer expects x with shape (batch, {self.input_dim}, "
                f"{self.freq_bins}, {self.time_frames})"
            )
        tokens = self.patch_embed(x.float()).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position_embedding
        return self.encoder(tokens)[:, 0]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        del lengths
        return self.head(self.forward_features(x))
