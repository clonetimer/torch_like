"""HuBERT-style self-supervised speech representation model.

HuBERT predicts discrete hidden units from masked speech features. This module
implements the reusable model-side contract only: raw waveform feature
encoding, mask application, Transformer contextualization, hidden-unit logits,
and cross-entropy loss. Offline clustering, label generation, CTC decoding,
and pretrained checkpoint conversion stay outside the model class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.audio.common import AudioBatch, AudioConfig, AudioModel


@dataclass
class HuBERTConfig(AudioConfig):
    feature_channels: tuple[int, ...] = (64, 128)
    kernel_sizes: tuple[int, ...] = (10, 3)
    strides: tuple[int, ...] = (5, 2)
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    mask_prob: float = 0.065
    mask_length: int = 10


class HuBERT(AudioModel):
    """Compact HuBERT-style masked hidden-unit prediction model."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 100,
        d_model: int = 256,
        dropout: float = 0.1,
        feature_channels: tuple[int, ...] = (64, 128),
        kernel_sizes: tuple[int, ...] = (10, 3),
        strides: tuple[int, ...] = (5, 2),
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        mask_prob: float = 0.065,
        mask_length: int = 10,
    ):
        super().__init__()
        if not (len(feature_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("feature_channels, kernel_sizes, and strides must have the same length")
        if min(input_dim, output_dim, d_model, n_heads, n_layers, d_ff, mask_length) <= 0:
            raise ValueError("HuBERT dimensions, layer counts, and mask_length must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1]")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self._config = HuBERTConfig(
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
            mask_prob=mask_prob,
            mask_length=mask_length,
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
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
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
        self.unit_projection = nn.Linear(d_model, output_dim)
        self._init_weights()

    @classmethod
    def hubert_tiny(cls, **kwargs) -> "HuBERT":
        return cls(
            d_model=128,
            feature_channels=(64, 128),
            kernel_sizes=(10, 3),
            strides=(5, 2),
            n_heads=4,
            n_layers=2,
            d_ff=512,
            **kwargs,
        )

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != self.input_dim:
            raise ValueError(f"HuBERT expects x with shape (batch, time) or (batch, {self.input_dim}, time)")
        features = self.feature_encoder(x.float()).transpose(1, 2)
        return self.feature_projection(features)

    def build_mask(
        self,
        features: torch.Tensor,
        mask_prob: float | None = None,
        mask_length: int | None = None,
    ) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError("features must have shape (batch, frames, channels)")
        if mask_prob is None:
            mask_prob = self.mask_prob
        if mask_length is None:
            mask_length = self.mask_length
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1]")
        if mask_length <= 0:
            raise ValueError("mask_length must be positive")

        batch_size, frames, _ = features.shape
        starts = torch.rand(batch_size, frames, device=features.device) < mask_prob
        mask = torch.zeros(batch_size, frames, device=features.device, dtype=torch.bool)
        for offset in range(mask_length):
            if offset >= frames:
                break
            mask[:, offset:] |= starts[:, : frames - offset]
        return mask

    def mask_features(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = self.build_mask(features)
        if mask.shape != features.shape[:2]:
            raise ValueError("mask must have shape (batch, frames)")
        masked = features.clone()
        masked[mask] = self.mask_embedding.to(device=features.device, dtype=features.dtype)
        return masked, mask

    def encode(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del lengths
        features = self.extract_features(x)
        if mask is not None:
            features, _ = self.mask_features(features, mask)
        return self.encoder(features)

    def predict_units(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.unit_projection(self.encode(x, lengths=lengths, mask=mask))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.predict_units(x, lengths=lengths)

    def loss_step(self, batch: AudioBatch) -> torch.Tensor:
        if batch.y is None:
            raise ValueError("HuBERT loss_step requires hidden-unit labels in batch.y")
        features = self.extract_features(batch.x)
        mask = self.build_mask(features)
        encoded = self.encoder(self.mask_features(features, mask)[0])
        logits = self.unit_projection(encoded)
        labels = batch.y.long().to(logits.device)
        if labels.shape != logits.shape[:2]:
            raise ValueError("HuBERT labels must have shape (batch, frames)")
        if mask.any():
            return F.cross_entropy(logits[mask], labels[mask])
        return F.cross_entropy(logits.reshape(-1, self.output_dim), labels.reshape(-1))
