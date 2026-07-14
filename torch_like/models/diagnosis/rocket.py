"""ROCKET-style random convolutional features for time-series classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.diagnosis.common import DiagnosisConfig, DiagnosisModel


@dataclass
class ROCKETConfig(DiagnosisConfig):
    num_kernels: int = 256
    kernel_sizes: tuple[int, ...] = (7, 9, 11)


class ROCKET(DiagnosisModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 256,
        dropout: float = 0.1,
        num_kernels: int = 256,
        kernel_sizes: tuple[int, ...] = (7, 9, 11),
    ):
        del d_model
        super().__init__()
        if min(input_dim, num_classes, seq_len, num_kernels) <= 0:
            raise ValueError("input_dim, num_classes, seq_len, and num_kernels must be positive")
        if not kernel_sizes or any(size <= 0 or size % 2 == 0 for size in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_kernels = num_kernels
        self.kernel_sizes = tuple(kernel_sizes)
        self._config = ROCKETConfig(
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=seq_len,
            dropout=dropout,
            num_kernels=num_kernels,
            kernel_sizes=tuple(kernel_sizes),
        )
        max_kernel = max(kernel_sizes)
        weights = torch.zeros(num_kernels, input_dim, max_kernel)
        bias = torch.empty(num_kernels).uniform_(-1.0, 1.0)
        dilations = torch.empty(num_kernels, dtype=torch.long)
        paddings = torch.empty(num_kernels, dtype=torch.long)
        for idx in range(num_kernels):
            kernel = kernel_sizes[idx % len(kernel_sizes)]
            weight = torch.randn(input_dim, kernel)
            weight = weight - weight.mean(dim=-1, keepdim=True)
            weights[idx, :, :kernel] = weight
            max_dilation = max(1, (seq_len - 1) // max(kernel - 1, 1))
            dilation = int(torch.randint(1, max_dilation + 1, (1,)).item())
            dilations[idx] = dilation
            paddings[idx] = ((kernel - 1) * dilation) // 2 if idx % 2 == 0 else 0
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)
        self.register_buffer("dilations", dilations)
        self.register_buffer("paddings", paddings)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * num_kernels, num_classes)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(f"ROCKET expects x with shape (batch, {self.seq_len}, {self.input_dim})")
        x_t = x.transpose(1, 2)
        features = []
        for idx in range(self.num_kernels):
            kernel = self.kernel_sizes[idx % len(self.kernel_sizes)]
            weight = self.weights[idx : idx + 1, :, :kernel]
            conv = torch.conv1d(
                x_t,
                weight,
                bias=self.bias[idx : idx + 1],
                dilation=int(self.dilations[idx].item()),
                padding=int(self.paddings[idx].item()),
            )
            max_feature = conv.amax(dim=-1)
            ppv_feature = (conv > 0).to(dtype=x.dtype).mean(dim=-1)
            features.extend([max_feature, ppv_feature])
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        return self.classifier(self.dropout(self.transform(x)))
