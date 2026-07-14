"""InceptionTime network for time-series classification and diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.diagnosis.common import DiagnosisConfig, DiagnosisModel


@dataclass
class InceptionTimeConfig(DiagnosisConfig):
    num_blocks: int = 6
    bottleneck_dim: int = 32
    kernel_sizes: tuple[int, ...] = (9, 19, 39)


class InceptionModule1D(nn.Module):
    def __init__(self, input_dim: int, branch_dim: int, bottleneck_dim: int, kernel_sizes: tuple[int, ...]):
        super().__init__()
        self.bottleneck = nn.Conv1d(input_dim, bottleneck_dim, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            [nn.Conv1d(bottleneck_dim, branch_dim, kernel, padding=kernel // 2, bias=False) for kernel in kernel_sizes]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_dim, branch_dim, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(branch_dim * (len(kernel_sizes) + 1))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        outputs = [branch(bottleneck) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        return self.activation(self.norm(torch.cat(outputs, dim=1)))


class InceptionTime(DiagnosisModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        dropout: float = 0.1,
        num_blocks: int = 6,
        bottleneck_dim: int = 32,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
    ):
        super().__init__()
        if not kernel_sizes or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers")
        if num_blocks <= 0 or d_model % (len(kernel_sizes) + 1) != 0:
            raise ValueError("num_blocks must be positive and d_model divisible by the number of branches")
        self.input_dim = input_dim
        self.seq_len = seq_len
        branch_dim = d_model // (len(kernel_sizes) + 1)
        self._config = InceptionTimeConfig(
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            num_blocks=num_blocks,
            bottleneck_dim=bottleneck_dim,
            kernel_sizes=tuple(kernel_sizes),
        )
        modules = []
        residuals = []
        current_dim = input_dim
        for index in range(num_blocks):
            modules.append(InceptionModule1D(current_dim, branch_dim, bottleneck_dim, tuple(kernel_sizes)))
            if index % 3 == 2:
                residual_dim = input_dim if index == 2 else d_model
                residuals.append(
                    nn.Sequential(nn.Conv1d(residual_dim, d_model, 1, bias=False), nn.BatchNorm1d(d_model))
                )
            current_dim = d_model
        self.modules_list = nn.ModuleList(modules)
        self.residuals = nn.ModuleList(residuals)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        if x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(f"InceptionTime expects x with shape (batch, {self.seq_len}, {self.input_dim})")
        hidden = x.transpose(1, 2)
        residual = hidden
        residual_index = 0
        for index, module in enumerate(self.modules_list):
            hidden = module(hidden)
            if index % 3 == 2:
                hidden = torch.relu(hidden + self.residuals[residual_index](residual))
                residual = hidden
                residual_index += 1
        return self.classifier(self.dropout(hidden.mean(dim=-1)))
