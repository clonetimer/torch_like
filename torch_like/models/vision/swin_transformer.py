"""Swin Transformer image classification backbone.

Framework-oriented implementation of "Swin Transformer: Hierarchical Vision
Transformer using Shifted Windows". It keeps the core local-window attention,
shifted-window mixing, and hierarchical patch merging pattern in a compact
PyTorch-native module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.vision.common import VisionConfig, VisionModel


@dataclass
class SwinTransformerConfig(VisionConfig):
    patch_size: int = 4
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    use_shift: bool = True


def _pad_nhwc(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    _, height, width, _ = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, channels)


def _window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int, batch: int) -> torch.Tensor:
    channels = windows.size(-1)
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, channels)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in the range [0, window_size)")
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, orig_h, orig_w, _ = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x, pad_h, pad_w = _pad_nhwc(x, self.window_size)
        _, height, width, _ = x.shape
        windows = _window_partition(x, self.window_size)
        windows, _ = self.attn(windows, windows, windows, need_weights=False)
        x = _window_reverse(windows, self.window_size, height, width, batch)
        if pad_h or pad_w:
            x = x[:, :orig_h, :orig_w, :].contiguous()
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + self.drop(x)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, height, width, _ = x.shape
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        use_shift: bool,
        downsample_dim: int | None = None,
    ):
        super().__init__()
        blocks = []
        for idx in range(depth):
            shift_size = window_size // 2 if use_shift and idx % 2 == 1 else 0
            blocks.append(SwinBlock(dim, num_heads, window_size, shift_size, mlp_ratio, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.downsample = PatchMerging(dim, downsample_dim) if downsample_dim is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.downsample(x)


class SwinTransformer(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 96,
        dropout: float = 0.0,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        use_shift: bool = True,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if any(depth <= 0 for depth in depths):
            raise ValueError("depths must contain positive values")
        if any(head <= 0 for head in num_heads):
            raise ValueError("num_heads must contain positive values")
        if patch_size <= 0 or embed_dim <= 0 or window_size <= 0:
            raise ValueError("patch_size, embed_dim, and window_size must be positive")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self._config = SwinTransformerConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            use_shift=use_shift,
        )

        dims = [embed_dim * (2**idx) for idx in range(len(depths))]
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        stages = []
        for idx, depth in enumerate(depths):
            downsample_dim = dims[idx + 1] if idx < len(depths) - 1 else None
            stages.append(
                SwinStage(
                    dims[idx],
                    depth,
                    num_heads[idx],
                    window_size,
                    mlp_ratio,
                    dropout,
                    use_shift,
                    downsample_dim=downsample_dim,
                )
            )
        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self.feature_dim = dims[-1]
        self._init_weights()

    @classmethod
    def swin_tiny(cls, **kwargs) -> "SwinTransformer":
        return cls(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)

    @classmethod
    def swin_small(cls, **kwargs) -> "SwinTransformer":
        return cls(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("SwinTransformer expects x with shape (batch, channels, height, width)")
        if x.size(-2) != self.image_size or x.size(-1) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}, got {x.size(-2)}x{x.size(-1)}")
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.norm(x)
        return x.mean(dim=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
