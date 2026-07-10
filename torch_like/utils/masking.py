"""Masking utilities for attention mechanisms."""

import torch


def triangular_causal_mask(shape, device="cpu"):
    """Create a triangular causal mask of shape (B, L) for self-attention.

    Equivalent to `torch.triu(torch.ones(L, L), diagonal=1).bool()` expanded
    and transposed for attention mask format (B, 1, L, L).

    Args:
        shape: (B, L) or (L,) tuple.
        device: target device.

    Returns:
        bool mask of shape (B, 1, L, L), True = attend, False = mask.
    """
    if isinstance(shape, tuple) and len(shape) == 2:
        B, L = shape
    else:
        B, L = 1, shape[0]
    mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    return ~mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)


def prob_mask(shape, p=0.15, device="cpu"):
    """Random mask for data augmentation.

    Args:
        shape: (B, L) tuple.
        p: mask probability.
        device: target device.

    Returns:
        bool mask, True = keep, False = mask out.
    """
    return torch.rand(shape, device=device) > p
