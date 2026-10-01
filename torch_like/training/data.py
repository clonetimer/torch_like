"""Data helpers shared by the training runtime and task adapters."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch


def move_to_device(value: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    """Recursively move tensors in common batch containers to ``device``.

    Dataclass task batches are reconstructed with :func:`dataclasses.replace`,
    preserving the domain-specific batch type instead of flattening it.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: move_to_device(getattr(value, field.name), device, non_blocking=non_blocking)
            for field in fields(value)
        }
        return replace(value, **updates)
    if isinstance(value, dict):
        return {key: move_to_device(item, device, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking=non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in value]
    return value


def infer_batch_size(value: Any) -> int:
    """Infer the leading sample count from a nested batch.

    A clear error is preferred over silently treating every batch as size one,
    because incorrect weighting corrupts epoch loss for variable-size batches.
    """

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ValueError("cannot infer batch size from a scalar tensor")
        return int(value.shape[0])
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            try:
                return infer_batch_size(getattr(value, field.name))
            except ValueError:
                continue
        raise ValueError(f"cannot infer batch size from dataclass {type(value).__name__}")
    if isinstance(value, dict):
        for item in value.values():
            try:
                return infer_batch_size(item)
            except ValueError:
                continue
        raise ValueError("cannot infer batch size from mapping")
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return infer_batch_size(item)
            except ValueError:
                continue
        raise ValueError("cannot infer batch size from sequence")
    raise ValueError(f"cannot infer batch size from {type(value).__module__}.{type(value).__qualname__}")


__all__ = ["infer_batch_size", "move_to_device"]
