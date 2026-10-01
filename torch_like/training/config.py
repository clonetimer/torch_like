"""Configuration objects for the public PyTorch training runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainerConfig:
    """Stable configuration for :class:`torch_like.training.Trainer`.

    ``epochs`` is the total target epoch count.  When resuming a checkpoint
    written after epoch 3 with ``epochs=5``, the trainer executes epochs 4-5.
    """

    epochs: int = 1
    device: str = "auto"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler: str = "none"
    grad_clip_norm: float | None = None
    amp: bool = False
    early_stopping_patience: int | None = None
    min_delta: float = 0.0
    restore_best: bool = False
    checkpoint_dir: str | Path | None = None
    save_last: bool = True
    save_best: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer must be one of: adam, adamw, sgd")
        if self.lr_scheduler not in {"none", "cosine", "plateau"}:
            raise ValueError("lr_scheduler must be one of: none, cosine, plateau")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when provided")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when provided")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.checkpoint_dir is not None:
            payload["checkpoint_dir"] = str(self.checkpoint_dir)
        return payload


__all__ = ["TrainerConfig"]
