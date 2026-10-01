"""Result types produced by the public training runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    validation_loss: float | None
    learning_rate: float
    max_gradient_norm: float
    improved: bool
    validation_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    epochs_requested: int
    epochs_ran: int
    start_epoch: int
    best_epoch: int
    best_monitored_loss: float
    stopped_early: bool
    monitor: str
    device: str
    history: tuple[EpochResult, ...]
    last_checkpoint: str | None = None
    best_checkpoint: str | None = None
    resumed_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["history"] = [item.to_dict() for item in self.history]
        return payload


__all__ = ["EpochResult", "TrainingResult"]
