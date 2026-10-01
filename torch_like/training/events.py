"""Lifecycle events emitted by the public training runtime."""

from torch_like.core.events import EventEmitter

training_events = EventEmitter(name="torch_training")

__all__ = ["training_events"]
