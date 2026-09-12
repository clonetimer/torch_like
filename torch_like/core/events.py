"""Type-agnostic event bus with broadcast and waterfall semantics.

* :meth:`EventEmitter.emit` notifies every subscriber with the same arguments
  (fan-out broadcast).
* :meth:`EventEmitter.waterfall` threads a value through subscribers in
  registration order; a subscriber returning ``None`` keeps the previous value
  and the chain continues (Cordis ``waterfall`` nullish behavior), otherwise its
  return becomes the next input.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any, TypeVar

from torch_like.core.scope import Scope

T = TypeVar("T")
Handler = Callable[..., Any]


class EventEmitter:
    """Fires events to subscribers; subscriptions are disposable."""

    def __init__(self, scope: Scope | None = None, name: str = "events") -> None:
        self._scope = scope if scope is not None else Scope(name)
        self._lock = RLock()
        self._handlers: list[Handler] = []

    @property
    def scope(self) -> Scope:
        return self._scope

    def on(self, handler: Handler) -> Callable[[], None]:
        """Register ``handler``; returns a disposer for unsubscription."""
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            self._handlers.append(handler)

        def _dispose() -> None:
            with self._lock:
                if handler in self._handlers:
                    self._handlers.remove(handler)

        self._scope.add(_dispose)
        return _dispose

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for handler in tuple(self._handlers):
            handler(*args, **kwargs)

    def waterfall(self, value: T, *args: Any, **kwargs: Any) -> T:
        current = value
        for handler in tuple(self._handlers):
            result = handler(current, *args, **kwargs)
            if result is not None:
                current = result
        return current
