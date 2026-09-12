"""Effect scopes: group registrations so they can be disposed together.

A :class:`Scope` records disposers (callables) in registration order and runs
them in reverse (LIFO) order on :meth:`Scope.dispose`. Scopes nest through a
``parent`` link, so disposing an outer scope disposes inner ones too. This is
the Cordis "registrations are effects" idea adapted to Python: every side
effect a plugin produces can be undone simply by disposing the scope that
hosted the plugin.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

Disposer = Callable[[], None]


class ScopeDisposeError(RuntimeError):
    """Raised when one or more disposers raised during :meth:`Scope.dispose`."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        joined = "; ".join(f"{type(e).__name__}: {e}" for e in errors)
        super().__init__(f"{len(errors)} error(s) while disposing scope: {joined}")


class Scope:
    """A disposable container for side effects.

    Thread-safe. Registrations after :meth:`dispose` are rejected so callers
    cannot silently leak effects onto a torn-down context.
    """

    __slots__ = ("_name", "_parent", "_lock", "_disposers", "_disposed")

    def __init__(self, name: str = "scope", parent: "Scope | None" = None) -> None:
        self._name = name
        self._parent = parent
        self._lock = RLock()
        self._disposers: list[Disposer] = []
        self._disposed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent(self) -> "Scope | None":
        return self._parent

    @property
    def active(self) -> bool:
        return not self._disposed

    def add(self, disposer: Disposer) -> Disposer:
        """Register a disposer; returns it unchanged so it can be chained."""
        if not callable(disposer):
            raise TypeError("disposer must be callable")
        with self._lock:
            if self._disposed:
                raise RuntimeError(f"cannot add disposer to disposed scope {self._name!r}")
            self._disposers.append(disposer)
        return disposer

    def dispose(self) -> None:
        """Run all disposers in reverse registration order."""
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            disposers = list(reversed(self._disposers))
            self._disposers.clear()
        errors: list[BaseException] = []
        for disposer in disposers:
            try:
                disposer()
            except BaseException as exc:  # noqa: BLE001 - aggregate, then raise once
                errors.append(exc)
        if errors:
            raise ScopeDisposeError(errors)

    def child(self, name: str) -> "Scope":
        """Create a nested scope whose effects roll up to this one."""
        return Scope(name, parent=self)

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.dispose()
