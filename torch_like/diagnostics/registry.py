"""Diagnostic algorithm factory registry preserving each backend's native object model."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from torch_like.diagnostics.types import DiagnosticAlgorithmSpec, DiagnosticTask


class DiagnosticRegistry:
    """Store capability metadata and instantiate native backend objects."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._specs: dict[str, DiagnosticAlgorithmSpec] = {}
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, spec: DiagnosticAlgorithmSpec, factory: Callable[..., Any]) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable")
        with self._lock:
            if spec.name in self._specs:
                raise ValueError(f"diagnostic algorithm {spec.name!r} is already registered")
            self._specs[spec.name] = spec
            self._factories[spec.name] = factory

    def get_spec(self, name: str) -> DiagnosticAlgorithmSpec:
        with self._lock:
            try:
                return self._specs[name]
            except KeyError as exc:
                raise KeyError(f"unknown diagnostic algorithm {name!r}") from exc

    def list_specs(
        self, *, task: DiagnosticTask | str | None = None, backend: str | None = None
    ) -> dict[str, DiagnosticAlgorithmSpec]:
        task_filter = DiagnosticTask(task) if task is not None else None
        with self._lock:
            return {
                name: spec
                for name, spec in self._specs.items()
                if (task_filter is None or spec.task == task_filter) and (backend is None or spec.backend == backend)
            }

    def create(self, name: str, **kwargs: Any) -> Any:
        with self._lock:
            try:
                factory = self._factories[name]
            except KeyError as exc:
                raise KeyError(f"unknown diagnostic algorithm {name!r}") from exc
        return factory(**kwargs)


diagnostic_registry = DiagnosticRegistry()


def register_diagnostic_algorithm(spec: DiagnosticAlgorithmSpec, factory: Callable[..., Any]) -> None:
    diagnostic_registry.register(spec, factory)


def list_diagnostic_algorithms(
    *, task: DiagnosticTask | str | None = None, backend: str | None = None
) -> dict[str, DiagnosticAlgorithmSpec]:
    return diagnostic_registry.list_specs(task=task, backend=backend)


def get_diagnostic_algorithm_info(name: str) -> DiagnosticAlgorithmSpec:
    return diagnostic_registry.get_spec(name)


def create_diagnostic_algorithm(name: str, **kwargs: Any) -> Any:
    return diagnostic_registry.create(name, **kwargs)
