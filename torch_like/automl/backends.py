"""Backend registry: decouples candidate backends from orchestration code.

Backends are registered entries carrying a ``kind`` ("sklearn" / "baseline" /
"torch"), a ``model_family`` ("torch_like" / "baseline"), and -- once attached --
the ``runner`` / ``cv_runner`` callables that actually execute a trial. The
orchestrator dispatches purely through :func:`get_backend`, so a third party can
add a trainer without editing core files. Registration is module-level, so it is
safe inside worker subprocesses (the builtins are present before any task runs).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

from torch_like.core import Scope

Runner = Callable[..., Any]
CvRunner = Callable[..., Any]
_UNSET = object()


@dataclass(frozen=True)
class BackendSpec:
    """Describe one candidate execution backend.

    ``runner`` / ``cv_runner`` are attached after registration via
    :func:`attach_runner`, so this module never has to import the training
    libraries -- registration stays cheap and cycle-free inside workers.
    """

    name: str
    kind: str  # "sklearn" | "baseline" | "torch"
    model_family: str  # "torch_like" | "baseline"
    optional_dependency: str | None = None
    runner: Runner | None = None
    cv_runner: CvRunner | None = None
    tracks_epochs: bool = False
    detection_family: bool = False


_BACKENDS: dict[str, BackendSpec] = {}
_LOCK = RLock()

_VALID_KINDS = frozenset({"sklearn", "baseline", "torch"})
_VALID_FAMILIES = frozenset({"torch_like", "baseline"})


def register_backend(spec: BackendSpec, scope: "Scope | None" = None) -> BackendSpec:
    """Register a backend; raises on invalid fields or conflicting re-registration.

    When ``scope`` is provided the registration is bound to that disposable
    :class:`~torch_like.core.scope.Scope`: disposing it removes this entry (unless
    another registration replaced it first). This is the Cordis "registrations are
    effects" model, letting a plugin unregister cleanly on teardown.
    """
    if not spec.name:
        raise ValueError("backend name must be non-empty")
    if spec.kind not in _VALID_KINDS:
        raise ValueError(f"unknown backend kind: {spec.kind!r}")
    if spec.model_family not in _VALID_FAMILIES:
        raise ValueError(f"unknown model family: {spec.model_family!r}")
    with _LOCK:
        if spec.name in _BACKENDS and _BACKENDS[spec.name] != spec:
            raise ValueError(f"backend {spec.name!r} is already registered")
        _BACKENDS[spec.name] = spec
    if scope is not None:

        def _dispose() -> None:
            if _BACKENDS.get(spec.name) is spec:
                del _BACKENDS[spec.name]

        scope.add(_dispose)
    return spec


def attach_runner(
    name: str,
    *,
    runner: Any = _UNSET,
    cv_runner: Any = _UNSET,
    scope: "Scope | None" = None,
) -> BackendSpec:
    """Attach execution handlers to a registered backend, replacing its spec.

    Attaching is a Cordis-style effect: pass ``scope`` and disposing it restores
    the spec that was registered before, which lets a plugin (or a test) swap in
    fake runners and tear them down again.
    """
    with _LOCK:
        previous = get_backend(name)
        updated = replace(
            previous,
            runner=previous.runner if runner is _UNSET else runner,
            cv_runner=previous.cv_runner if cv_runner is _UNSET else cv_runner,
        )
        _BACKENDS[name] = updated
    if scope is not None:

        def _dispose() -> None:
            if _BACKENDS.get(name) is updated:
                _BACKENDS[name] = previous

        scope.add(_dispose)
    return updated


def get_backend(name: str) -> BackendSpec:
    try:
        return _BACKENDS[name]
    except KeyError:
        raise KeyError(f"unknown backend: {name!r}") from None


def has_backend(name: str) -> bool:
    return name in _BACKENDS


def suppress_backend(name: str, scope: "Scope | None" = None) -> BackendSpec | None:
    """Temporarily remove a registered backend; restored when ``scope`` is disposed.

    This is the inverse of :func:`register_backend` and the mechanism behind
    config-driven capability assembly: a run suppresses what its config leaves
    out, and disposing the run's scope puts the registry back as it was.
    """
    with _LOCK:
        previous = _BACKENDS.pop(name, None)
    if previous is None or scope is None:
        return previous

    def _restore() -> None:
        if name not in _BACKENDS:
            _BACKENDS[name] = previous

    scope.add(_restore)
    return previous


def list_backends() -> tuple[BackendSpec, ...]:
    return tuple(_BACKENDS.values())


def backends_by_kind(kind: str) -> tuple[BackendSpec, ...]:
    return tuple(spec for spec in _BACKENDS.values() if spec.kind == kind)


def _register_builtins() -> None:
    register_backend(BackendSpec("sklearn", "sklearn", "baseline"))
    register_backend(BackendSpec("baseline", "baseline", "baseline"))
    register_backend(BackendSpec("pytorch", "torch", "torch_like", tracks_epochs=True))
    register_backend(BackendSpec("anomaly_pytorch", "torch", "torch_like", detection_family=True))
    register_backend(BackendSpec("anomaly_baseline", "sklearn", "baseline"))


_register_builtins()


__all__ = [
    "BackendSpec",
    "CvRunner",
    "Runner",
    "attach_runner",
    "backends_by_kind",
    "get_backend",
    "has_backend",
    "list_backends",
    "register_backend",
    "suppress_backend",
]
