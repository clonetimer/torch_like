"""Native scikit-learn compatibility surface.

This namespace deliberately returns real scikit-learn estimator classes and
instances.  It provides a stable discovery/factory layer without pretending
that sklearn estimators are ``torch.nn.Module`` objects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TypedDict

from torch_like.classical._registry import ESTIMATOR_SPECS


class EstimatorInfo(TypedDict):
    name: str
    backend: str
    kind: str
    family: str
    class_path: str


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")


def has_estimator(name: str) -> bool:
    """Return whether *name* is in the curated sklearn compatibility catalog."""

    return name in ESTIMATOR_SPECS


def get_estimator_info(name: str) -> EstimatorInfo:
    """Return lightweight metadata without importing scikit-learn."""

    _validate_name(name)
    try:
        spec = ESTIMATOR_SPECS[name]
    except KeyError as exc:
        available = ", ".join(ESTIMATOR_SPECS)
        raise KeyError(f"Unknown sklearn estimator '{name}'. Available: {available}") from exc
    module = str(spec["module"])
    return EstimatorInfo(
        name=name,
        backend="sklearn",
        kind=str(spec["kind"]),
        family=str(spec["family"]),
        class_path=f"{module}.{name}",
    )


def list_estimators(kind: str | None = None) -> dict[str, EstimatorInfo]:
    """List curated sklearn estimators, optionally filtered by task kind."""

    if kind is not None and (not isinstance(kind, str) or not kind.strip()):
        raise ValueError("kind must be None or a non-empty string")
    result: dict[str, EstimatorInfo] = {}
    for name in ESTIMATOR_SPECS:
        info = get_estimator_info(name)
        if kind is None or info["kind"] == kind:
            result[name] = info
    return result


def get_estimator_class(name: str) -> type[Any]:
    """Resolve and return the original scikit-learn estimator class."""

    info = get_estimator_info(name)
    module_name, _, attr_name = info["class_path"].rpartition(".")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def create_estimator(name: str, **kwargs: Any) -> Any:
    """Instantiate a native scikit-learn estimator from the compatibility catalog."""

    return get_estimator_class(name)(**kwargs)


_ESTIMATOR_EXPORTS = tuple(ESTIMATOR_SPECS)
__all__ = [
    "EstimatorInfo",
    "has_estimator",
    "get_estimator_info",
    "list_estimators",
    "get_estimator_class",
    "create_estimator",
    *_ESTIMATOR_EXPORTS,
]


def __getattr__(name: str):
    if name in ESTIMATOR_SPECS:
        return get_estimator_class(name)
    raise AttributeError(f"module 'torch_like.classical' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
