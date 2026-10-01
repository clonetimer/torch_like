"""Internal helpers for zero-cost model namespace facades."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from torch_like.models._domain_exports import DOMAIN_EXPORTS


def resolve_export(namespace: dict[str, Any], exports: dict[str, tuple[str, str]], name: str) -> Any:
    target = exports.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    namespace[name] = value
    return value


def resolve_domain_export(domain: str, namespace: dict[str, Any], name: str) -> Any:
    if name == "model_zoo":
        from torch_like.models.registry import model_zoo

        value = model_zoo[domain]
        namespace[name] = value
        return value
    try:
        exports = DOMAIN_EXPORTS[domain]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return resolve_export(namespace, exports, name)
