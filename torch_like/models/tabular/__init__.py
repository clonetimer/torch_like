"""Lazy public facade for the ``tabular`` model domain."""

from __future__ import annotations

from torch_like.models._domain_exports import DOMAIN_ALL
from torch_like.models._lazy import resolve_domain_export

_DOMAIN = 'tabular'
__all__ = list(DOMAIN_ALL[_DOMAIN])


def __getattr__(name: str):
    try:
        return resolve_domain_export(_DOMAIN, globals(), name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
