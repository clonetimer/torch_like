"""Backend-aware construction helpers."""

from __future__ import annotations

from typing import Any

from torch_like.classical import create_estimator, has_estimator
from torch_like.models import create_model, find_model_domain

_TORCH_BACKENDS = frozenset({"torch_like", "torch", "pytorch"})
_SKLEARN_BACKENDS = frozenset({"sklearn", "classical"})


def create(
    name: str,
    *,
    backend: str = "auto",
    domain: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a torch_like model or native sklearn estimator by name.

    ``backend='auto'`` resolves built-in torch_like model names first and then
    the curated sklearn catalog.  Returned objects keep their native APIs:
    torch_like models remain ``torch.nn.Module`` objects and sklearn estimators
    retain ``fit``/``predict`` semantics.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    normalized = backend.lower().strip()
    if normalized != "auto" and normalized not in _TORCH_BACKENDS | _SKLEARN_BACKENDS:
        raise ValueError("backend must be one of: auto, torch_like, torch, pytorch, sklearn, classical")

    if normalized in _SKLEARN_BACKENDS:
        if domain is not None:
            raise ValueError("domain is only valid for torch_like/PyTorch models")
        return create_estimator(name, **kwargs)

    if normalized in _TORCH_BACKENDS or domain is not None:
        resolved_domain = domain or find_model_domain(name)
        return create_model(name, domain=resolved_domain, **kwargs)

    try:
        resolved_domain = find_model_domain(name)
    except KeyError:
        resolved_domain = None
    if resolved_domain is not None:
        return create_model(name, domain=resolved_domain, **kwargs)
    if has_estimator(name):
        return create_estimator(name, **kwargs)
    raise KeyError(f"Unknown algorithm '{name}' in torch_like models and sklearn compatibility catalog")


__all__ = ["create"]
