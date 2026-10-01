"""Public model namespace with lazy implementation imports and registry access."""

from __future__ import annotations

from torch_like.models._lazy import resolve_export
from torch_like.models._public_exports import PUBLIC_EXPORTS
from torch_like.models.catalog_evidence import get_evidence_record, list_evidence_records
from torch_like.models.registry import (
    MODEL_DOMAINS,
    MODEL_ZOO,
    create_model,
    find_model_domain,
    get_model_class,
    get_model_info,
    list_domains,
    list_model_names,
    list_models,
    list_models_by_family,
    list_models_by_fidelity,
    list_models_by_maturity,
    list_models_by_support,
    model_zoo,
    register_model,
    register_model_catalog,
)

MODEL_EXPORTS = tuple(PUBLIC_EXPORTS)

__all__ = [
    "MODEL_DOMAINS",
    "MODEL_ZOO",
    "model_zoo",
    "register_model",
    "register_model_catalog",
    "list_domains",
    "list_model_names",
    "list_models",
    "list_models_by_support",
    "list_models_by_family",
    "list_models_by_fidelity",
    "list_models_by_maturity",
    "get_model_class",
    "get_model_info",
    "find_model_domain",
    "create_model",
    "get_evidence_record",
    "list_evidence_records",
    *MODEL_EXPORTS,
]


def __getattr__(name: str):
    try:
        return resolve_export(globals(), PUBLIC_EXPORTS, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'torch_like.models' has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
