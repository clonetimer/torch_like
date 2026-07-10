"""Model registry utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict


MODEL_ZOO: Dict[str, Dict[str, Dict[str, Any]]] = {
    "forecasting": {},
    "diagnosis": {},
    "detection": {},
    "graph": {},
    "rul": {},
    "sequence": {},
    "vision": {},
    "multimodal": {},
    "generative": {},
}
_BOOTSTRAPPED = False


def register_model(domain: str, name: str, cls, **meta):
    domain_models = MODEL_ZOO.setdefault(domain, {})
    candidate = {"cls": cls, **meta}
    existing = domain_models.get(name)
    if existing is not None:
        if existing == candidate:
            return cls
        raise ValueError(f"Model '{name}' is already registered in domain '{domain}'")
    conflicting_domains = [other_domain for other_domain, models in MODEL_ZOO.items() if other_domain != domain and name in models]
    if conflicting_domains:
        raise ValueError(
            f"Model '{name}' is already registered in domain(s) {conflicting_domains}; "
            "model names must be globally unique"
        )
    domain_models[name] = candidate
    return cls


def register_model_catalog(domain: str, catalog: Dict[str, Dict[str, Any]]):
    for name, cfg in catalog.items():
        register_model(domain, name, cfg["cls"], **{k: v for k, v in cfg.items() if k != "cls"})
    return catalog


def ensure_model_registrations():
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for module_name in (
        "torch_like.models.forecasting",
        "torch_like.models.detection",
        "torch_like.models.graph",
        "torch_like.models.sequence",
        "torch_like.models.rul_catalog",
        "torch_like.models.diagnosis",
        "torch_like.models.vision",
        "torch_like.models.multimodal",
        "torch_like.models.generative",
    ):
        import_module(module_name)
    _BOOTSTRAPPED = True


def list_models(domain: str | None = None):
    ensure_model_registrations()
    if domain:
        return dict(MODEL_ZOO.get(domain, {}))
    flat = {}
    for registered_domain, domain_models in MODEL_ZOO.items():
        duplicate_names = flat.keys() & domain_models.keys()
        if duplicate_names:
            raise RuntimeError(f"Cross-domain model name collision in '{registered_domain}': {sorted(duplicate_names)}")
        flat.update(domain_models)
    return flat


def list_models_by_support(support: str, domain: str = "forecasting"):
    ensure_model_registrations()
    models = MODEL_ZOO.get(domain, {})
    return {name: cfg for name, cfg in models.items() if support in cfg.get("supports", [])}


def list_models_by_family(family: str, domain: str = "forecasting"):
    ensure_model_registrations()
    models = MODEL_ZOO.get(domain, {})
    return {name: cfg for name, cfg in models.items() if cfg.get("family") == family}


def create_model(name: str, domain: str = "forecasting", **kwargs):
    ensure_model_registrations()
    cfg = MODEL_ZOO.get(domain, {}).get(name)
    if cfg is None:
        raise KeyError(
            f"Model '{name}' not found in domain '{domain}'. "
            f"Available: {list(MODEL_ZOO.get(domain, {}).keys())}"
        )
    return cfg["cls"](**kwargs)
