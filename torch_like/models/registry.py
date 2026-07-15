"""Model registry, metadata validation, and discovery utilities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module
from threading import RLock
from typing import Any, TypedDict


class ModelMetadata(TypedDict):
    """Normalized metadata stored for every registered model."""

    cls: type[Any]
    family: str
    layer: str
    integration_level: str
    supports: list[str]
    paper: str
    description: str
    complexity: str
    automl_default: bool


ModelCatalog = dict[str, ModelMetadata]

_DOMAIN_MODULES = {
    "forecasting": "torch_like.models.forecasting",
    "diagnosis": "torch_like.models.diagnosis",
    "detection": "torch_like.models.detection",
    "graph": "torch_like.models.graph",
    "rul": "torch_like.models.rul_catalog",
    "sequence": "torch_like.models.sequence",
    "vision": "torch_like.models.vision",
    "multimodal": "torch_like.models.multimodal",
    "generative": "torch_like.models.generative",
    "tabular": "torch_like.models.tabular",
    "recommendation": "torch_like.models.recommendation",
    "audio": "torch_like.models.audio",
    "reinforcement": "torch_like.models.reinforcement",
}
MODEL_DOMAINS = tuple(_DOMAIN_MODULES)
MODEL_ZOO: dict[str, ModelCatalog] = {domain: {} for domain in MODEL_DOMAINS}

_METADATA_DEFAULTS = {
    "family": "custom",
    "layer": "Custom models",
    "integration_level": "user_registered",
    "supports": [],
    "paper": "Not specified",
    "description": "User-registered model",
    "complexity": "Not specified",
}
_METADATA_FIELDS = frozenset(_METADATA_DEFAULTS)
_METADATA_OPTIONAL_FIELDS = frozenset({"automl_default"})
_METADATA_ALLOWED_FIELDS = _METADATA_FIELDS | _METADATA_OPTIONAL_FIELDS
_REGISTRY_LOCK = RLock()
_BOOTSTRAPPED = False


def _validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _normalize_metadata(
    cls: type[Any],
    metadata: Mapping[str, Any],
    *,
    require_complete: bool,
) -> ModelMetadata:
    if not isinstance(cls, type):
        raise TypeError("cls must be a class")

    unknown = set(metadata) - _METADATA_ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Unknown model metadata fields: {sorted(unknown)}")
    if require_complete:
        missing = _METADATA_FIELDS - set(metadata)
        if missing:
            raise ValueError(f"Missing model metadata fields: {sorted(missing)}")

    normalized = {**_METADATA_DEFAULTS, **metadata}
    for field in _METADATA_FIELDS - {"supports"}:
        _validate_identifier(normalized[field], field)

    supports = normalized["supports"]
    if not isinstance(supports, (list, tuple)):
        raise TypeError("supports must be a list or tuple of strings")
    if any(not isinstance(item, str) or not item.strip() for item in supports):
        raise ValueError("supports must contain only non-empty strings")
    if len(set(supports)) != len(supports):
        raise ValueError("supports must not contain duplicates")
    automl_default = metadata.get("automl_default", True)
    if not isinstance(automl_default, bool):
        raise TypeError("automl_default must be a bool")

    return ModelMetadata(
        cls=cls,
        family=normalized["family"],
        layer=normalized["layer"],
        integration_level=normalized["integration_level"],
        supports=list(supports),
        paper=normalized["paper"],
        description=normalized["description"],
        complexity=normalized["complexity"],
        automl_default=automl_default,
    )


def _copy_metadata(metadata: ModelMetadata) -> ModelMetadata:
    copied = dict(metadata)
    copied["supports"] = list(metadata["supports"])
    return copied  # type: ignore[return-value]


def _find_conflicting_domains(domain: str, name: str) -> list[str]:
    return [other for other, models in MODEL_ZOO.items() if other != domain and name in models]


def register_model(domain: str, name: str, cls: type[Any], **meta: Any) -> type[Any]:
    """Register a model class, filling standard metadata defaults when omitted."""

    _validate_identifier(domain, "domain")
    _validate_identifier(name, "name")
    candidate = _normalize_metadata(cls, meta, require_complete=False)

    with _REGISTRY_LOCK:
        domain_models = MODEL_ZOO.setdefault(domain, {})
        existing = domain_models.get(name)
        if existing is not None:
            if existing == candidate:
                return cls
            raise ValueError(f"Model '{name}' is already registered in domain '{domain}'")

        conflicting_domains = _find_conflicting_domains(domain, name)
        if conflicting_domains:
            raise ValueError(
                f"Model '{name}' is already registered in domain(s) {conflicting_domains}; "
                "model names must be globally unique"
            )
        domain_models[name] = candidate
    return cls


def register_model_catalog(domain: str, catalog: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    """Validate and atomically register a complete domain catalog."""

    _validate_identifier(domain, "domain")
    if not isinstance(catalog, Mapping):
        raise TypeError("catalog must be a mapping")

    normalized: ModelCatalog = {}
    for name, config in catalog.items():
        _validate_identifier(name, "model name")
        if not isinstance(config, Mapping):
            raise TypeError(f"Catalog entry '{name}' must be a mapping")
        if "cls" not in config:
            raise ValueError(f"Catalog entry '{name}' is missing 'cls'")
        normalized[name] = _normalize_metadata(
            config["cls"],
            {key: value for key, value in config.items() if key != "cls"},
            require_complete=True,
        )

    with _REGISTRY_LOCK:
        domain_models = MODEL_ZOO.setdefault(domain, {})
        for name, candidate in normalized.items():
            existing = domain_models.get(name)
            if existing is not None and existing != candidate:
                raise ValueError(f"Model '{name}' is already registered in domain '{domain}'")
            conflicting_domains = _find_conflicting_domains(domain, name)
            if conflicting_domains:
                raise ValueError(
                    f"Model '{name}' is already registered in domain(s) {conflicting_domains}; "
                    "model names must be globally unique"
                )
        domain_models.update(normalized)
    return catalog


def ensure_model_registrations() -> None:
    """Import built-in domain catalogs once, preserving lazy package imports."""

    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    with _REGISTRY_LOCK:
        if _BOOTSTRAPPED:
            return
        for module_name in _DOMAIN_MODULES.values():
            import_module(module_name)
        _BOOTSTRAPPED = True


class ModelZooView(Mapping[str, ModelCatalog]):
    """Lazy mapping view over the model registry.

    Importing :mod:`torch_like` remains cheap, while the first mapping operation
    guarantees that all built-in domain catalogs are visible.
    """

    def __getitem__(self, domain: str) -> ModelCatalog:
        ensure_model_registrations()
        return MODEL_ZOO[domain]

    def __iter__(self) -> Iterator[str]:
        ensure_model_registrations()
        return iter(MODEL_ZOO)

    def __len__(self) -> int:
        ensure_model_registrations()
        return len(MODEL_ZOO)

    def __repr__(self) -> str:
        ensure_model_registrations()
        return repr(MODEL_ZOO)


model_zoo = ModelZooView()


def _require_domain(domain: str) -> ModelCatalog:
    _validate_identifier(domain, "domain")
    ensure_model_registrations()
    try:
        return MODEL_ZOO[domain]
    except KeyError as exc:
        available = ", ".join(MODEL_ZOO)
        raise KeyError(f"Unknown model domain '{domain}'. Available: {available}") from exc


def list_domains() -> tuple[str, ...]:
    """Return all registered model domains in stable registration order."""

    ensure_model_registrations()
    return tuple(MODEL_ZOO)


def list_models(domain: str | None = None) -> ModelCatalog:
    """Return a defensive metadata snapshot for one domain or the full registry."""

    ensure_model_registrations()
    if domain is not None:
        return {name: _copy_metadata(metadata) for name, metadata in _require_domain(domain).items()}

    flat: ModelCatalog = {}
    for registered_domain, domain_models in MODEL_ZOO.items():
        duplicate_names = flat.keys() & domain_models.keys()
        if duplicate_names:
            raise RuntimeError(f"Cross-domain model name collision in '{registered_domain}': {sorted(duplicate_names)}")
        flat.update({name: _copy_metadata(metadata) for name, metadata in domain_models.items()})
    return flat


def get_model_info(name: str, domain: str = "forecasting") -> ModelMetadata:
    """Return a defensive copy of one model's catalog metadata."""

    _validate_identifier(name, "name")
    models = _require_domain(domain)
    try:
        return _copy_metadata(models[name])
    except KeyError as exc:
        available = ", ".join(models)
        raise KeyError(f"Model '{name}' not found in domain '{domain}'. Available: {available}") from exc


def list_models_by_support(support: str, domain: str = "forecasting") -> ModelCatalog:
    """Return models in *domain* advertising a capability tag."""

    _validate_identifier(support, "support")
    return {name: meta for name, meta in list_models(domain).items() if support in meta["supports"]}


def list_models_by_family(family: str, domain: str = "forecasting") -> ModelCatalog:
    """Return models in *domain* belonging to a model family."""

    _validate_identifier(family, "family")
    return {name: meta for name, meta in list_models(domain).items() if meta["family"] == family}


def create_model(name: str, domain: str = "forecasting", **kwargs: Any) -> Any:
    """Instantiate a registered model by name."""

    return get_model_info(name, domain)["cls"](**kwargs)
