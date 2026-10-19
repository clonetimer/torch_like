"""Model registry, metadata validation, and discovery utilities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module
from threading import RLock
from typing import Any, TypedDict

from torch_like.models._builtin_index import BUILTIN_MODEL_INDEX
from torch_like.models.catalog_evidence import evidence_metadata_for_model
from torch_like.models.catalog_schema import DERIVED_METADATA_FIELDS, derive_catalog_v2_metadata


class ModelMetadata(TypedDict):
    """Normalized metadata stored for every registered model."""

    cls: type[Any]
    family: str
    layer: str
    integration_level: str
    implementation_fidelity: str
    implementation_scope: str
    execution_profile: str
    maturity: str
    benchmark_evidence: str
    evidence_record: str | None
    supports: list[str]
    paper: str
    description: str
    complexity: str
    automl_default: bool


ModelCatalog = dict[str, ModelMetadata]

_DOMAIN_MODULES = {
    "forecasting": "torch_like.models.forecasting._catalog",
    "diagnosis": "torch_like.models.diagnosis._catalog",
    "detection": "torch_like.models.detection._catalog",
    "graph": "torch_like.models.graph._catalog",
    "rul": "torch_like.models.rul_catalog",
    "sequence": "torch_like.models.sequence._catalog",
    "vision": "torch_like.models.vision._catalog",
    "multimodal": "torch_like.models.multimodal._catalog",
    "generative": "torch_like.models.generative._catalog",
    "tabular": "torch_like.models.tabular._catalog",
    "recommendation": "torch_like.models.recommendation._catalog",
    "audio": "torch_like.models.audio._catalog",
    "reinforcement": "torch_like.models.reinforcement._catalog",
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
_METADATA_OPTIONAL_FIELDS = frozenset({"automl_default", *DERIVED_METADATA_FIELDS})
_METADATA_ALLOWED_FIELDS = _METADATA_FIELDS | _METADATA_OPTIONAL_FIELDS
_REGISTRY_LOCK = RLock()
_BOOTSTRAPPED_DOMAINS: set[str] = set()


def _validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _normalize_metadata(
    cls: type[Any],
    metadata: Mapping[str, Any],
    *,
    require_complete: bool,
    evidence_metadata: Mapping[str, Any] | None = None,
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

    evidence_metadata = evidence_metadata or {}
    conflicts = {field for field, value in evidence_metadata.items() if field in metadata and metadata[field] != value}
    if conflicts:
        raise ValueError(f"Catalog metadata conflicts with audited evidence fields: {sorted(conflicts)}")
    effective_metadata = {**metadata, **evidence_metadata}
    normalized = {**_METADATA_DEFAULTS, **effective_metadata}
    for field in _METADATA_FIELDS - {"supports"}:
        _validate_identifier(normalized[field], field)

    supports = normalized["supports"]
    if not isinstance(supports, (list, tuple)):
        raise TypeError("supports must be a list or tuple of strings")
    if any(not isinstance(item, str) or not item.strip() for item in supports):
        raise ValueError("supports must contain only non-empty strings")
    if len(set(supports)) != len(supports):
        raise ValueError("supports must not contain duplicates")
    automl_default = effective_metadata.get("automl_default", True)
    if not isinstance(automl_default, bool):
        raise TypeError("automl_default must be a bool")
    v2_metadata = derive_catalog_v2_metadata(normalized["integration_level"], effective_metadata)

    return ModelMetadata(
        cls=cls,
        family=normalized["family"],
        layer=normalized["layer"],
        integration_level=normalized["integration_level"],
        **v2_metadata,
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
    conflicts = [other for other, models in MODEL_ZOO.items() if other != domain and name in models]
    builtin = BUILTIN_MODEL_INDEX.get(name)
    if builtin is not None:
        builtin_domain = builtin[0]
        if builtin_domain != domain and builtin_domain not in conflicts:
            conflicts.append(builtin_domain)
    return conflicts


def register_model(domain: str, name: str, cls: type[Any], **meta: Any) -> type[Any]:
    """Register a model class, filling standard metadata defaults when omitted."""

    _validate_identifier(domain, "domain")
    _validate_identifier(name, "name")
    builtin = BUILTIN_MODEL_INDEX.get(name)
    if builtin is not None and builtin[0] == domain:
        ensure_domain_registration(domain)
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
            evidence_metadata=evidence_metadata_for_model(domain, name),
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


def ensure_domain_registration(domain: str) -> None:
    """Load one built-in domain catalog without importing unrelated domains."""

    module_name = _DOMAIN_MODULES.get(domain)
    if module_name is None or domain in _BOOTSTRAPPED_DOMAINS:
        return
    with _REGISTRY_LOCK:
        if domain in _BOOTSTRAPPED_DOMAINS:
            return
        import_module(module_name)
        _BOOTSTRAPPED_DOMAINS.add(domain)


def ensure_model_registrations() -> None:
    """Load every built-in catalog.

    Most callers should prefer :func:`ensure_domain_registration`; this function
    remains for full-catalog operations and backwards compatibility.
    """

    for domain in MODEL_DOMAINS:
        ensure_domain_registration(domain)


class ModelZooView(Mapping[str, ModelCatalog]):
    """Lazy mapping view over the model registry.

    Importing :mod:`torch_like` remains cheap, while the first mapping operation
    guarantees that all built-in domain catalogs are visible.
    """

    def __getitem__(self, domain: str) -> ModelCatalog:
        ensure_domain_registration(domain)
        return MODEL_ZOO[domain]

    def __iter__(self) -> Iterator[str]:
        return iter(MODEL_ZOO)

    def __len__(self) -> int:
        return len(MODEL_ZOO)

    def __repr__(self) -> str:
        loaded = sorted(_BOOTSTRAPPED_DOMAINS)
        return f"ModelZooView(domains={list(MODEL_ZOO)!r}, loaded={loaded!r})"


model_zoo = ModelZooView()


def _require_domain(domain: str) -> ModelCatalog:
    _validate_identifier(domain, "domain")
    ensure_domain_registration(domain)
    try:
        return MODEL_ZOO[domain]
    except KeyError as exc:
        available = ", ".join(MODEL_ZOO)
        raise KeyError(f"Unknown model domain '{domain}'. Available: {available}") from exc


def list_domains() -> tuple[str, ...]:
    """Return known model domains without importing their implementations."""

    return tuple(MODEL_ZOO)


def list_model_names(domain: str | None = None) -> tuple[str, ...]:
    """Return model names without importing built-in model implementations."""

    if domain is not None:
        _validate_identifier(domain, "domain")
        if domain not in MODEL_ZOO:
            available = ", ".join(MODEL_ZOO)
            raise KeyError(f"Unknown model domain '{domain}'. Available: {available}")
        builtins = [name for name, spec in BUILTIN_MODEL_INDEX.items() if spec[0] == domain]
        runtime = [name for name in MODEL_ZOO[domain] if name not in BUILTIN_MODEL_INDEX]
        return tuple((*builtins, *runtime))
    builtins = list(BUILTIN_MODEL_INDEX)
    runtime = [name for models in MODEL_ZOO.values() for name in models if name not in BUILTIN_MODEL_INDEX]
    return tuple((*builtins, *runtime))


def list_models(domain: str | None = None) -> ModelCatalog:
    """Return a defensive metadata snapshot for one domain or the full registry."""

    if domain is not None:
        return {name: _copy_metadata(metadata) for name, metadata in _require_domain(domain).items()}

    ensure_model_registrations()
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


def list_models_by_maturity(maturity: str, domain: str = "forecasting") -> ModelCatalog:
    """Return models in *domain* carrying one explicit maturity state."""

    _validate_identifier(maturity, "maturity")
    return {name: meta for name, meta in list_models(domain).items() if meta["maturity"] == maturity}


def list_models_by_fidelity(fidelity: str, domain: str = "forecasting") -> ModelCatalog:
    """Return models in *domain* carrying one implementation-fidelity state."""

    _validate_identifier(fidelity, "fidelity")
    return {name: meta for name, meta in list_models(domain).items() if meta["implementation_fidelity"] == fidelity}


def find_model_domain(name: str) -> str:
    """Return the unique domain for a built-in or runtime-registered model name."""

    _validate_identifier(name, "name")
    builtin = BUILTIN_MODEL_INDEX.get(name)
    if builtin is not None:
        return builtin[0]
    matches = [domain for domain, models in MODEL_ZOO.items() if name in models]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Unknown model '{name}'")
    raise RuntimeError(f"Model '{name}' is registered in multiple domains: {matches}")


def get_model_class(name: str, domain: str | None = None) -> type[Any]:
    """Resolve one model class while importing only its implementation.

    For built-ins this uses the static index and therefore avoids loading the
    full domain catalog. Runtime-registered models are resolved through the
    mutable registry.
    """

    _validate_identifier(name, "name")
    resolved_domain = domain or find_model_domain(name)
    _validate_identifier(resolved_domain, "domain")
    builtin = BUILTIN_MODEL_INDEX.get(name)
    if builtin is not None:
        builtin_domain, module_name, attr_name = builtin
        if builtin_domain != resolved_domain:
            raise KeyError(f"Model '{name}' belongs to domain '{builtin_domain}', not '{resolved_domain}'")
        value = getattr(import_module(module_name), attr_name)
        if not isinstance(value, type):
            raise TypeError(f"Indexed model '{name}' did not resolve to a class")
        return value
    return get_model_info(name, resolved_domain)["cls"]


def create_model(name: str, domain: str = "forecasting", **kwargs: Any) -> Any:
    """Instantiate a registered PyTorch-native model by name.

    Built-ins resolve directly from the static index, so construction imports
    only the requested implementation. Runtime-registered models continue to
    use the mutable registry.
    """

    return get_model_class(name, domain=domain)(**kwargs)
