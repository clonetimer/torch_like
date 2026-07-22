"""Runtime validation and compatibility derivation for model catalog schema v2."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, TypedDict

CATALOG_SCHEMA_PATH = Path(__file__).with_name("catalog_schema_v2.json")
DERIVED_METADATA_FIELDS = (
    "implementation_fidelity",
    "implementation_scope",
    "execution_profile",
    "maturity",
    "benchmark_evidence",
    "evidence_record",
)


class CatalogV2Metadata(TypedDict):
    implementation_fidelity: str
    implementation_scope: str
    execution_profile: str
    maturity: str
    benchmark_evidence: str
    evidence_record: str | None


def _validate_policy(policy: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "policy_id",
        "policy_version",
        "target_catalog_schema_version",
        "default_maturity",
        "default_benchmark_evidence",
        "allowed_implementation_fidelity",
        "allowed_implementation_scope",
        "allowed_execution_profile",
        "maturity_levels",
        "benchmark_evidence_levels",
        "legacy_integration_levels",
    }
    missing = required - policy.keys()
    if missing:
        raise ValueError(f"Catalog schema policy is missing fields: {sorted(missing)}")
    if policy["schema_version"] != 1 or policy["target_catalog_schema_version"] != 2:
        raise ValueError("Catalog schema policy must describe schema version 1 targeting catalog schema 2")
    if policy["default_maturity"] not in policy["maturity_levels"]:
        raise ValueError("default_maturity must be declared in maturity_levels")
    if policy["default_benchmark_evidence"] not in policy["benchmark_evidence_levels"]:
        raise ValueError("default_benchmark_evidence must be declared in benchmark_evidence_levels")

    allowed = {
        "implementation_fidelity": set(policy["allowed_implementation_fidelity"]),
        "implementation_scope": set(policy["allowed_implementation_scope"]),
        "execution_profile": set(policy["allowed_execution_profile"]),
    }
    for legacy, mapping in policy["legacy_integration_levels"].items():
        if not legacy or set(mapping) != set(allowed):
            raise ValueError(f"Invalid legacy integration mapping: {legacy!r}")
        for field, values in allowed.items():
            if mapping[field] not in values:
                raise ValueError(f"Unsupported {field} in legacy mapping {legacy!r}: {mapping[field]!r}")


@lru_cache(maxsize=None)
def _load_policy(path: str) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_policy(policy)
    return policy


def load_catalog_schema_policy(path: str | Path = CATALOG_SCHEMA_PATH) -> dict[str, Any]:
    """Return a defensive policy copy for runtime and governance tooling."""

    return deepcopy(_load_policy(str(Path(path).resolve())))


def derive_catalog_v2_metadata(
    integration_level: str,
    metadata: Mapping[str, Any],
    *,
    policy: dict[str, Any] | None = None,
) -> CatalogV2Metadata:
    """Derive additive v2 fields and reject ambiguous or unsupported promotion claims."""

    policy = policy or load_catalog_schema_policy()
    try:
        mapping = policy["legacy_integration_levels"][integration_level]
    except KeyError as exc:
        raise ValueError(f"Unmapped integration_level: {integration_level!r}") from exc

    resolved = dict(mapping)
    for field in ("implementation_fidelity", "implementation_scope", "execution_profile"):
        provided = metadata.get(field)
        if provided is not None and provided != resolved[field]:
            raise ValueError(
                f"{field}={provided!r} conflicts with integration_level={integration_level!r}; "
                f"expected {resolved[field]!r}"
            )

    maturity = metadata.get("maturity", policy["default_maturity"])
    if maturity not in policy["maturity_levels"]:
        raise ValueError(f"Unsupported maturity: {maturity!r}")
    benchmark = metadata.get("benchmark_evidence", policy["default_benchmark_evidence"])
    if benchmark not in policy["benchmark_evidence_levels"]:
        raise ValueError(f"Unsupported benchmark_evidence: {benchmark!r}")
    evidence_record = metadata.get("evidence_record")
    if evidence_record is not None and (not isinstance(evidence_record, str) or not evidence_record.strip()):
        raise ValueError("evidence_record must be a non-empty string when provided")
    if (
        maturity != policy["default_maturity"] or benchmark != policy["default_benchmark_evidence"]
    ) and evidence_record is None:
        raise ValueError("Promoted maturity or benchmark_evidence requires evidence_record")

    return CatalogV2Metadata(
        implementation_fidelity=resolved["implementation_fidelity"],
        implementation_scope=resolved["implementation_scope"],
        execution_profile=resolved["execution_profile"],
        maturity=maturity,
        benchmark_evidence=benchmark,
        evidence_record=evidence_record,
    )


__all__ = [
    "CATALOG_SCHEMA_PATH",
    "DERIVED_METADATA_FIELDS",
    "CatalogV2Metadata",
    "derive_catalog_v2_metadata",
    "load_catalog_schema_policy",
]
