"""Validated model-benchmark evidence shipped with the runtime catalog."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

MODEL_EVIDENCE_PATH = Path(__file__).with_name("model_evidence.json")
_REQUIRED_RECORD_FIELDS = {
    "domain",
    "dataset",
    "task",
    "protocol",
    "maturity",
    "benchmark_evidence",
    "evaluated_models",
    "config",
    "report",
    "seed",
    "dataset_fingerprint",
    "limitations",
}


def _non_empty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _validate_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != 1:
        raise ValueError("Model evidence registry must use schema_version 1")
    records = registry.get("records")
    mappings = registry.get("model_evidence")
    if not isinstance(records, dict) or not isinstance(mappings, dict):
        raise ValueError("Model evidence registry requires records and model_evidence mappings")

    for record_id, record in records.items():
        _non_empty_string(record_id, "record id")
        if not isinstance(record, dict) or set(record) != _REQUIRED_RECORD_FIELDS:
            raise ValueError(f"Evidence record {record_id!r} has invalid fields")
        for field in _REQUIRED_RECORD_FIELDS - {"evaluated_models", "seed", "dataset_fingerprint"}:
            _non_empty_string(record[field], f"{record_id}.{field}")
        models = record["evaluated_models"]
        if not isinstance(models, list) or not models or len(models) != len(set(models)):
            raise ValueError(f"{record_id}.evaluated_models must be a non-empty unique list")
        for model in models:
            _non_empty_string(model, f"{record_id}.evaluated_models")
        if not isinstance(record["seed"], int) or isinstance(record["seed"], bool):
            raise ValueError(f"{record_id}.seed must be an integer")
        fingerprint = record["dataset_fingerprint"]
        if fingerprint is not None:
            _non_empty_string(fingerprint, f"{record_id}.dataset_fingerprint")

    for model_key, record_id in mappings.items():
        _non_empty_string(model_key, "model evidence key")
        if model_key.count("/") != 1:
            raise ValueError(f"Invalid model evidence key: {model_key!r}")
        if record_id not in records:
            raise ValueError(f"Unknown evidence record {record_id!r} for {model_key!r}")
        domain, model = model_key.split("/", 1)
        record = records[record_id]
        if domain != record["domain"] or model not in record["evaluated_models"]:
            raise ValueError(f"Evidence mapping {model_key!r} conflicts with record {record_id!r}")


@lru_cache(maxsize=None)
def _load_registry(path: str) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_registry(registry)
    return registry


def load_model_evidence_registry(path: str | Path = MODEL_EVIDENCE_PATH) -> dict[str, Any]:
    """Return a defensive copy of the validated evidence registry."""

    return deepcopy(_load_registry(str(Path(path).resolve())))


def evidence_metadata_for_model(domain: str, name: str) -> dict[str, str]:
    """Return catalog promotion fields for one built-in model, if audited."""

    registry = _load_registry(str(MODEL_EVIDENCE_PATH.resolve()))
    record_id = registry["model_evidence"].get(f"{domain}/{name}")
    if record_id is None:
        return {}
    record = registry["records"][record_id]
    return {
        "maturity": record["maturity"],
        "benchmark_evidence": record["benchmark_evidence"],
        "evidence_record": record_id,
    }


def get_evidence_record(record_id: str) -> Mapping[str, Any]:
    """Return one defensive evidence-record snapshot."""

    _non_empty_string(record_id, "record_id")
    registry = _load_registry(str(MODEL_EVIDENCE_PATH.resolve()))
    try:
        return deepcopy(registry["records"][record_id])
    except KeyError as exc:
        raise KeyError(f"Unknown model evidence record: {record_id!r}") from exc


def list_evidence_records(domain: str | None = None) -> dict[str, Mapping[str, Any]]:
    """List defensive evidence-record snapshots, optionally filtered by domain."""

    registry = load_model_evidence_registry()
    if domain is None:
        return registry["records"]
    _non_empty_string(domain, "domain")
    return {key: value for key, value in registry["records"].items() if value["domain"] == domain}


__all__ = [
    "MODEL_EVIDENCE_PATH",
    "evidence_metadata_for_model",
    "get_evidence_record",
    "list_evidence_records",
    "load_model_evidence_registry",
]
