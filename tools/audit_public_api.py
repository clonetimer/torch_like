"""Generate and verify the protected public-API stability baseline."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_like  # noqa: E402

DEFAULT_POLICY_PATH = ROOT / "governance" / "api_stability_policy.json"
DEFAULT_DEPRECATIONS_PATH = ROOT / "governance" / "api_deprecations.json"
DEFAULT_BASELINE_PATH = ROOT / "governance" / "public_api_baseline.json"


def _version(value: str) -> tuple[int, int, int]:
    core = value.split("+", maxsplit=1)[0].split("-", maxsplit=1)[0]
    parts = core.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"Version must use MAJOR.MINOR.PATCH: {value!r}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "policy_id",
        "policy_version",
        "protected_namespaces",
        "minimum_minor_releases",
        "minimum_deprecation_days",
        "warning_category",
        "allowed_change_types",
        "allowed_statuses",
    }
    missing = required - policy.keys()
    if missing:
        raise ValueError(f"API stability policy is missing fields: {sorted(missing)}")
    if policy["schema_version"] != 1:
        raise ValueError(f"Unsupported API stability policy schema: {policy['schema_version']!r}")
    if not policy["protected_namespaces"] or len(set(policy["protected_namespaces"])) != len(
        policy["protected_namespaces"]
    ):
        raise ValueError("protected_namespaces must be non-empty and unique")
    if policy["minimum_minor_releases"] < 1 or policy["minimum_deprecation_days"] < 1:
        raise ValueError("Deprecation release and day minimums must be positive")
    return policy


def load_deprecations(
    policy: dict[str, Any] | None = None,
    path: Path = DEFAULT_DEPRECATIONS_PATH,
) -> list[dict[str, Any]]:
    policy = policy or load_policy()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("deprecations"), list):
        raise ValueError("API deprecations must use schema_version=1 and contain a deprecations list")
    required = {
        "symbol",
        "change_type",
        "deprecated_in",
        "remove_after",
        "announced_on",
        "remove_after_date",
        "replacement",
        "reason",
        "status",
    }
    seen: set[tuple[str, str]] = set()
    for record in document["deprecations"]:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"API deprecation record is missing fields: {sorted(missing)}")
        key = (record["symbol"], record["change_type"])
        if key in seen:
            raise ValueError(f"Duplicate API deprecation record: {key}")
        seen.add(key)
        if record["change_type"] not in policy["allowed_change_types"]:
            raise ValueError(f"Unsupported API deprecation change_type: {record['change_type']!r}")
        if record["status"] not in policy["allowed_statuses"]:
            raise ValueError(f"Unsupported API deprecation status: {record['status']!r}")
        deprecated = _version(record["deprecated_in"])
        remove_after = _version(record["remove_after"])
        if remove_after[0] == deprecated[0] and remove_after[1] - deprecated[1] < policy["minimum_minor_releases"]:
            raise ValueError(f"API deprecation {record['symbol']!r} does not span enough minor releases")
        if remove_after <= deprecated:
            raise ValueError(f"API deprecation {record['symbol']!r} must remove after its introduction")
        announced = date.fromisoformat(record["announced_on"])
        removal_date = date.fromisoformat(record["remove_after_date"])
        if (removal_date - announced).days < policy["minimum_deprecation_days"]:
            raise ValueError(f"API deprecation {record['symbol']!r} does not span enough calendar days")
        if not record["reason"]:
            raise ValueError(f"API deprecation {record['symbol']!r} requires a reason")
        if record["change_type"] == "rename" and not record["replacement"]:
            raise ValueError(f"API rename {record['symbol']!r} requires a replacement")
    return document["deprecations"]


def _default_value(value: Any) -> Any:
    if value is inspect.Parameter.empty:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)) and all(
        item is None or isinstance(item, (bool, int, float, str)) for item in value
    ):
        return list(value)
    return repr(value)


def _signature(value: Any) -> list[dict[str, Any]] | None:
    # ``inspect.signature(EnumSubclass)`` reflects EnumMeta internals and changed
    # across Python releases (for example 3.12 -> 3.13). Keep the historical
    # manifest shape deterministic so interpreter upgrades are not reported as
    # breaking torch_like API changes.
    if inspect.isclass(value) and issubclass(value, Enum):
        return [
            {"name": "value", "kind": "POSITIONAL_OR_KEYWORD", "required": True, "default": None},
            {"name": "names", "kind": "POSITIONAL_OR_KEYWORD", "required": False, "default": None},
            {"name": "module", "kind": "KEYWORD_ONLY", "required": False, "default": None},
            {"name": "qualname", "kind": "KEYWORD_ONLY", "required": False, "default": None},
            {"name": "type", "kind": "KEYWORD_ONLY", "required": False, "default": None},
            {"name": "start", "kind": "KEYWORD_ONLY", "required": False, "default": 1},
        ]
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return None
    return [
        {
            "name": parameter.name,
            "kind": parameter.kind.name,
            "required": parameter.default is inspect.Parameter.empty
            and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD},
            "default": _default_value(parameter.default),
        }
        for parameter in signature.parameters.values()
        if parameter.name not in {"self", "cls"}
    ]


def _symbol_record(value: Any) -> dict[str, Any]:
    if inspect.isclass(value):
        kind = "class"
    elif inspect.isfunction(value) or inspect.ismethod(value) or callable(value):
        kind = "callable"
    elif inspect.ismodule(value):
        kind = "module"
    else:
        kind = "value"
    value_type = value if inspect.isclass(value) else type(value)
    target = f"{getattr(value, '__module__', value_type.__module__)}.{getattr(value, '__qualname__', value_type.__qualname__)}"
    return {"kind": kind, "target": target, "signature": _signature(value) if callable(value) else None}


def build_manifest(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or load_policy()
    namespaces: dict[str, Any] = {}
    symbols: dict[str, Any] = {}
    for namespace in policy["protected_namespaces"]:
        module = importlib.import_module(namespace)
        exports = sorted(set(getattr(module, "__all__", ())))
        namespaces[namespace] = {"exports": exports}
        for name in exports:
            value = getattr(module, name)
            symbols[f"{namespace}.{name}"] = _symbol_record(value)
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "policy_version": policy["policy_version"],
        "package_version": torch_like.__version__,
        "namespaces": namespaces,
        "symbols": dict(sorted(symbols.items())),
    }


def render_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def _signature_is_breaking(previous: list[dict[str, Any]] | None, current: list[dict[str, Any]] | None) -> bool:
    if previous is None:
        return current is not None
    if current is None:
        return True
    current_by_name = {parameter["name"]: parameter for parameter in current}
    for parameter in previous:
        replacement = current_by_name.get(parameter["name"])
        if replacement is None:
            return True
        if any(replacement[field] != parameter[field] for field in ("kind", "required", "default")):
            return True
    previous_names = {parameter["name"] for parameter in previous}
    return any(parameter["required"] for parameter in current if parameter["name"] not in previous_names)


def _breaking_changes(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, str]]:
    changes = []
    for namespace, record in previous["namespaces"].items():
        current_exports = set(current["namespaces"].get(namespace, {}).get("exports", ()))
        for name in set(record["exports"]) - current_exports:
            changes.append({"symbol": f"{namespace}.{name}", "change_type": "remove"})
    for symbol, record in previous["symbols"].items():
        current_record = current["symbols"].get(symbol)
        if current_record is None:
            continue
        if current_record["kind"] != record["kind"] or current_record["target"] != record["target"]:
            changes.append({"symbol": symbol, "change_type": "replace"})
        elif _signature_is_breaking(record["signature"], current_record["signature"]):
            changes.append({"symbol": symbol, "change_type": "signature"})
    return changes


def validate_transition(
    previous: dict[str, Any],
    current: dict[str, Any],
    deprecations: list[dict[str, Any]],
    *,
    today: date | None = None,
) -> list[str]:
    today = today or date.today()
    version = _version(current["package_version"])
    records = {(record["symbol"], record["change_type"]): record for record in deprecations}
    errors = []
    for change in _breaking_changes(previous, current):
        key = (change["symbol"], change["change_type"])
        record = records.get(key)
        if record is None and change["change_type"] == "remove":
            record = records.get((change["symbol"], "rename"))
        if record is None:
            errors.append(f"Breaking API change lacks deprecation record: {change['change_type']} {change['symbol']}")
            continue
        if record["status"] == "cancelled":
            errors.append(f"Breaking API change uses a cancelled deprecation: {change['symbol']}")
        elif record["status"] != "removed":
            errors.append(f"Removed API must be marked removed in the deprecation ledger: {change['symbol']}")
        elif version < _version(record["remove_after"]):
            errors.append(f"API removal version has not been reached: {change['symbol']} -> {record['remove_after']}")
        elif today < date.fromisoformat(record["remove_after_date"]):
            errors.append(f"API removal date has not been reached: {change['symbol']} -> {record['remove_after_date']}")
    return errors


def validate_committed_baseline(
    manifest: dict[str, Any] | None = None,
    path: Path = DEFAULT_BASELINE_PATH,
) -> list[str]:
    manifest = manifest or build_manifest()
    if not path.is_file():
        return [f"Missing public API baseline: {path}"]
    return (
        [] if path.read_text(encoding="utf-8") == render_manifest(manifest) else [f"Stale public API baseline: {path}"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="validate transitions and write the current baseline")
    mode.add_argument("--check", action="store_true", help="fail when the committed baseline differs from public APIs")
    args = parser.parse_args()

    policy = load_policy()
    deprecations = load_deprecations(policy)
    manifest = build_manifest(policy)
    if args.write:
        if DEFAULT_BASELINE_PATH.is_file():
            previous = json.loads(DEFAULT_BASELINE_PATH.read_text(encoding="utf-8"))
            errors = validate_transition(previous, manifest, deprecations)
            if errors:
                for error in errors:
                    print(f"ERROR: {error}")
                return 1
        DEFAULT_BASELINE_PATH.write_text(render_manifest(manifest), encoding="utf-8")
        print(f"Wrote {DEFAULT_BASELINE_PATH.relative_to(ROOT)} with {len(manifest['symbols'])} protected symbols")
        return 0

    errors = validate_committed_baseline(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("Run: python tools/audit_public_api.py --write")
        return 1
    print(f"Public API baseline is current: {len(manifest['symbols'])} protected symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
