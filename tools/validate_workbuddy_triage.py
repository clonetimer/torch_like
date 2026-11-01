"""Validate an offline Workbuddy triage snapshot, never scientific admission."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ("review_next", "hold", "skip_low_value", "out_of_backbone_scope")
LICENSE_STATUSES = ("verified", "declared", "unknown", "mixed")
TOP_FIELDS = {"schema_version", "reviewed_on", "baseline", "scope", "items"}
BASELINE_FIELDS = {
    "report_count",
    "unique_report_names",
    "mapped_names",
    "unmapped_names",
    "catalog_total",
    "forecasting_total",
    "public_api_records",
}
TEXT_FIELDS = {
    "name",
    "canonical_title",
    "task_role",
    "reason_category",
    "incremental_value",
    "rationale",
    "reopen_when",
    "code_license",
}
ITEM_FIELDS = TEXT_FIELDS | {
    "report_date",
    "decision",
    "comparison_models",
    "primary_url",
    "code_url",
    "source_revision",
    "code_license_status",
    "license_evidence_url",
    "followup_priority",
    "notes",
}


def parse_remaining_names(text: str) -> set[str]:
    """Read the three R5 period rows and fail closed on incomplete inventory."""
    periods = {"June", "July", "October"}
    seen_periods: set[str] = set()
    names: set[str] = set()
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().split("|")[1:-1]]
        if not cells:
            continue
        period = cells[0].split("/")[0].strip()
        if period not in periods:
            continue
        if period in seen_periods:
            raise ValueError(f"duplicate remaining inventory period: {period}")
        seen_periods.add(period)
        if len(cells) != 3 or not cells[1].isdigit():
            raise ValueError(f"invalid remaining inventory count/columns for {period}")
        row_names = [name.strip() for name in cells[2].split(",")]
        if any(not name for name in row_names) or len(row_names) != int(cells[1]):
            raise ValueError(f"remaining inventory count mismatch for {period}")
        for name in row_names:
            if name in names:
                raise ValueError(f"duplicate remaining inventory name: {name}")
            names.add(name)
    if seen_periods != periods:
        raise ValueError("remaining inventory must include each June/July/October period exactly once")
    return names


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fields(record: dict, expected: set[str], label: str, errors: list[str]) -> None:
    missing = expected - record.keys()
    extra = record.keys() - expected
    if missing:
        errors.append(f"{label}: missing fields {', '.join(sorted(missing))}")
    if extra:
        errors.append(f"{label}: unexpected fields {', '.join(sorted(map(str, extra)))}")


def _date(value: object, label: str, errors: list[str]) -> date | None:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    errors.append(f"{label}: expected a real ISO date YYYY-MM-DD")
    return None


def _https_url(value: object) -> bool:
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return False
    try:
        parts = urlsplit(value)
        return parts.scheme == "https" and bool(parts.hostname) and parts.username is None and parts.password is None
    except ValueError:
        return False


def validate_triage(data: object, expected_names: set[str], catalog_names: set[str]) -> list[str]:
    """Return structural/provenance errors; do not award licenses or scores."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["triage: expected a JSON object"]
    _fields(data, TOP_FIELDS, "triage", errors)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        errors.append("schema_version: expected integer 1")
    reviewed_on = _date(data.get("reviewed_on"), "reviewed_on", errors)
    if not _nonempty_text(data.get("scope")):
        errors.append("scope: expected a nonempty scope/claim boundary")

    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        errors.append("baseline: expected a JSON object")
    else:
        _fields(baseline, BASELINE_FIELDS, "baseline", errors)
        valid_counts = True
        for field in sorted(BASELINE_FIELDS):
            value = baseline.get(field)
            if type(value) is not int or value < 0:
                errors.append(f"baseline.{field}: expected a nonnegative integer, not a boolean")
                valid_counts = False
        if valid_counts:
            if baseline["mapped_names"] + baseline["unmapped_names"] != baseline["unique_report_names"]:
                errors.append("baseline: mapped_names + unmapped_names must equal unique_report_names")
            if baseline["unmapped_names"] != len(expected_names):
                errors.append("baseline.unmapped_names: does not match the historical inventory")
            if baseline["forecasting_total"] > len(catalog_names):
                errors.append("baseline.forecasting_total: exceeds the current forecasting catalog")
            if baseline["forecasting_total"] > baseline["catalog_total"]:
                errors.append("baseline.catalog_total: cannot be smaller than forecasting_total")

    items = data.get("items")
    if not isinstance(items, list):
        errors.append("items: expected an array")
        return errors
    seen: set[str] = set()
    for index, record in enumerate(items):
        label = f"items[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{label}: expected a JSON object")
            continue
        _fields(record, ITEM_FIELDS, label, errors)
        for field in sorted(TEXT_FIELDS):
            if not _nonempty_text(record.get(field)):
                errors.append(f"{label}.{field}: expected nonempty text")
        name = record.get("name")
        if _nonempty_text(name):
            if name != name.strip():
                errors.append(f"{label}.name: surrounding whitespace is not part of a report identity")
            if name in seen:
                errors.append(f"{label}: duplicate candidate {name}")
            seen.add(name)
        report_date = _date(record.get("report_date"), f"{label}.report_date", errors)
        if report_date and reviewed_on and report_date > reviewed_on:
            errors.append(f"{label}.report_date: later than reviewed_on")
        for field, allowed in (
            ("decision", DECISIONS),
            ("code_license_status", LICENSE_STATUSES),
            ("followup_priority", ("high", "medium", "low")),
        ):
            if record.get(field) not in allowed:
                errors.append(f"{label}.{field}: expected one of {', '.join(allowed)}")
        peers = record.get("comparison_models")
        if (
            not isinstance(peers, list)
            or len(peers) != 2
            or not all(isinstance(peer, str) for peer in peers)
            or len(set(peers)) != 2
            or not set(peers) <= catalog_names
        ):
            errors.append(f"{label}.comparison_models: require two distinct existing forecasting catalog names")
        notes = record.get("notes")
        if not isinstance(notes, list) or not all(_nonempty_text(note) for note in notes):
            errors.append(f"{label}.notes: expected an array of nonempty strings")
        for field in ("primary_url", "code_url", "license_evidence_url"):
            value = record.get(field)
            if value is None and field != "primary_url":
                continue
            if not _https_url(value):
                errors.append(f"{label}.{field}: expected a credential-free HTTPS URL")
        revision = record.get("source_revision")
        if revision is not None:
            if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                errors.append(f"{label}.source_revision: expected a 40-hex immutable commit or null")
            if record.get("code_url") is None:
                errors.append(f"{label}.code_url: required when source_revision is supplied")
        status = record.get("code_license_status")
        if status in ("verified", "mixed") and record.get("license_evidence_url") is None:
            errors.append(f"{label}.license_evidence_url: required for verified/mixed code permission evidence")
        if record.get("decision") == "review_next" and (
            status not in ("verified", "mixed") or record.get("code_url") is None
        ):
            errors.append(f"{label}.review_next: source and inspected code permission required; otherwise use hold")

    missing = expected_names - seen
    extra = seen - expected_names
    if missing:
        errors.append(f"inventory: missing candidates {', '.join(sorted(missing))}")
    if extra:
        errors.append(f"inventory: unexpected candidates {', '.join(sorted(extra))}")
    return errors


def summarize_triage(data: dict) -> dict:
    """Summarize already validated data without converting review into mapping."""
    counts = Counter(record["decision"] for record in data["items"])
    baseline = data["baseline"]
    return {
        "reviewed_on": data["reviewed_on"],
        "reviewed_candidates": len(data["items"]),
        "mapped_names": baseline["mapped_names"],
        "unmapped_names": baseline["unmapped_names"],
        "unique_report_names": baseline["unique_report_names"],
        "decision_counts": {decision: counts[decision] for decision in DECISIONS},
    }


def _catalog_names() -> set[str]:
    """Read the generated literal index without importing torch or any model."""
    path = ROOT / "torch_like/models/_builtin_index.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BUILTIN_MODEL_INDEX" for target in node.targets
        ):
            index = ast.literal_eval(node.value)
            return {name for name, (domain, _, _) in index.items() if domain == "forecasting"}
    raise ValueError("static forecasting catalog index not found")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", type=Path, default=ROOT / "governance/workbuddy_forecasting_triage_r6.json")
    parser.add_argument("--inventory", type=Path, default=ROOT / "docs/WORKBUDDY_FORECASTING_R5.md")
    parser.add_argument("--json", action="store_true", help="print machine-readable derived review/mapping counts")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.triage.read_text(encoding="utf-8"))
        names = parse_remaining_names(args.inventory.read_text(encoding="utf-8"))
        errors = validate_triage(data, names, _catalog_names())
    except (OSError, ValueError) as exc:
        print(f"Invalid triage input: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("Triage validation failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    summary = summarize_triage(data)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Validated {summary['reviewed_candidates']} triaged candidates; "
            f"mapping remains {summary['mapped_names']}/{summary['unique_report_names']}, "
            f"{summary['unmapped_names']} unmapped. Triage is not admission or implementation."
        )
        for decision, count in summary["decision_counts"].items():
            print(f"{decision}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
