"""Behavioral checks for offline Workbuddy review integrity, not model quality."""

import copy
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "validate_workbuddy_triage.py"


def validator():
    # An absent validator must fail as a missing feature, not a collection error.
    assert importlib.util.find_spec("tools.validate_workbuddy_triage") is not None, "triage validator is missing"
    return importlib.import_module("tools.validate_workbuddy_triage")


def item(name="Candidate A", decision="hold"):
    return {
        "name": name,
        "report_date": "2026-07-02",
        "canonical_title": f"{name}: An Example Forecasting Architecture",
        "task_role": "supervised forecasting architecture",
        "decision": decision,
        "reason_category": "source_audit",
        "comparison_models": ["DeepAR", "PatchTST"],
        "incremental_value": "Conditional state-space inference differs from patch mixing and autoregressive output.",
        "rationale": "A pinned reference and independent evidence still need inspection.",
        "reopen_when": "Inspect a licensed source and a controlled independently run comparison.",
        "primary_url": "https://example.org/paper",
        "code_url": None,
        "source_revision": None,
        "code_license_status": "unknown",
        "code_license": "Unknown; no verified permission grant.",
        "license_evidence_url": None,
        "followup_priority": "medium",
        "notes": [],
    }


def document():
    return {
        "schema_version": 1,
        "reviewed_on": "2026-11-01",
        "baseline": {
            "report_count": 3,
            "unique_report_names": 5,
            "mapped_names": 3,
            "unmapped_names": 2,
            "catalog_total": 4,
            "forecasting_total": 2,
            "public_api_records": 20,
        },
        "scope": "Engineering triage only; no formal admission, implementation or benchmark claim.",
        "items": [item(), item("Candidate B", "out_of_backbone_scope")],
    }


def validate(data):
    return validator().validate_triage(data, {"Candidate A", "Candidate B"}, {"DeepAR", "PatchTST"})


def test_valid_review_reports_reviewed_and_unmapped_separately():
    module = validator()
    data = document()
    assert validate(data) == []
    assert module.summarize_triage(data) == {
        "reviewed_on": "2026-11-01",
        "reviewed_candidates": 2,
        "mapped_names": 3,
        "unmapped_names": 2,
        "unique_report_names": 5,
        "decision_counts": {"review_next": 0, "hold": 1, "skip_low_value": 0, "out_of_backbone_scope": 1},
    }
    assert data == document(), "validation and reporting must not mutate the snapshot"


def test_parser_reads_only_the_three_historical_remaining_rows():
    text = """# Snapshot
| Other table | 40 | Not a candidate |
| June / 六月 | 1 | Candidate A |
| July / 七月 | 2 | Candidate B, Candidate C |
| October / 十月 | 1 | Candidate D |
"""
    assert validator().parse_remaining_names(text) == {"Candidate A", "Candidate B", "Candidate C", "Candidate D"}


@pytest.mark.parametrize(
    "text, phrase",
    [
        ("# No table", "remaining"),
        ("| June / 六月 | 1 | Candidate A |", "period"),
        (
            "| June / 六月 | 2 | A |\n| July / 七月 | 1 | B |\n| October / 十月 | 1 | C |",
            "count",
        ),
        (
            "| June / 六月 | 1 | A |\n| July / 七月 | 1 | A |\n| October / 十月 | 1 | C |",
            "duplicate",
        ),
        (
            "| June / 六月 | 1 | A |\n| June / 六月 | 1 | D |\n| July / 七月 | 1 | B |\n| October / 十月 | 1 | C |",
            "period",
        ),
    ],
)
def test_invalid_inventory_fails_closed(text, phrase):
    with pytest.raises(ValueError, match=phrase):
        validator().parse_remaining_names(text)


@pytest.mark.parametrize("bad", [None, [], "review", 57, True])
def test_invalid_top_level_types_are_reported_without_tracebacks(bad):
    assert any("object" in error for error in validate(bad))


@pytest.mark.parametrize(
    "key, bad, phrase",
    [
        ("schema_version", True, "schema_version"),
        ("schema_version", 2, "schema_version"),
        ("reviewed_on", "2026-02-30", "reviewed_on"),
        ("reviewed_on", None, "reviewed_on"),
        ("scope", " ", "scope"),
        ("baseline", None, "baseline"),
        ("items", {}, "items"),
        ("items", [], "missing"),
        ("items", [None], "object"),
    ],
)
def test_invalid_top_level_values_are_rejected(key, bad, phrase):
    data = document()
    data[key] = bad
    assert any(phrase in error for error in validate(data))


@pytest.mark.parametrize("bad", [True, -1, 2.5, "2", None])
def test_baseline_counts_require_nonnegative_integers_not_booleans(bad):
    data = document()
    data["baseline"]["unmapped_names"] = bad
    assert any("unmapped_names" in error for error in validate(data))


def test_inconsistent_baseline_mapping_and_catalog_counts_are_rejected():
    data = document()
    data["baseline"].update(unique_report_names=6, unmapped_names=3, forecasting_total=5)
    errors = validate(data)
    assert any("inventory" in error for error in errors)
    assert any("catalog" in error for error in errors)
    data["baseline"]["unique_report_names"] = 9
    assert any("mapped" in error for error in validate(data))


def test_historical_snapshot_survives_later_catalog_additions():
    data = document()
    errors = validator().validate_triage(
        data,
        {"Candidate A", "Candidate B"},
        {"DeepAR", "PatchTST", "NewModelAddedAfterReview"},
    )
    assert errors == [], "adding a future model must not force a historical baseline rewrite"
    assert data["baseline"]["forecasting_total"] == 2


@pytest.mark.parametrize("level", ["top", "baseline", "item"])
def test_unknown_fields_cannot_silently_introduce_admission_claims(level):
    data = document()
    target = data if level == "top" else data["baseline"] if level == "baseline" else data["items"][0]
    target["formal_score"] = 99
    assert any("formal_score" in error for error in validate(data))


def test_duplicate_missing_and_extra_names_are_all_reported():
    data = document()
    data["items"] = [item(), item(), item("Unknown model")]
    errors = validate(data)
    assert any("duplicate" in error and "Candidate A" in error for error in errors)
    assert any("missing" in error and "Candidate B" in error for error in errors)
    assert any("unexpected" in error and "Unknown model" in error for error in errors)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("name", []),
        ("canonical_title", ""),
        ("task_role", None),
        ("decision", "core"),
        ("reason_category", " "),
        ("incremental_value", ""),
        ("rationale", None),
        ("reopen_when", ""),
        ("followup_priority", "ready"),
        ("code_license_status", "open-source"),
        ("code_license", ""),
        ("notes", "not an array"),
        ("notes", [None]),
        ("notes", [""]),
        ("report_date", "2026-13-01"),
        ("report_date", "2026-11-02"),
    ],
)
def test_invalid_record_fields_are_rejected(field, bad):
    data = document()
    data["items"][0][field] = bad
    assert any(field in error for error in validate(data))


@pytest.mark.parametrize("peers", [None, [], ["DeepAR"], ["DeepAR", "DeepAR"], ["DeepAR", "Missing"], [{}, "PatchTST"]])
def test_peers_must_be_two_distinct_current_catalog_models(peers):
    data = document()
    data["items"][0]["comparison_models"] = peers
    assert any("comparison_models" in error for error in validate(data))


@pytest.mark.parametrize("field", ["primary_url", "code_url", "license_evidence_url"])
@pytest.mark.parametrize(
    "url", ["not-a-url", "https:///no-host", "file:///tmp/model.py", "https://user:secret@example.org/"]
)
def test_source_urls_must_be_public_https_locations(field, url):
    data = document()
    data["items"][0][field] = url
    assert any(field in error for error in validate(data))


@pytest.mark.parametrize("revision", ["main", "abc123", "g" * 40, "1" * 39, 123])
def test_source_revision_is_an_immutable_hex_commit_not_a_branch(revision):
    data = document()
    data["items"][0].update(code_url="https://github.com/example/model", source_revision=revision)
    assert any("source_revision" in error for error in validate(data))


def test_source_revision_requires_a_source_location():
    data = document()
    data["items"][0]["source_revision"] = "a" * 40
    assert any("code_url" in error for error in validate(data))


@pytest.mark.parametrize("status", ["verified", "mixed"])
def test_verified_or_mixed_license_requires_actual_evidence_url(status):
    data = document()
    data["items"][0]["code_license_status"] = status
    assert any("license_evidence_url" in error for error in validate(data))


def test_review_priority_cannot_bypass_unknown_code_permission():
    data = document()
    data["items"][0]["decision"] = "review_next"
    assert any("review_next" in error for error in validate(data))
    data["items"][0].update(
        code_url="https://github.com/example/model",
        code_license_status="verified",
        code_license="MIT; exact reference license inspected, not a general asset clearance.",
        license_evidence_url="https://github.com/example/model/blob/" + "a" * 40 + "/LICENSE",
        source_revision="a" * 40,
    )
    assert validate(data) == []
    assert validator().summarize_triage(data)["mapped_names"] == 3


def test_missing_required_fields_fail_without_key_errors():
    data = document()
    del data["items"][0]["source_revision"]
    del data["baseline"]["mapped_names"]
    errors = validate(data)
    assert any("source_revision" in error for error in errors)
    assert any("mapped_names" in error for error in errors)


def test_repository_cli_validates_all57_and_preserves_mapping():
    assert SCRIPT.is_file(), "triage CLI is missing"
    result = subprocess.run([sys.executable, str(SCRIPT), "--json"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["reviewed_candidates"] == 57
    assert summary["unmapped_names"] == 57
    assert summary["mapped_names"] == 39
    assert summary["unique_report_names"] == 96
    assert sum(summary["decision_counts"].values()) == 57


def test_cli_reports_invalid_json_without_traceback(tmp_path):
    assert SCRIPT.is_file(), "triage CLI is missing"
    path = tmp_path / "invalid.json"
    path.write_text("not json", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--triage", str(path)], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "invalid" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_cli_rejects_semantically_broken_snapshot(tmp_path):
    assert SCRIPT.is_file(), "triage CLI is missing"
    data = json.loads((ROOT / "governance/workbuddy_forecasting_triage_r6.json").read_text(encoding="utf-8"))
    broken = copy.deepcopy(data)
    broken["items"].append(copy.deepcopy(broken["items"][0]))
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--triage", str(path)], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "duplicate" in result.stderr
    assert "Traceback" not in result.stderr
