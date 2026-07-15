"""Validate candidate-pool scoring, evidence grades, and admission decisions."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "governance" / "model_admission_policy.json"
DEFAULT_BASELINE_PATH = ROOT / "governance" / "model_registry_baseline.txt"


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load and structurally validate the versioned admission policy."""

    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "policy_id",
        "policy_version",
        "effective_date",
        "dimensions",
        "evidence_coefficients",
        "evidence_requirements",
        "thresholds",
        "allowed_gate_results",
        "allowed_decisions",
        "allowed_decision_qualifiers",
        "review",
    }
    missing = required - policy.keys()
    if missing:
        raise ValueError(f"Admission policy is missing fields: {sorted(missing)}")
    if policy["schema_version"] != 1:
        raise ValueError(f"Unsupported admission policy schema: {policy['schema_version']!r}")
    if tuple(policy["dimensions"]) != tuple("RPUEIAX"):
        raise ValueError("Admission policy dimensions must be ordered R/P/U/E/I/A/X")
    if set(policy["evidence_coefficients"]) != set(policy["evidence_requirements"]):
        raise ValueError("Evidence coefficients and requirements must define the same grades")
    if policy["thresholds"]["core"] <= policy["thresholds"]["observation"]:
        raise ValueError("Core threshold must be greater than observation threshold")
    date.fromisoformat(policy["effective_date"])
    return policy


POLICY = load_policy()
EVIDENCE_COEFFICIENTS = POLICY["evidence_coefficients"]
DIMENSION_LIMITS = tuple(POLICY["dimensions"].values())
SCORECARD_HEADER = (
    "Model",
    "R/25",
    "P/20",
    "U/15",
    "E/10",
    "I/10",
    "A/10",
    "X/10",
    "Value",
    "Evidence",
    "Risk",
    "Final",
    "Decision",
)
HARD_GATE_HEADER = (
    "Model/version",
    "Identity",
    "Primary source",
    "Relevant",
    "Verifiable",
    "Non-duplicate",
    "License/availability",
    "Gate",
)
REVIEW_HEADER = ("Model/version", "Last reviewed", "Review by", "Owner", "Early-review trigger")
REQUIRED_RECORD_FIELDS = (
    "Category",
    "Core value",
    "Evidence",
    "Peer comparison",
    "Decision",
    "Conclusion reason",
    "Missing information",
)


def load_registry_baseline(path: Path = DEFAULT_BASELINE_PATH) -> set[tuple[str, str]]:
    """Load the registry snapshot grandfathered when governance became mandatory."""

    entries: set[tuple[str, str]] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count(":") != 1:
            raise ValueError(f"Invalid registry baseline entry at line {line_number}: {line!r}")
        domain, name = line.split(":", maxsplit=1)
        if not domain or not name:
            raise ValueError(f"Empty registry baseline field at line {line_number}")
        entry = (domain, name)
        if entry in entries:
            raise ValueError(f"Duplicate registry baseline entry: {domain}:{name}")
        entries.add(entry)
    if not entries:
        raise ValueError("Registry baseline contains no models")
    return entries


@dataclass(frozen=True)
class CandidateScore:
    model: str
    dimensions: tuple[int, ...]
    value: int
    evidence: str
    risk: float
    final: float
    decision: str


@dataclass(frozen=True)
class HardGate:
    model: str
    checks: tuple[str, ...]
    license_availability: str
    gate: str


@dataclass(frozen=True)
class ReviewRecord:
    model: str
    last_reviewed: date
    review_by: date
    owner: str
    trigger: str


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_table(markdown: str, header: tuple[str, ...]) -> list[list[str]]:
    lines = markdown.splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if tuple(_cells(line)) == header)
    except StopIteration as exc:
        raise ValueError(f"Candidate pool is missing table header: {' | '.join(header)}") from exc
    rows: list[list[str]] = []
    for line in lines[header_index + 2 :]:
        if not line.lstrip().startswith("|"):
            break
        cells = _cells(line)
        if len(cells) != len(header):
            raise ValueError(f"Table row has {len(cells)} columns instead of {len(header)}: {line}")
        rows.append(cells)
    return rows


def _model_key(name: str) -> str:
    base = re.split(r"\s*(?:/|\()", name, maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]", "", base.lower())


def _same_model(first: str, second: str) -> bool:
    first_key, second_key = _model_key(first), _model_key(second)
    return first_key.startswith(second_key) or second_key.startswith(first_key)


def parse_hard_gates(markdown: str) -> list[HardGate]:
    rows = _parse_table(markdown, HARD_GATE_HEADER)
    if not rows:
        raise ValueError("Hard-gate table contains no candidates")
    return [HardGate(row[0], tuple(row[1:6]), row[6], row[7].replace("**", "").lower()) for row in rows]


def parse_scorecard(markdown: str) -> list[CandidateScore]:
    """Parse the canonical scorecard table from candidate-pool Markdown."""
    rows: list[CandidateScore] = []
    try:
        table_rows = _parse_table(markdown, SCORECARD_HEADER)
    except ValueError as exc:
        raise ValueError("Candidate pool is missing the canonical scorecard header") from exc
    for cells in table_rows:
        try:
            dimensions = tuple(int(value) for value in cells[1:8])
            rows.append(
                CandidateScore(
                    model=cells[0],
                    dimensions=dimensions,
                    value=int(cells[8]),
                    evidence=cells[9],
                    risk=float(cells[10]),
                    final=float(cells[11]),
                    decision=cells[12].lower(),
                )
            )
        except ValueError as exc:
            raise ValueError(f"Invalid numeric scorecard row for {cells[0]!r}") from exc
    if not rows:
        raise ValueError("Candidate scorecard contains no rows")
    return rows


def parse_review_register(markdown: str) -> list[ReviewRecord]:
    """Parse candidate ownership and freshness deadlines."""

    records: list[ReviewRecord] = []
    for row in _parse_table(markdown, REVIEW_HEADER):
        try:
            records.append(ReviewRecord(row[0], date.fromisoformat(row[1]), date.fromisoformat(row[2]), row[3], row[4]))
        except ValueError as exc:
            raise ValueError(f"Invalid review date for {row[0]!r}") from exc
    if not records:
        raise ValueError("Review register contains no candidates")
    return records


def expected_decision(final: float, evidence: str, policy: dict[str, Any] | None = None) -> str:
    active_policy = POLICY if policy is None else policy
    if final >= active_policy["thresholds"]["core"] and evidence in {"A", "B"}:
        return "core"
    if final >= active_policy["thresholds"]["observation"]:
        return "observation"
    return "rejected"


def validate_scores(rows: list[CandidateScore], policy: dict[str, Any] | None = None) -> list[str]:
    """Return all scorecard violations instead of failing on the first row."""
    active_policy = POLICY if policy is None else policy
    coefficients = active_policy["evidence_coefficients"]
    dimension_limits = tuple(active_policy["dimensions"].values())
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.model:
            errors.append("Scorecard contains an empty model name")
        elif row.model in seen:
            errors.append(f"{row.model}: duplicate scorecard row")
        seen.add(row.model)

        if len(row.dimensions) != len(dimension_limits):
            errors.append(f"{row.model}: expected seven dimensions")
            continue
        for label, score, limit in zip("RPUEIAX", row.dimensions, dimension_limits, strict=True):
            if not 0 <= score <= limit:
                errors.append(f"{row.model}: {label}={score} is outside 0..{limit}")
        calculated_value = sum(row.dimensions)
        if row.value != calculated_value:
            errors.append(f"{row.model}: value {row.value} != dimension sum {calculated_value}")
        if row.evidence not in coefficients:
            errors.append(f"{row.model}: unknown evidence grade {row.evidence!r}")
            continue
        if row.risk < 0:
            errors.append(f"{row.model}: risk deduction must be non-negative")
        calculated_final = row.value * coefficients[row.evidence] - row.risk
        if abs(row.final - calculated_final) > 0.051:
            errors.append(f"{row.model}: final {row.final:g} != calculated {calculated_final:g}")
        actual_decision = row.decision.split(",", maxsplit=1)[0].strip()
        decision_parts = [part.strip() for part in row.decision.split(",")]
        qualifiers = decision_parts[1:]
        unknown_qualifiers = set(qualifiers) - set(active_policy["allowed_decision_qualifiers"])
        if actual_decision not in active_policy["allowed_decisions"]:
            errors.append(f"{row.model}: unsupported decision {actual_decision!r}")
        if unknown_qualifiers:
            errors.append(f"{row.model}: unsupported decision qualifier(s) {sorted(unknown_qualifiers)}")
        if len(qualifiers) != len(set(qualifiers)):
            errors.append(f"{row.model}: duplicate decision qualifier")
        if "implemented" in qualifiers and "no code" in qualifiers:
            errors.append(f"{row.model}: decision cannot be both implemented and no code")
        if actual_decision == "rejected" and "implemented" in qualifiers:
            errors.append(f"{row.model}: rejected candidate cannot authorize implementation")
        required_decision = expected_decision(calculated_final, row.evidence, active_policy)
        if actual_decision != required_decision:
            errors.append(
                f"{row.model}: decision {actual_decision!r} must be {required_decision!r} "
                f"for final={calculated_final:g}, evidence={row.evidence}"
            )
    return errors


def validate_document(
    path: Path,
    *,
    policy: dict[str, Any] | None = None,
    registry_baseline: set[tuple[str, str]] | None = None,
    as_of: date | None = None,
) -> list[str]:
    markdown = path.read_text(encoding="utf-8")
    active_policy = POLICY if policy is None else policy
    active_baseline = load_registry_baseline() if registry_baseline is None else registry_baseline
    rows = parse_scorecard(markdown)
    gates = parse_hard_gates(markdown)
    reviews = parse_review_register(markdown)
    errors = validate_scores(rows, active_policy)
    errors.extend(_validate_scope(markdown, gates, active_policy))
    errors.extend(_validate_hard_gates(gates, rows, active_policy))
    errors.extend(_validate_reviews(reviews, gates, active_policy, as_of or date.today()))
    rationale_models = _parse_rationale_models(markdown)
    for row in rows:
        if row.model not in rationale_models:
            errors.append(f"{row.model}: missing Dimension Rationale row")
    errors.extend(_validate_detailed_records(markdown, rows, gates, active_policy))
    errors.extend(_validate_registry_admission(markdown, rows, active_baseline))
    decisions = {row.decision.split(",", maxsplit=1)[0].strip() for row in rows}
    missing_decisions = {"core", "observation", "rejected"} - decisions
    if missing_decisions:
        errors.append(
            f"Scorecard must contain core, observation, and rejected results; missing {sorted(missing_decisions)}"
        )
    return errors


def _validate_registry_admission(
    markdown: str,
    scores: list[CandidateScore],
    baseline: set[tuple[str, str]],
) -> list[str]:
    """Reject registry additions that bypassed the evidence-governance workflow."""

    from torch_like.models.registry import list_domains, list_models

    actual = {(domain, name) for domain in list_domains() for name in list_models(domain)}
    errors = [
        f"Registry baseline entry disappeared without governance update: {domain}:{name}"
        for domain, name in sorted(baseline - actual)
    ]

    implemented_registry_names: set[str] = set()
    sections = list(re.finditer(r"^### (.+)$", markdown, flags=re.MULTILINE))
    for score in scores:
        if "implemented" not in score.decision:
            continue
        matching_sections = [match for match in sections if _same_model(match.group(1), score.model)]
        if len(matching_sections) != 1:
            continue
        match = matching_sections[0]
        next_positions = [item.start() for item in sections if item.start() > match.start()]
        body = markdown[match.end() : min(next_positions, default=len(markdown))]
        registry_match = re.search(r"^- Registry: `([^`]+)`;", body, flags=re.MULTILINE)
        if registry_match:
            implemented_registry_names.add(registry_match.group(1))

    for domain, name in sorted(actual - baseline):
        if name not in implemented_registry_names:
            errors.append(
                f"Ungoverned registry addition {domain}:{name}; add and approve a candidate-pool record first"
            )
    return errors


def _validate_scope(markdown: str, gates: list[HardGate], policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    target_match = re.search(r"^- Target domain: (.+)$", markdown, flags=re.MULTILINE)
    minimum_match = re.search(r"^- Minimum candidate count: (\d+)\.$", markdown, flags=re.MULTILINE)
    date_match = re.search(r"^- Evaluation date: (\d{4}-\d{2}-\d{2})\.$", markdown, flags=re.MULTILINE)
    policy_match = re.search(r"^- Governance policy: `([^`]+)` version ([\w.-]+)\.$", markdown, flags=re.MULTILINE)
    if not target_match or not target_match.group(1).strip():
        errors.append("Scope is missing a non-empty Target domain")
    if not minimum_match:
        errors.append("Scope is missing Minimum candidate count")
    else:
        minimum = int(minimum_match.group(1))
        if minimum <= 0:
            errors.append("Minimum candidate count must be positive")
        elif len(gates) < minimum:
            errors.append(f"Hard-gate pool has {len(gates)} candidates, below declared minimum {minimum}")
    if not date_match:
        errors.append("Scope is missing Evaluation date")
    else:
        try:
            date.fromisoformat(date_match.group(1))
        except ValueError:
            errors.append(f"Evaluation date is invalid: {date_match.group(1)!r}")
    if not policy_match:
        errors.append("Scope is missing Governance policy declaration")
    else:
        declared_path, declared_version = policy_match.groups()
        if declared_path != "governance/model_admission_policy.json":
            errors.append(f"Governance policy path is not canonical: {declared_path!r}")
        if declared_version != policy["policy_version"]:
            errors.append(f"Governance policy version {declared_version!r} does not match {policy['policy_version']!r}")
    return errors


def _validate_hard_gates(gates: list[HardGate], scores: list[CandidateScore], policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for gate in gates:
        key = _model_key(gate.model)
        if key in seen:
            errors.append(f"{gate.model}: duplicate hard-gate row")
        seen.add(key)
        passed = gate.gate.startswith("pass")
        if gate.gate not in set(policy["allowed_gate_results"]):
            errors.append(f"{gate.model}: unsupported gate result {gate.gate!r}")
        if passed and any(
            "unknown" in value.lower() or not value for value in (*gate.checks, gate.license_availability)
        ):
            errors.append(f"{gate.model}: a passing hard gate contains unknown or empty evidence")
        matches = [score for score in scores if _same_model(gate.model, score.model)]
        if passed and len(matches) != 1:
            errors.append(f"{gate.model}: passing hard gate must have exactly one formal score, found {len(matches)}")
        if not passed and matches:
            errors.append(f"{gate.model}: failed hard gate must not receive a formal score")
    for score in scores:
        matches = [gate for gate in gates if gate.gate.startswith("pass") and _same_model(gate.model, score.model)]
        if len(matches) != 1:
            errors.append(
                f"{score.model}: formal score must map to exactly one passing hard gate, found {len(matches)}"
            )
    return errors


def _validate_reviews(
    reviews: list[ReviewRecord],
    gates: list[HardGate],
    policy: dict[str, Any],
    as_of: date,
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    maximum_horizon = policy["review"]["maximum_horizon_days"]
    expected_owner = policy["review"]["owner"]
    for review in reviews:
        key = _model_key(review.model)
        if key in seen:
            errors.append(f"{review.model}: duplicate review-register row")
        seen.add(key)
        matches = [gate for gate in gates if _same_model(review.model, gate.model)]
        if len(matches) != 1:
            errors.append(f"{review.model}: review row must map to exactly one hard gate, found {len(matches)}")
        if review.last_reviewed > as_of:
            errors.append(f"{review.model}: last-reviewed date {review.last_reviewed} is in the future")
        if review.review_by < review.last_reviewed:
            errors.append(f"{review.model}: review deadline precedes its last review")
        if (review.review_by - review.last_reviewed).days > maximum_horizon:
            errors.append(f"{review.model}: review horizon exceeds {maximum_horizon} days")
        if policy["review"]["expired_is_error"] and review.review_by < as_of:
            errors.append(f"{review.model}: review expired on {review.review_by}")
        if review.owner != expected_owner:
            errors.append(f"{review.model}: owner must be {expected_owner!r}")
        if not review.trigger or review.trigger.lower() == "unknown":
            errors.append(f"{review.model}: early-review trigger is missing")
    for gate in gates:
        matches = [review for review in reviews if _same_model(gate.model, review.model)]
        if len(matches) != 1:
            errors.append(f"{gate.model}: hard gate must have exactly one review row, found {len(matches)}")
    return errors


def _validate_detailed_records(
    markdown: str,
    scores: list[CandidateScore],
    gates: list[HardGate],
    policy: dict[str, Any],
) -> list[str]:
    sections: list[tuple[str, str, str]] = []
    matches = list(re.finditer(r"^### (.+)$", markdown, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        parent_matches = list(re.finditer(r"^## (.+)$", markdown[: match.start()], flags=re.MULTILINE))
        parent = parent_matches[-1].group(1).strip() if parent_matches else ""
        sections.append((match.group(1).strip(), markdown[match.end() : end], parent))

    errors: list[str] = []
    registry_catalog = None
    for gate in gates:
        matching = [(heading, body) for heading, body, _ in sections if _same_model(heading, gate.model)]
        if len(matching) != 1:
            errors.append(f"{gate.model}: expected exactly one detailed record, found {len(matching)}")
            continue
        _, body = matching[0]
        for field in REQUIRED_RECORD_FIELDS:
            if not re.search(rf"^- {re.escape(field)}:", body, flags=re.MULTILINE):
                errors.append(f"{gate.model}: detailed record is missing '{field}'")
        evidence_match = re.search(r"^- Evidence:(.+)$", body, flags=re.MULTILINE)
        if evidence_match and evidence_match.group(1).count("](") < 2:
            errors.append(f"{gate.model}: Evidence must link at least two sources")
        peer_match = re.search(r"^- Peer comparison:(.+)$", body, flags=re.MULTILINE)
        if peer_match and len(re.findall(r"`[^`]+`", peer_match.group(1))) < 2:
            errors.append(f"{gate.model}: Peer comparison must name at least two peer models")

    for score in scores:
        matching = [(heading, body, parent) for heading, body, parent in sections if _same_model(heading, score.model)]
        if len(matching) != 1:
            errors.append(f"{score.model}: expected exactly one detailed record, found {len(matching)}")
            continue
        _, body, parent = matching[0]
        evidence_match = re.search(r"^- Evidence:(.+)$", body, flags=re.MULTILINE)
        if evidence_match:
            evidence = evidence_match.group(1).lower()
            primary_count = evidence.count("[primary:")
            official_count = evidence.count("[official:")
            independent_count = evidence.count("[independent:")
            requirements = policy["evidence_requirements"][score.evidence]
            if primary_count < requirements["primary"]:
                errors.append(
                    f"{score.model}: evidence grade {score.evidence} requires "
                    f"{requirements['primary']} primary source(s), found {primary_count}"
                )
            if official_count < requirements["official"]:
                errors.append(
                    f"{score.model}: evidence grade {score.evidence} requires "
                    f"{requirements['official']} official source(s), found {official_count}"
                )
            required_independent = requirements["independent"]
            if independent_count < required_independent:
                errors.append(
                    f"{score.model}: evidence grade {score.evidence} requires {required_independent} "
                    f"independent source(s), found {independent_count}"
                )
        score_match = re.search(
            r"^- Score: value (\d+); evidence ([A-D]); risk ([\d.]+); final ([\d.]+)\.$",
            body,
            flags=re.MULTILINE,
        )
        if not score_match:
            errors.append(f"{score.model}: detailed record has no canonical Score line")
        else:
            detailed_value = int(score_match.group(1))
            detailed_evidence = score_match.group(2)
            detailed_risk = float(score_match.group(3))
            detailed_final = float(score_match.group(4))
            if (
                detailed_value != score.value
                or detailed_evidence != score.evidence
                or detailed_risk != score.risk
                or abs(detailed_final - score.final) > 0.051
            ):
                errors.append(f"{score.model}: detailed Score does not match scorecard")
        expected_parent = {
            "core": "Core Inclusion /",
            "observation": "Observation /",
            "rejected": "Rejected /",
        }[score.decision.split(",", maxsplit=1)[0].strip()]
        if not parent.startswith(expected_parent):
            errors.append(f"{score.model}: decision does not match detailed section {parent!r}")
        if "implemented" in score.decision:
            registry_match = re.search(
                r"^- Registry: `([^`]+)`; integration level `([^`]+)`\.$",
                body,
                flags=re.MULTILINE,
            )
            if not registry_match:
                errors.append(f"{score.model}: implemented candidate is missing canonical Registry claim")
            else:
                if registry_catalog is None:
                    from torch_like.models.registry import list_models

                    registry_catalog = list_models()
                registry_name, claimed_level = registry_match.groups()
                metadata = registry_catalog.get(registry_name)
                if metadata is None:
                    errors.append(f"{score.model}: Registry model {registry_name!r} does not exist")
                elif metadata["integration_level"] != claimed_level:
                    errors.append(
                        f"{score.model}: claimed integration level {claimed_level!r} does not match "
                        f"registry {metadata['integration_level']!r}"
                    )
    return errors


def _parse_rationale_models(markdown: str) -> set[str]:
    marker = "## Dimension Rationale /"
    try:
        section = markdown.split(marker, maxsplit=1)[1]
    except IndexError as exc:
        raise ValueError("Candidate pool is missing Dimension Rationale section") from exc
    section = section.split("\n## ", maxsplit=1)[0]
    models: set[str] = set()
    for line in section.splitlines():
        cells = _cells(line) if line.lstrip().startswith("|") else []
        if len(cells) == 8 and cells[0] not in {"Model", "---"}:
            if any(not reason or reason == "unknown" for reason in cells[1:]):
                raise ValueError(f"{cells[0]} has an empty or unknown dimension rationale")
            models.add(cells[0])
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("docs/MODEL_CANDIDATE_POOL.md"),
        help="candidate-pool Markdown path",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="versioned model-admission policy path",
    )
    parser.add_argument(
        "--registry-baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="grandfathered built-in registry snapshot",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="freshness audit date in YYYY-MM-DD form (default: today)",
    )
    args = parser.parse_args()
    try:
        policy = load_policy(args.policy)
        baseline = load_registry_baseline(args.registry_baseline)
        errors = validate_document(
            args.path,
            policy=policy,
            registry_baseline=baseline,
            as_of=args.as_of,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"candidate-pool validation failed: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"candidate-pool validation failed: {error}")
        return 1
    markdown = args.path.read_text(encoding="utf-8")
    print(
        f"candidate-pool validation passed under policy {policy['policy_version']}: "
        f"{len(parse_hard_gates(markdown))} gated candidates, "
        f"{len(parse_scorecard(markdown))} formally scored, "
        f"{len(parse_review_register(markdown))} review records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
