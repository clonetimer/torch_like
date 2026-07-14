"""Validate candidate-pool scoring, evidence grades, and admission decisions."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

EVIDENCE_COEFFICIENTS = {"A": 1.0, "B": 0.9, "C": 0.75, "D": 0.5}
DIMENSION_LIMITS = (25, 20, 15, 10, 10, 10, 10)
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
REQUIRED_RECORD_FIELDS = (
    "Category",
    "Core value",
    "Evidence",
    "Peer comparison",
    "Decision",
    "Conclusion reason",
    "Missing information",
)


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


def expected_decision(final: float, evidence: str) -> str:
    if final >= 80 and evidence in {"A", "B"}:
        return "core"
    if final >= 65:
        return "observation"
    return "rejected"


def validate_scores(rows: list[CandidateScore]) -> list[str]:
    """Return all scorecard violations instead of failing on the first row."""
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.model:
            errors.append("Scorecard contains an empty model name")
        elif row.model in seen:
            errors.append(f"{row.model}: duplicate scorecard row")
        seen.add(row.model)

        if len(row.dimensions) != len(DIMENSION_LIMITS):
            errors.append(f"{row.model}: expected seven dimensions")
            continue
        for label, score, limit in zip("RPUEIAX", row.dimensions, DIMENSION_LIMITS, strict=True):
            if not 0 <= score <= limit:
                errors.append(f"{row.model}: {label}={score} is outside 0..{limit}")
        calculated_value = sum(row.dimensions)
        if row.value != calculated_value:
            errors.append(f"{row.model}: value {row.value} != dimension sum {calculated_value}")
        if row.evidence not in EVIDENCE_COEFFICIENTS:
            errors.append(f"{row.model}: unknown evidence grade {row.evidence!r}")
            continue
        if row.risk < 0:
            errors.append(f"{row.model}: risk deduction must be non-negative")
        calculated_final = row.value * EVIDENCE_COEFFICIENTS[row.evidence] - row.risk
        if abs(row.final - calculated_final) > 0.051:
            errors.append(f"{row.model}: final {row.final:g} != calculated {calculated_final:g}")
        actual_decision = row.decision.split(",", maxsplit=1)[0].strip()
        required_decision = expected_decision(calculated_final, row.evidence)
        if actual_decision != required_decision:
            errors.append(
                f"{row.model}: decision {actual_decision!r} must be {required_decision!r} "
                f"for final={calculated_final:g}, evidence={row.evidence}"
            )
    return errors


def validate_document(path: Path) -> list[str]:
    markdown = path.read_text(encoding="utf-8")
    rows = parse_scorecard(markdown)
    gates = parse_hard_gates(markdown)
    errors = validate_scores(rows)
    errors.extend(_validate_scope(markdown, gates))
    errors.extend(_validate_hard_gates(gates, rows))
    rationale_models = _parse_rationale_models(markdown)
    for row in rows:
        if row.model not in rationale_models:
            errors.append(f"{row.model}: missing Dimension Rationale row")
    errors.extend(_validate_detailed_records(markdown, rows, gates))
    decisions = {row.decision.split(",", maxsplit=1)[0].strip() for row in rows}
    missing_decisions = {"core", "observation", "rejected"} - decisions
    if missing_decisions:
        errors.append(
            f"Scorecard must contain core, observation, and rejected results; missing {sorted(missing_decisions)}"
        )
    return errors


def _validate_scope(markdown: str, gates: list[HardGate]) -> list[str]:
    errors: list[str] = []
    target_match = re.search(r"^- Target domain: (.+)$", markdown, flags=re.MULTILINE)
    minimum_match = re.search(r"^- Minimum candidate count: (\d+)\.$", markdown, flags=re.MULTILINE)
    date_match = re.search(r"^- Evaluation date: (\d{4}-\d{2}-\d{2})\.$", markdown, flags=re.MULTILINE)
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
    return errors


def _validate_hard_gates(gates: list[HardGate], scores: list[CandidateScore]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for gate in gates:
        key = _model_key(gate.model)
        if key in seen:
            errors.append(f"{gate.model}: duplicate hard-gate row")
        seen.add(key)
        passed = gate.gate.startswith("pass")
        if gate.gate not in {"pass", "pass with risk", "pass with high risk", "fail"}:
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


def _validate_detailed_records(markdown: str, scores: list[CandidateScore], gates: list[HardGate]) -> list[str]:
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
            if score.evidence in {"A", "B", "C"} and primary_count < 1:
                errors.append(f"{score.model}: evidence grade {score.evidence} requires a labeled primary source")
            if score.evidence in {"A", "B", "C"} and official_count < 1:
                errors.append(f"{score.model}: evidence grade {score.evidence} requires a labeled official source")
            required_independent = 2 if score.evidence == "A" else 1 if score.evidence == "B" else 0
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
    args = parser.parse_args()
    try:
        errors = validate_document(args.path)
    except (OSError, ValueError) as exc:
        print(f"candidate-pool validation failed: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"candidate-pool validation failed: {error}")
        return 1
    markdown = args.path.read_text(encoding="utf-8")
    print(
        f"candidate-pool validation passed: {len(parse_hard_gates(markdown))} gated candidates, "
        f"{len(parse_scorecard(markdown))} formally scored"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
