# Workbuddy R6 Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Continue the established inline execution
> and authorized merge; do not re-ask routine choices. One fresh whole-branch
> review at the end. Research helpers do not implement repository changes.

**Goal:** Finish evidence-grounded dispositions for all57 R5 remaining names,
with an executable integrity check and no fake model/admission completion.

**Architecture:** A versioned JSON snapshot plus an offline standard-library
validator; bilingual human review links to the primary evidence. Existing
catalog and R5 table provide independent identity/coverage boundaries.

**Tech Stack:** Existing Python>=3.10, pytest and Ruff; no dependencies added.

**Spec:** `docs/superpowers/specs/2026-11-01-workbuddy-r6-triage-design.md`.

## Global Constraints

- Keep all57 original names and discovery dates; no formal score or admission.
- Preserve171 total/49 forecasting/928 API/161 grandfathered names and39/96
  mapping; no model implementation, Stable, version, dependency or workflow edits.
- Distinguish source/license/asset holds from low-value and task-role exclusions.
- Null means not verified; no invented pins, licenses, independent experiments
  or benchmark results. Primary URLs and exact-source evidence remain traceable.
- Local base93850259677f39b626b4d83c5eb68d88c980fff9 equals actual main
  1c0039181aafd1097fbdfdd2af06062f007ceece at tree
  da341dce3a453976916dbd7aebb72a5b5d862b6e. Preserve unrelated PR30.

## Review Focus

- No queue decision can masquerade as formal admission, implementation, matched
  Workbuddy coverage, local benchmark success or a universal quality judgment.
- A missing license is unknown permission, not proof that a model lacks value;
  public repository existence is not a complete released implementation.
- Global-fitting models and two-stage representations must not be incorrectly
  described as nonexistent algorithms merely because their task contracts differ.
- Independently check narrow high-impact source assertions: CTLite completeness,
  code versus weight terms, TStruct license correction and native numerical holds.
- Negative inputs must fail cleanly, including JSON nulls, wrong container types,
  booleans as counts, duplicate/missing inventory rows and false source pins.
- The validator checks structure, not scientific claims. Documentation must not
  present a successful schema check as experimental or permission certification.

---

### Task 1: Executable triage integrity contract

**Files:** Create `tests/test_workbuddy_triage.py`,
`tools/validate_workbuddy_triage.py`,
`governance/workbuddy_forecasting_triage_r6.json`.

**Interfaces:** `parse_remaining_names(text) -> set[str]`,
`validate_triage(data, expected_names, catalog_names) -> list[str]`,
`summarize_triage(data) -> dict`; CLI accepts `--triage`, `--inventory`, `--json`.
Default inventory is R5, catalog is the existing lightweight forecasting catalog.
Top-level fields: schema_version, reviewed_on, baseline, scope, items.
Item fields: name, report_date, canonical_title, task_role, decision,
reason_category, comparison_models, incremental_value, rationale, reopen_when,
primary_url, code_url, source_revision, code_license_status, code_license,
license_evidence_url, followup_priority, notes.

- [x] Write behavior tests first. Valid two-item fixtures return no errors and
  honest derived counts; corrupted types, rows, decisions, peers and provenance
  fail. CLI succeeds for real57 rows and fails on malformed data.
- [x] Run targeted tests and observe missing-validator failures. Implement the
  smallest offline validator and merge the reviewed57 records, then rerun.
- [x] Run `python -m pytest tests/test_workbuddy_triage.py -q` and
  `python tools/validate_workbuddy_triage.py --json`; Expected: PASS and57reviewed,
  57unmapped with no implementation count increase. Ruff/format; commit and ledger.

### Task 2: Bilingual outcome and corrected intake

**Files:** Create `docs/WORKBUDDY_FORECASTING_R6.md`; update README.md and
docs/WORKBUDDY_FORECASTING_INTAKE.md. Do not modify historical R5.

**Interfaces:** Human table consumes Task1 records. Source pins/license status
must agree; all57 names and decision/reopen reasons remain visible.

- [x] Write all57 concise rows, priority explanations, scoped exclusion reasons,
  source corrections, reproducible validator command and evidence limitations.
- [x] Update only current-progress intake/README wording and add links. Preserve
  historical mapping snapshots. Do not add catalog/API/admission entries.
- [x] Inspect the complete diff and compare table coverage/counts to CLI output.
  Expected: every57 record represented, existing model-governance behavior intact.
- [x] Run targeted triage plus candidate-pool tests and validators; Expected:
  PASS. Commit and ledger. Human prose has review, not brittle text-match tests.

### Task 3: Verification, independent review and authorized publication

**Files:** Only preceding task files and plan completion/evidence notes.

**Interfaces:** No runtime/public API change; snapshot must equal unchanged
baseline contracts. Publish tree parented to actual main, not synthetic history.

- [ ] Run Ruff/format, candidate/API/catalog/lazy-index and AutoML validators;
  full pytest with coverage>=83%; Expected: all PASS, no protected-file diff.
- [ ] Dispatch one fresh-context most-capable reviewer with spec/plan/ledger,
  exact branch range and Review Focus. Regrade; one RED→GREEN fix pass for
  Important/Critical, whole-suite rerun as needed; no second reviewer pass.
- [ ] Publish a draft PR from a tree-verified snapshot, verify quality/DDP/Stable
  CI and current head/main leases/comments, merge under existing authorization.
  Expected: actual merged tree equals verified local tree; three push workflows
  pass and unrelated PR30 remains unchanged. Record real outcomes, no predictions.

## Pre-flight / 预检

Task1 owns machine data/validation; Task2 consumes that data without changing
decisions secretly; Task3 verifies the same final snapshot. Research fragments
are scratch inputs only, never separate policy authorities. No conflicting
interface or shared implementation ownership exists.
