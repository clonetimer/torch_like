# Workbuddy R6 Value Triage Design / 第六批价值筛选设计

## Goal / 目标

中文：按用户“继续推进剩下的57个，价值较低就不添加”的授权，逐项审核
R5 剩余名单，区分值得继续研究、证据或接口暂缓、当前边际价值低、非独立
预测骨干四种情况。筛选不是正式准入，不为完成数量制造模型或分数。

English: Review every one of the 57 names remaining after R5. Separate promising
follow-up, unresolved holds, scope-relative low incremental value, and methods
outside the standalone native forecasting-backbone queue. Triage is not formal
admission, implementation, or an accuracy ranking.

## Deliverables / 交付

- A machine-readable 57-item evidence/disposition snapshot, with report dates,
  primary identities, source/permission uncertainty, two existing peers, concrete
  incremental value, rationale and reopening conditions.
- A small offline validator that checks the snapshot against the historical R5
  remaining-name table and current forecasting catalog; reports derived counts;
  fails on missing/duplicate names, malformed provenance and invalid states.
- Bilingual R6 outcome and a concise 57-row human table, linked from the README
  and intake. Correct stale current-progress wording without rewriting R1–R5.
- Tests of validator behavior with valid and deliberately broken fixtures,
  complete repository validation, fresh full suite and independent final review.

## Scope and invariants / 范围与不变量

中文：本批不把初筛结果登记成 `core`、`observation` 或 `rejected`，不修改
模型实现、公共 API、正式候选分数、依赖、版本、Stable 范围或工作流。
57/57 已筛选不等于57项已实现：映射仍为39/96，57项仍未映射。
进一步实现必须单独完成原有六项硬门槛、证据与评分及实现验收。

English: No new model or formal admission is implied by this tranche. Keep
171 models, 49 forecasting models, 928 API records, the grandfathered 161-name
baseline, formal policy/candidate pool, Stable, dependencies, version, workflows
and unrelated PR30 unchanged. A `review_next` item is a research priority only.
Missing permission/assets cannot be relabeled low scientific value. Methods
excluded from this backbone queue may still belong in separately scoped utility,
training, retrieval, calibration or classical-fitting work.

No weights, external datasets, source implementations or training recipes are
downloaded into the package. Search failures mean unverified evidence, not proof
of nonexistence. Mutable branches are not immutable source audits. Code and weight
licenses are separate; author benchmark claims are not local reproduction.

## Data contract / 数据契约

`governance/workbuddy_forecasting_triage_r6.json` has schema version 1, review
date, baseline report/mapping/catalog counts, a scope statement and `items`.
Each item carries the exact fields described in the implementation plan.
Decisions: `review_next`, `hold`, `skip_low_value`, `out_of_backbone_scope`.
License statuses: `verified`, `declared`, `unknown`, `mixed`. Optional URLs and
immutable 40-hex revisions use JSON null when not verified. The review contains
no formal score, evidence grade, admission or implementation flag.

The validator reads local files only. It does not contact source hosts, certify
license interpretation or assess prose truth. Current catalog peers are validated
against lightweight catalog metadata; no model implementation import is needed.
Unknown fields are rejected so later formal-admission claims cannot silently
appear in a triage record. Output explicitly keeps mapping and review counts apart.
Baseline counts describe R5, not a demand to keep the future catalog frozen;
additive later models must not force a historical snapshot rewrite.

## Acceptance / 验收

All 57 R5 names appear once, with no extra candidate or fabricated peer. Counts
are computed from items, not maintained as a second hand-written total. CLI
failure is nonzero and names the malformed field or coverage error. Real
repository validation and negative-fixture tests pass, as do existing policy,
API, catalog/lazy-index and AutoML checks. Full coverage stays at least83%.
The independent reviewer checks source/decision honesty beyond what schema tests
can establish. Authorized GitHub publication uses the actual fresh main tree,
CI and head/base leases; preserve PR30 and verify the merge and push workflows.
