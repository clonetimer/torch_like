# Non-stationary Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one admitted, tested Non-stationary Transformer forecasting model from the Workbuddy literature.

**Architecture:** Use model-private de-stationary attention, raw-input tau/delta projectors, and an encoder–decoder forecaster without changing shared attention or existing model behavior. Keep source/admission review, mathematical operator, forecaster and public registry integration in separately testable tasks.

**Tech Stack:** Python >=3.10, PyTorch >=2, pytest, ruff, existing lazy-registry generator.

**Spec:** `docs/superpowers/specs/2026-10-31-nonstationary-transformer-design.md`

## Global Constraints

- Preserve original 161-line grandfathered model baseline and all existing 912 protected public API records; additive exports only.
- No runtime dependency, wheel/Stable whitelist, version, workflow, or public shared-attention API change.
- The `NonstationaryTransformer` registry key and `paper_aligned_portable_subset` integration level are binding.
- Forecast shape `(B,pred_len,c_out)`, `enc_in==dec_in==c_out`, no use of `x_dec` values or future target statistics.
- External masks mean True=allowed, decoder self attention remains causal, encoder/cross factor positions match the spec.
- Model maturity and dataset benchmark evidence remain `unassessed`.
- Maintain bilingual docs headings/essential descriptions and exact source/official pins.
- Write tests before implementation, observe the expected failure, then minimal implementation and green test.

## Review Focus

1. A user passes a nonconstant per-key delta to encoder/cross attention: its effect must not cancel as a per-query bias (Task 2 tests).
2. A user supplies a lower-triangular decoder mask plus a fully blocked row: future positions must not leak and invalid rows must fail clearly (Task 2 tests).
3. A user changes future `x_dec` values to NaN with fixed observed inputs: the forecast must be unchanged (Task 4 tests).
4. A user sets `label_len=0`, passes constant inputs, or moves model and data to float64: output must remain finite and correctly shaped (Task 4 tests).
5. A user imports through root facade and saves/loads a model: laziness, identity, serialization and unchanged admission baseline must hold (Task 5 tests).

---

### Task 1: Formal discovery, hard gates, scoring and core admission

**Files:** Modify `docs/MODEL_CANDIDATE_POOL.md`, `tests/test_candidate_pool.py`; create no model code.

**Interfaces:** Produces `core` authorization for key `NonstationaryTransformer` with evidence A (primary NeurIPS 2022, official pinned MIT code, independent TFB Table 6 and DERITS Table 1). Do not declare `implemented` yet.

- [ ] **Step 1: Establish the current governance gate.** Run `/workspace/scratch/5f17efc96b8c/torch-env/bin/python -m tools.validate_candidate_pool --as-of 2026-10-31`. Expected: existing 18 gated, 16 scored, 18 review records; no failures.
- [ ] **Step 2: Edit the governance parser test** from 18 to 19 gated and add an assertion that the new scorecard decision is `core`; run `pytest -q tests/test_candidate_pool.py`. Expected: failure while the new candidate is absent.
- [ ] **Step 3: Record discovered candidate** from `report-2026-06-25.md`, preserving primary/official/independent provenance and limited protocols; add six hard-gate checks and review owner/deadline 2027-01-31.
- [ ] **Step 4: Score only after the passing gate.** R/P/U/E/I/A/X = 24/16/14/7/8/9/9, value 87, evidence A 1.0, risk 6, final 81. Decision `core`, not yet implemented. Document dimension rationales and risks (dense quadratic attention, factor numerical risk, API/reference differences), peers `Informer` and `ITransformer` and missing local benchmarks; run `pytest -q tests/test_candidate_pool.py` and validator. Expected: all green, 19 gated/17 scored/19 reviews.
- [ ] **Step 5: Commit** the admission record and tests separately (`docs: admit Non-stationary Transformer candidate`).

### Task 2: Independent de-stationary attention operator

**Files:** Create `torch_like/models/forecasting/nonstationary_attention.py` and `tests/test_nonstationary_attention.py`.

**Interfaces:** Produce `_DeStationaryAttention(dropout: float = 0.0)` accepting `q,k,v` shaped `(B,heads,Q,d)`, `(B,heads,K,d)`, `(B,heads,K,dv)` plus `tau: (B,)`, optional `delta: (B,K)`, keep mask and causal flag; return `(B,heads,Q,dv)`. Produce `_DeStationaryMultiHeadAttention(d_model,n_heads,dropout)` accepting `(B,Q,d_model)` etc and returning `(B,Q,d_model)` with projections.

- [ ] **Step 1: Write failing attention tests** for a direct hand-evaluated float64 softmax oracle using `Q!=K` and nonconstant delta, `tau=1/delta=0` normal attention, delta key-bias effect and uniform key-bias invariance; mask True=keep and exact lower triangle; rejected all-blocked row, invalid factor/mask shapes, and finite `q/k/v/tau/delta` derivatives.
- [ ] **Step 2: Run** `pytest -q tests/test_nonstationary_attention.py`. Expected: import failure because the module does not exist.
- [ ] **Step 3: Implement** `_DeStationaryAttention.forward` and `_DeStationaryMultiHeadAttention.forward` using dense PyTorch score matmul, factor broadcasting, key mask intersection and no NaN-producing all-masked softmax; use ordinary `nn.Linear` Q/K/V/out projections.
- [ ] **Step 4: Run** the same test module. Expected: green.
- [ ] **Step 5: Commit** (`feat: implement de-stationary temporal attention`).

### Task 3: Learned raw-window scale and shift projectors

**Files:** Create/extend `torch_like/models/forecasting/nonstationary_transformer.py`; create `tests/test_nonstationary_projectors.py`.

**Interfaces:** Produce `_DeStationaryProjector(seq_len,channels,out_features,hidden)` where `forward(raw,stats)` uses circular feature-axis Conv1d, concatenates detached `raw` and `stats`, then two-hidden-layer MLP. Keep exponentiation in model (`tau = exp(clamp(log_tau,-15,15))`).

- [ ] **Step 1: Write failing tests** for projector output shapes, circular feature-axis convolution fixture, raw/stat detachment, finite projector weight gradients, and `seq_len`-length delta.
- [ ] **Step 2: Run** `pytest -q tests/test_nonstationary_projectors.py`. Expected: `_DeStationaryProjector` missing.
- [ ] **Step 3: Implement** the private projector with `Conv1d(seq_len,1,3,padding=1,padding_mode='circular')` and `Linear(2*channels,hidden)->ReLU->Linear(hidden,hidden)->ReLU->Linear(hidden,out_features)`.
- [ ] **Step 4: Run** the projector module tests. Expected: green.
- [ ] **Step 5: Commit** (`feat: add non-stationary factor projectors`).

### Task 4: Native encoder–decoder forecasting model

**Files:** Extend `torch_like/models/forecasting/nonstationary_transformer.py`; create `tests/test_nonstationary_transformer.py`.

**Interfaces:** Produce `NonstationaryTransformerConfig(ForecastingConfig)` with `factor_hidden=128`, `eps=1e-5`, and `NonstationaryTransformer(ForecastingModel)` with common config constructor plus these two parameters, tensor `forward`, inherited `predict`, full config/state semantics.

- [ ] **Step 1: Write failing tests** for `from_config`/float64 state roundtrip, `(B,H,C)` and Adam gradient updates, population variance and independent denormalization, constant inputs, `label_len=0/seq_len`, `x_dec` future poisoning, optional 4-column time markers, explicit mask validation, causal decoder invariance and invalid config/input dimensions.
- [ ] **Step 2: Run** `pytest -q tests/test_nonstationary_transformer.py`. Expected: model/config imports fail.
- [ ] **Step 3: Implement** sinusoidal position/value/time embedding; encoder with tau+delta self attention; decoder with tau-only causal self and tau+delta cross attention; residual/FFN/layer norm; zero-future normalized decoder; observed-only normalization and denormalization; documented shape/dtype checks.
- [ ] **Step 4: Run** all three Nonstationary test modules. Expected: green.
- [ ] **Step 5: Commit** (`feat(forecasting): add NonstationaryTransformer`).

### Task 5: Public catalog, admission completion and integration

**Files:** Modify `torch_like/models/forecasting/_catalog.py`, `docs/MODEL_CANDIDATE_POOL.md`, `docs/WORKBUDDY_FORECASTING_INTAKE.md`; create `docs/WORKBUDDY_FORECASTING_R2.md`, `tests/test_workbuddy_nonstationary_integration.py`; regenerate `torch_like/models/_domain_exports.py`, `_public_exports.py`, `_builtin_index.py` and additive `governance/public_api_baseline.json` using existing tools.

**Interfaces:** Expose registry `NonstationaryTransformer`, config and class in three facades, with `integration_level='paper_aligned_portable_subset'`, full runtime capability and `unassessed` maturity/evidence. Candidate record may now say `core, implemented`; 35 report matches become 36, remaining 61 become 60.

- [ ] **Step 1: Write failing factory/public runtime tests** for model creation, predict, `torch_like.save_model/load_model`, lazy export identity, unchanged 161-entry historic baseline, and all old protected API records.
- [ ] **Step 2: Run** `pytest -q tests/test_workbuddy_nonstationary_integration.py`. Expected: `NonstationaryTransformer` missing from catalog/public export.
- [ ] **Step 3: Add** catalog metadata and `__all__` entries, regenerate three lazy indices, run API baseline audit `--write`; update core record to `implemented` with real registry and scope; write bilingual R2 notes and update inventory counts (do not rewrite historical tables).
- [ ] **Step 4: Run** integration, model, candidate-policy, lazy index, catalog and API checks. Expected: all green and 912 previous protected symbols unchanged.
- [ ] **Step 5: Run** the complete regression `pytest -q`, ruff check and format check, full/Stable wheel build and isolated import smoke. Expected: no failures, all evidence reported with actual commands and exit status.
- [ ] **Step 6: Commit** (`feat(forecasting): register and document NonstationaryTransformer`).

### Task 6: Independent branch verification and remote PR

**Files:** No feature code beyond focused fixes substantiated by failing tests; local verification record may be added under `validation/`.

- [ ] **Step 1: Self-review** every requirement in the spec against tests, actual catalog state, documented limits, and git diff; check old baseline unchanged.
- [ ] **Step 2: Obtain a fresh independent code review** of the complete diff. Fix Critical/Important findings with RED→GREEN evidence, then repeat affected/full checks; defer minor issues explicitly.
- [ ] **Step 3: Publish feature branch and PR** through authorized GitHub integration against current `main`, with exact base/head SHA lease and no unrelated PR/release/Stable change.
- [ ] **Step 4: Read remote CI results** for quality, CPU/Gloo DDP and Stable wheel; do not claim success until each run completed successfully.
- [ ] **Step 5: Report** PR, SHA, test results, admissions, model count and deferred risks. Do not merge this new PR without its own authorization.

