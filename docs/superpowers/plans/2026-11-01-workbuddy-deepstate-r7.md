# Workbuddy R7 DeepState Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a formally admitted native DeepState probabilistic forecaster and close the two other R6 priority reviews without adding below-threshold models.

**Architecture:** A private structural-state-space numerical module supports a public univariate DeepState model and explicit batch/config classes. Existing registry,runtime and serialization contracts receive additive integration only.

**Tech Stack:** Python,existing PyTorch,numpy/pytest test dependencies; no new package dependency.

**Spec:** docs/superpowers/specs/2026-11-01-workbuddy-deepstate-r7-design.md

## Global Constraints

- Require enc_in == dec_in == c_out == 1; Gaussian,float32/float64,dropout0 only.
- No inferred calendars,categories,future targets or observed masks.
- Preserve emit -> update -> transition,rank-one state innovations and residual LSTM layers.
- Preserve shared batch schemas,admission policy,grandfathered161,Stable packaging,version,workflows and PR30.
- All new code is native PyTorch; no pretrained assets,datasets or new dependencies.
- DeepState admission observation68,not core; maturity and benchmark evidence unassessed.
- Paloma50.25 and SEER60.5 remain no-code; R6 snapshot is historical and unchanged.

## Review Focus

- Masked NaN/Inf must have no influence or gradient contamination; all-missing context is legal but empty training labels are not.
- Nonidentity transition at the history/future boundary must not be applied twice; sample paths must retain temporal covariance.
- Caller-supplied mutable config and changed default dtype must not corrupt reconstruction or numerical bounds.
- Decoder target values and ignored prefixes must not leak into parameter generation,predictions or scaling.
- Runtime and training dataclass/device adaptation must preserve every custom mask/calendar/category field.

## File structure

- Create torch_like/models/forecasting/deepstate_layers.py: structural coefficients,Kalman likelihood,marginals,joint sampling,residual LSTM.
- Create torch_like/models/forecasting/deepstate.py: public config/batch/model,input validation,scaling,loss and sampling APIs.
- Modify torch_like/models/forecasting/_catalog.py and generated _domain_exports.py, _public_exports.py and _builtin_index.py: additive model/config/batch exports and metadata; these are the actual generator destinations.
- Modify torch_like/contracts/inspect.py: narrow DeepState template only.
- Modify docs/MODEL_CANDIDATE_POOL.md,README.md,docs/WORKBUDDY_FORECASTING_INTAKE.md; create docs/WORKBUDDY_FORECASTING_R7.md and NOTICE; include NOTICE via root pyproject.toml license-files only (no Stable packaging change).
- Regenerate governance/model_catalog_audit.json, docs/MODEL_CATALOG_AUDIT.md and governance/public_api_baseline.json only via existing tools; preserve all preexisting records.
- Create tests/test_deepstate_layers.py,tests/test_deepstate.py,tests/test_workbuddy_deepstate_integration.py; extend tests/test_candidate_pool.py for three new scored records.

### Task 1: state-space mathematics

**Interfaces:** Produces private `_ISSM(periods,add_trend,dtype)` with transition buffer,state_dim and emission(indices,reference); `_kalman_filter(targets,observed,mean,cov,emission,transition,innovation,noise_std,residual)` returning log_prob[B,T],next mean[B,D],next covariance[B,D,D]; `_marginals(...)` returning mean/variance[B,T,1]; `_sample_paths(...,num_samples)` returning[S,B,T,1]; `_ResidualLSTM(input_size,hidden_size,num_layers)` returning output and per-layer states.

- [x] Write tests before implementation. Literal trend/seasonal coefficients; scalar update mean1.6,cov.8 from prior mean0,var4,target2,noise1; transition propagation; missing-middle sequence; full-joint Gaussian NLL vs sequential filter; rank-one cross-block innovation; float64 gradcheck; joint sample covariance; split-vs-concatenated recurrent continuation; bounded Cholesky behavior.
- [x] Run `/workspace/scratch/5f17efc96b8c/torch-env/bin/python -m pytest -q tests/test_deepstate_layers.py`; expect intentional missing-feature assertion failures,not collection errors.
- [x] Implement the declared private interfaces,including Joseph/symmetry and observed masking before arithmetic. Derive all expected matrices in tests independently from implementation helpers.
- [x] Rerun the task test command; expect all tests pass,finite gradients and sample-covariance tolerance.
- [x] Commit only task files plus design/plan: `feat(forecasting): add native DeepState state-space primitives`.

### Task 2: public model contract

**Consumes:** Task1 interfaces. **Produces:** DeepStateConfig and DeepStateBatch exactly as spec; DeepState forward,predict,predict_distribution,sample,fit_step,context_nll,make_probabilistic_forecast_batch,config.

- [x] Write failing public-model tests: default/seasonal/static-categorical forward and optimizer; direct first-future boundary oracle; covariate-only heads; dense joint conditional NLL; context objective; observed masks and NaN/Inf invariance; target/prefix isolation; scaling/Jacobian; all-missing context and future rules; parameter/config/bounds validation; exact dtype/owned-list roundtrips; autocast safety.
- [x] Run `/workspace/scratch/5f17efc96b8c/torch-env/bin/python -m pytest -q tests/test_deepstate.py`; expect missing-model assertion failures.
- [x] Implement public types/model with strict input validation and explicit narrow adaptations in docstrings. No persistent posterior session state. Mask targets with where before arithmetic; do not feed observations or scale into RNN. Include both context and conditional-future objectives,not independent-marginal NLL.
- [x] Run both DeepState test files; expect pass. Audit backward reaches all active heads and no ignored future value changes predictions.
- [x] Commit: `feat(forecasting): implement DeepState probabilistic batch and likelihood contract`.

### Task 3: admission and public integration

**Consumes:** Public DeepState types. **Produces:** factory/three-namespace lazy exports,model-specific runtime batch descriptor,persistence,registered observation implementation,three reviewed priority records and current40/96 mapping.

- [x] Write failing integration tests for factory/runtime loss/distribution/sample; custom batch extras through training/device adapter; float32/64 public save-load including changed default dtype; three lazy namespaces; candidate validation and no grandfathered expansion.
- [x] Run `/workspace/scratch/5f17efc96b8c/torch-env/bin/python -m pytest -q tests/test_workbuddy_deepstate_integration.py`; expect absent-export/registration/descriptor failures.
- [x] Add catalog import/entry/exports and DeepState-only contract template. Use the existing export generator discovered via `rg --files tools`; do not hand-edit unrelated generated records.
- [x] Add all six gates,dimension rationales,review deadlines and full evidence/peer/risk records for DeepState,Paloma,SEER. Set DeepState implemented only after model/integration tests pass. Never count copied TAGI Table2 or official Paloma seeds as independent evidence.
- [x] Write R7 usage example and explicit API/claim boundaries,Apache reference attribution in NOTICE,README/intake current progress. Preserve historical R1-R6 content. Do not assertion-test human prose.
- [x] Regenerate public API/catalog audits with existing tool flags; verify new records additive and all old entries preserved.
- [x] Run DeepState tests plus forecasting/runtime/candidate/R6 checks,then all offline validators; expect pass.
- [x] Commit: `feat(forecasting): integrate admitted DeepState and close R7 priority reviews`.

### Task 4: verification and authorized publication

**Consumes:** Exact completed tree. **Produces:** reviewed/tested PR,merge and post-merge checks,with branch and PR30 protected.

- [ ] Run full pytest coverage with current repository flags (minimum83),Ruff lint/format,and real catalog/API/lazy/candidate/AutoML validators. Expect all pass; record exact counts/skips/coverage and commands.
- [ ] Dispatch one fresh most-capable whole-branch reviewer with spec,plan,ledger and exact base/head. Review numerical,API,evidence and risk boundaries.
- [ ] Re-grade findings by user impact. Fix Critical/Important once with RED->GREEN and full-suite verification; record deferred Minor items. No automatic second review.
- [ ] Publish only the exact tested tree on an actual-current-main-parented feature commit; compare Git blob/tree hashes and fresh main/PR30 leases. Do not push synthetic local history.
- [ ] Create/update PR and wait for every actually applicable quality/DDP/Stable workflow. Stable model path is expected to trigger this time; verify actual workflow configuration. Stop on failure/permission blocker,never bypass.
- [ ] Merge under existing user authorization only after real gates and fresh expected head/base verification; verify actual main tree and post-merge CI. Preserve PR30 and existing user files.
- [ ] Complete ledger and report added model,held candidates,tests,links,all rulings and deferred minors. No local benchmark or production-readiness claim.

## Plan self-review

Every mathematical/input/runtime requirement is owned by Task1,2 or3. Task2
consumes exact Task1 shapes; Task3 uses the named public classes and no shared
batch additions. Review Focus has concrete tests in all owning tasks. Existing
generator paths/CLI flags must be read before use rather than assumed.
