# Pyraformer R4 implementation plan

> Execute inline using test-driven development, then one independent whole-branch
> review. User has authorized continuation and merge; no repeated execution-choice
> question is needed.

Spec: `docs/superpowers/specs/2026-10-31-pyraformer-design.md`.
Local base: `68f8c3af282d4d510460f228199bd091215fc400` (tree equals GitHub main
`f4aae8ea6a00951869b55116775625b0d2d86258`).

## Global constraints

Only Pyraformer and its admission/docs/tests/generated indexes. Preserve all 920
existing API records, grandfathered 161-model baseline, Stable/version/workflows
and unrelated PR #30. Literal source reference is pinned in the spec. Admission
is observation69, not core; qualifier `implemented` only after contracts pass.
Every implementation batch starts with a real failing test. Use the existing
torch-env, one CPU thread, and localhost permission for full Gloo tests.

## Task 1 — admission and sparse operators

Files: candidate pool, `tests/test_candidate_pool.py`,
`tests/test_pyraformer_graph.py`, `tests/test_pyraformer_attention.py`,
`torch_like/models/forecasting/pyraformer_layers.py`.

Interfaces: `_pyramid_graph(fine_length, factors, inner_size)` returns
`(lengths, neighbors, valid, last_ancestors)`; `_PyramidalAttention` and
`_PyramidEncoderLayer` consume `(B,S,D)` plus neighbor/valid buffers.

1. Test admission count21, score69 observation; observe RED before doc addition.
2. Test literal length11 factors(3,2) graph and length23 factors(3,2,2) ragged
   boundaries, symmetry, self-membership, no duplicates, constant degree bound.
3. Test attention and encoder against independent dense oracle in double,
   weighted-output gradients for inputs and all parameters, extra-padding
   invariance, dropout train/eval behavior and gradcheck; observe RED.
4. Implement only these operators, then run the task tests; expected all PASS.
5. Commit admission/operators; ledger exact results.

## Task 2 — public forecasting model

Files: `pyraformer.py`, `test_pyraformer.py`, plus constructor in layers file.
Interfaces: standard ForecastingModel/ForecastingConfig, four-level graph and
encoder from Task1; future mark index is label_len, not zero.

1. Write RED tests for CSCM hand/dense-convolution oracle, ragged tail behavior,
   exact train/eval BN state, preflight degeneracy guard before state changes.
2. Implement CSCM and public model to the spec. Tests cover full dense oracle,
   output order, observed-only dependency, unused decoder data, invalid config,
   shape/dtype/device/finite boundaries, optimizer, AMP and position extensions.
3. Intercept real tensor operation outputs to reject ephemeral S-by-S tensors.
4. Run all Pyraformer tests; expected PASS. Commit and ledger results.

## Task 3 — integrate, review, verify and publish

Files: forecasting catalog/lazy exports, API/catalog snapshots, integration tests,
R4 scope doc, intake progress, README, admission qualifier.

1. Write RED public factory/lazy/runtime/persistence tests, implement registration,
   regenerate indexes and additive API snapshot. Expected 170 models/48
   forecasting/924 API symbols, all old920 unchanged, old161 unchanged.
2. Update R4 boundaries/evidence and intake: 38/96 mapped,58 remain. Run candidate,
   API/catalog/index,11 AutoML example, empty-run, Ruff/compile checks.
3. Full pytest coverage >=83%, full+Stable wheel builds, source-isolated installed
   runtime/save-load/Stable smoke. Expected PASS; do not silently ignore failures.
4. One fresh whole-branch review on the strongest available model. Triage by user
   impact; one RED→GREEN fix pass for Important/Critical, full rerun, no re-review.
5. Publish snapshot through GitData with actual remote main as parent; draft PR,
   quality/DDP/Stable CI all green; verify head/main leases, authorized merge,
   postmerge workflows and unchanged PR #30. No branch deletion requested.

## Review Focus

Check ragged parent/child edges versus dropped convolution tails; accidental
dense masks or S-by-S attention intermediates; padded attention probabilities
and gradients; first future mark and circular prediction-token embedding;
BatchNorm state on rejected forwards; serialization of integer/bool/float64
buffers; numerical overflow outside bounded oracle inputs; sparse versus full
model complexity claims; independence of TimesNet-M4 versus duplicate long-range
table values and the A69/B61.4 sensitivity; untouched existing API and Stable.
