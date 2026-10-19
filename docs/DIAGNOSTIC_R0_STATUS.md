# R0 — Diagnostics library quality baseline

This milestone repairs the existing 0.9.0 framework before new diagnostic algorithms are admitted.

## Leakage-safe legacy RUL comparison

`runner/rul_runner.py` previously ranked candidates by repeatedly observing a held-out **test** bearing every epoch.
That path could silently select on test performance, invalidating final estimates.

The legacy comparison now derives a bearing-disjoint train/validation partition **only from the existing development bearings**.
It selects a candidate using the best validation RMSE and evaluates the sealed test loader **once for the chosen candidate**.
The script reports validation rankings; test metrics are recorded only for the selected candidate.
It requires at least two development bearings and identifiable `bearing_names` / `index_map` dataset attributes.
For stronger cross-bearing evidence, prefer the existing `group_cv` AutoML protocol with multiple folds, seeds and sealed final test.
This change does not claim paper-level RUL reproduction or multi-condition generalization.

## R0 acceptance

- GitHub Actions `quality` must pass lint, formatting, audits, all pytest tests and wheel build.
- `ddp-cpu-gloo` must pass independently.
- Every model report distinguishes interface/smoke evidence from dataset-specific benchmark evidence.
- New algorithm intake is deferred until R0 gates pass.

## Candidate-pool freshness gate

The October 19 CI uncovered ten October 15 stale review records. A narrow source/terms desk review is recorded in `docs/MODEL_CANDIDATE_POOL.md`, without changing admissions or scores. Follow-up technical reviews remain due November 19, 2026.

## Public-API snapshot reference environment

The 904-symbol manifest is environment-sensitive for deliberately re-exported native sklearn estimator classes and generic alias reflection. Inspecting the old and current manifest showed the same 904 exports, but sklearn 1.7.2 versus 1.8.0 changes `LogisticRegression` and `SGDClassifier` constructor defaults, and Python 3.10 versus 3.13 changes `MetricFunction` / `MetricState` alias introspection. These are not changes to torch_like implementations. R0 keeps the strict full public API manifest check and aligns the **baseline CI reference job** with Python 3.13, PyTorch 2.10.0 CPU and scikit-learn 1.8.0. Other supported environments require separate behavioral compatibility checks, not a baseline overwrite. No API replacement or deprecation exception is introduced.
