# torch_like Algorithm Library — P0–P3 convergence

This project is an **algorithm and interface library**, not AAS or any
other telemetry-evaluation platform. No database ingestion, application
label policy, alarm debounce, experiment scheduler, UI or dataset
provenance certification belongs in the slim delivery.

## User-facing contract: native algorithms first

| Family | Direct inputs | Fit/prepare | Primary caller output |
|---|---|---|---|
| Statistical detection | `SignalBatch (N,L,C)` | `fit(reference)` | `score(query): DiagnosticResult` |
| sklearn ML detection | `SignalBatch (N,L,C)` | `fit(reference)` | `score(query): DiagnosticResult` |
| sklearn fault classification | `SignalBatch (N,L,C)` | `fit(windows, labels, schema)` | `predict_proba(query): DiagnosticResult` |
| Health assessment | `SignalBatch (N,L,C)` | `fit(reference)` | `score(query): DiagnosticResult` |
| Neural fault classification | Native torch tensors / declared model batch | PyTorch optimizer or native class adapter | Raw logits or typed classifier result; do not claim automatic calibration |
| RUL regression | `SignalBatch (N,L,C)` | `RULRegressor.fit(windows, targets)` or externally loaded weights + `mark_ready()` | `predict(): DiagnosticResult` or `predict_with_unit(): RULForecast` |

All typed results preserve task, output kind, algorithm name and
sample-leading shape. No implicit conversion of anomaly scores into
fault probabilities or physical failure causes occurs.

## Explicit optional helpers

- `DiagnosticRegistry` provides opt-in construction, capability descriptions
  and strict validation *only when a parameter schema is declared*. The
  statistical CUSUM `k` allowance is **nonnegative**, including zero,
  consistent with the native constructor.
- `DiagnosticDecision` applies an explicit user/algorithm-selected
  decision; the caller still determines calibration reference and policy.
- `StreamingThreshold` owns only its threshold state, requiring explicit
  sample continuity and checkpoints. This does **not** checkpoint a neural
  optimizer or override per-window CUSUM/EWMA mathematical semantics.
- `project_anomaly_scores` requires an explicit score layout and an
  explicit continuous-time or independent-window resolution.
- Native C ABI v1 is a trusted, optional full-framework extension for
  dense `float64 (N,L,C)` anomaly-score plugins. It is excluded from the
  slim algorithms wheel.

## Convergence acceptance

**P0 — CI:** a hosted-runner failure with zero steps is not a test result.
Until actual Quality, DDP and Stable Wheel jobs execute and pass, no new
release is declared verified. See `docs/CI_RUNNER_START_TRIAGE.md`.

**P1 — v0.3.0 isolated distribution:** stage a deterministic allowlist,
explicitly version the new distribution, run an installed-wheel smoke
outside the source repository and preserve previously published v0.2.0.
The 0.3.0 candidate adds diagnostics parameter/decision/stream/projection
contracts but no experimental training platform or native DLL loader.

**P2 — consistent calls:** preserve native torch/sklearn model semantics;
provide common typed `SignalBatch` / `DiagnosticResult` wherever the
algorithm contract supports it. The integration test
`tests/test_diagnostics_user_surface.py` checks statistical, ML,
classification, health, RUL and score-only composition.

**P3 — numerical correctness and reproducibility:** maintain deterministic
CUSUM/EWMA/PCA-SPE/USAD/TranAD reference checks (AAS only serves as an
independent historical numerical oracle), guard against test-time
calibration, verify continuous/window boundaries, checkpoint atomicity
and ABI smoke on Linux/Windows. Training stability and performance beyond
those tested cases should never be asserted without further evidence.

## Local release-candidate verification (Linux and Windows)

From the repository root using a Python environment with the project's
algorithm dependencies plus `build` and `wheel`:

```shell
python packaging/stable/verify_local.py --report stable-v0.3.0-local-report.json
```

This stages v0.3, builds the wheel and executes the **installed**
`torch_like_stable` smoke inside a new temporary virtual environment,
using preinstalled system dependencies. No network download is triggered
by the script itself. A local pass is useful fallback evidence while
CI runners are blocked, but does not replace server-side checks.

## Freeze conditions

Stop new functional development unless a reproducible bug, compatibility
failure or concrete algorithm demand justifies it. Existing Stable v0.2.0
artifacts remain immutable. Do not merge the v0.3.0 candidate, claim
cross-platform validation or publish a GitHub Release until the required
gates have genuine passing outputs.
