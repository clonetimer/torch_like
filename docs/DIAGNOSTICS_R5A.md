# R5a — Health-assessment algorithm primitives

## Purpose and ownership

`torch_like.diagnostics` is a reusable fault-diagnosis and health-assessment
**algorithm library**. The caller owns telemetry acquisition, source authenticity,
labels, healthy-reference selection, validation plans and application policy.
There is no mandatory dataset manifest, provenance attestation, dataset admission
review, platform-managed data warehouse or background data verification.

The library checks only constraints **required for numerically meaningful
algorithm execution**: nonempty finite tensors, signal dimensions, reference
metadata compatibility, and fitted-state/parameter validity. Optional R3/R4
evaluation workflows are not prerequisites to an ordinary health-index call.

## Available models

- `GaussianHealthIndex`: after a caller-provided healthy-reference fit, take the
  RMS of per-channel standardized deviation over each input window.
- `MahalanobisHealthIndex`: use the same standardized reference plus a regularized
  cross-channel covariance precision matrix. `shrinkage` in (0, 1] shrinks
  covariance toward identity to avoid singularity; shrinkage=1 reproduces the
  Gaussian index.

Both report one index per input window:

```text
reference standardized residual z[t,c] = (x[t,c] - reference_mean[c]) / reference_std[c]
Gaussian deviation d = sqrt(mean_{t,c}( z[t,c]^2 ))
Mahalanobis deviation d = sqrt(mean_t(z[t,:]^T precision z[t,:]) / C)
HI = exp(-d / health_scale)
```

`health_scale` is an explicitly chosen, **dimensionless index-mapping scale**.
`min_std` bounds the baseline standard deviation. The map guarantees HI in
[0, 1] with **larger HI meaning greater resemblance to the selected healthy
reference**. HI=1 is only exact agreement with reference means, and normal
reference fluctuations produce HI below 1. HI is **not** a measured percentage
of physical remaining health, calibrated failure probability, severity
grade, causal explanation or predicted remaining useful life. No implicit
engineering threshold or health-state category is selected.

## Simple use

```python
import numpy as np
from torch_like.diagnostics import (
    GaussianHealthIndex,
    MahalanobisHealthIndex,
    SignalBatch,
)

# The application selects reference measurements and owns their correctness.
healthy = SignalBatch(
    np.random.default_rng(3).normal(size=(20, 128, 2)),
    sample_rate_hz=100,
    channel_names=("vibration", "temperature"),
    units=("g", "C"),
)
new_signals = SignalBatch(
    np.random.default_rng(4).normal(size=(4, 128, 2)),
    sample_rate_hz=100,
    channel_names=("vibration", "temperature"),
    units=("g", "C"),
)

estimator = GaussianHealthIndex(health_scale=1.2).fit(healthy)
result = estimator.score(new_signals)
assert result.task == "health_assessment"
assert result.output_kind == "health_index"
health_index = result.values  # (4,), larger=closer to reference
raw_deviation = estimator.deviation(new_signals)  # (4,), larger=farther

correlation_aware = MahalanobisHealthIndex(shrinkage=0.1).fit(healthy)
health_index_2 = correlation_aware.score(new_signals).values
```

Opt-in registry registration:

```python
from torch_like.diagnostics import DiagnosticRegistry, register_health_index_estimators

registry = DiagnosticRegistry()
register_health_index_estimators(registry)
detector = registry.create("GaussianHealthIndex", health_scale=2.0)
```

## Limitations and future algorithms

These two baselines are *reference-distance algorithms*, not learned physical
degradation/RUL predictors. They do not infer operating mode, aggregate trends
between independent input windows, apply mandatory alarm persistence protocols,
identify root causes, or certify a user's input source. Callers must decide
whether the reference is representative, whether operating conditions should be
stratified, how windows are produced, and what action any observed HI requires.

Next milestones are fault/health algorithm compositions, explicit health-state
classifiers, feature-wise contribution explanations, and adapters around the
existing RUL model family. Each requires its own mathematical and API tests.

CI requires pinned Ruff lint and format, public-API governance, full tests
with >=83% branch coverage, wheel, and CPU/Gloo DDP before merging.
