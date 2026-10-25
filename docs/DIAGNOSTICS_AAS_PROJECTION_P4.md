# P4 — AAS-to-torch_like score adapter, not a platform import

The attached AAS source keeps local copies/adaptations of five algorithms
(CUSUM, EWMA, PCA-SPE, USAD, TranAD) based on the upstream revision
`1e603915896e33fd9f326073c42a477718166b68`. AAS's
`BaseDetector`, thresholds, SQL storage, application reports and train/valid
task runner remain **application** responsibilities.

## What torch_like now provides

`project_anomaly_scores` makes AAS's essential *score-axis mapping*
explicit. A `DiagnosticResult` still preserves native score dimensions,
while `AnomalyScoreProjection` has a named `time` or `window` resolution:

| Score layout | Source shape | Continuous sequence (N=1) | Independent windows |
|---|---|---|---|
| `time_channel` | (N,L,C) | channel max/mean at each time => L | last timepoint, then channel max/mean => N |
| `time` | (N,L) | each timepoint => L | last timepoint => N |
| `window_channel` | (N,C) | unsupported | channel max/mean => N |
| `window` | (N,) | unsupported | values as-is => N |

No score layout or reduction is ever inferred from a 2-D shape. A reduction
for time/channel scores must be chosen explicitly. No alarm threshold is
derived or transferred implicitly, and no windows are silently joined.

### Example

```python
from torch_like.diagnostics import (
    CUSUMDetector, SignalBatch, project_anomaly_scores,
)

reference = SignalBatch(healthy_windows, sample_rate_hz=10)
algorithm = CUSUMDetector(k=0.5, h=5).fit(reference)

# One continuous segment is a single (1,T,C) SignalBatch.
continuous = SignalBatch.from_sequence(telemetry_segment, sample_rate_hz=10)
native_result = algorithm.score(continuous)
scores = project_anomaly_scores(
    native_result, score_layout="time_channel",
    sequence_mode="continuous", channel_reduction="max",
)
assert scores.resolution == "time"

# Independent evaluation windows are shape (N,L,C).
window_result = algorithm.score(independent_windows)
window_scores = project_anomaly_scores(
    window_result, score_layout="time_channel",
    sequence_mode="independent_windows", channel_reduction="max",
)
assert window_scores.resolution == "window"
```

## Replacement preconditions

- AAS must pin a compatible torch_like version (or future independently
  versioned algorithm-only package). Its current duplicated algorithms are
  **not** removed by this PR.
- CUSUM/EWMA must use equal baseline mean/std conventions, parameters,
  channel order and independent-window boundaries before comparing outputs.
  AAS uses per-time channel maximum for a continuous segment and last-time
  channel maximum per independent window.
- PCA-SPE expects sample-std normalization and a residual statistic;
  numerical tests should fix data, shape, PCA parameters and tolerances.
- USAD/TranAD architecture similarities do not establish *weight* or
  *checkpoint-format* compatibility. Native PyTorch weights require an
  explicit, tested key/shape mapping. Do not claim interchangeability until
  trained-model restoration and score parity tests pass.
- Application threshold policy, result mapping, labeling and task lifecycle
  remain in AAS. The core library does not depend on or import AAS.

## Verification boundary

Unit tests cover continuous and independent-window score projection and
numerical checks against direct CUSUM, EWMA, PCA-SPE output. They do not
claim end-to-end AAS equivalence or a migrated AAS delivery. A later
integration PR can replace each AAS copy only after an actual cross-project
fixture suite passes.

The published slim Stable 0.2.0 wheel is untouched; promotion requires a
new version and installed-wheel validation.
