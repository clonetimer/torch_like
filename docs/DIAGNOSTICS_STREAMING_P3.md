# P3 — Streaming diagnostic threshold state and checkpoint semantics

`torch_like.diagnostics.StreamingThreshold` is an **optional, directly
callable algorithm component**. No database, model runner, remote device,
source audit, application label policy, data partition or alert service is
included. It receives precomputed `DiagnosticResult` anomaly scores.

## Scope and inputs

Only an explicitly continuous **single sequence** of score chunks of shape
`(1, time, channels)` is supported. This avoids interpreting a batch of
independent windows as one continuous stream.

The caller specifies an absolute `start_index` for every chunk; it must
equal `observations` (the next expected index). Repeated/out-of-order chunks,
gaps, changing algorithm identity or channel count are rejected. If
`channel_names` were configured they must accompany each chunk in precisely
the original order. A new physical/independent sequence requires `reset()`
or a fresh instance; no hidden cross-run state is shared.

A score must already be computed by the upstream algorithm. The stream
component does not call `fit` or alter `CUSUMDetector`/`EWMADetector`
window-reset behavior. In particular, their existing independent-window
mathematics remain unchanged.

## Fixed threshold

```python
import numpy as np
from torch_like.diagnostics import DiagnosticResult, StreamingThreshold

stream = StreamingThreshold(
    algorithm_name="DetectorX", mode="fixed", thresholds=(2.0, 3.0),
    channel_names=("wheel_speed", "temperature"),
)
scores = DiagnosticResult(
    "anomaly_detection", "anomaly_score",
    np.array([[[1.0, 5.0], [3.0, 1.0]]]), "DetectorX",
)
decision = stream.decide_chunk(
    scores, start_index=0, channel_names=("wheel_speed", "temperature"),
)
print(decision.labels.tolist())  # [[[0, 1], [1, 0]]]
print(decision.thresholds.shape)  # (1, 2, 2)
```

The source score's optional static `threshold` is **not** treated as the
authority for this explicit decision; the declared stream threshold is.

## Adaptive EWMA threshold

The caller supplies `initial_mean` and `initial_std` from a prior,
independent reference. There is **no calibration using the current
evaluation scores or labels**. At every time `t`:

- Compare `score[t]` to the **existing** threshold
  `mean[t-1] + sigma * sqrt(variance[t-1])` (strictly greater).
- Emit the timestamp/channel label and the actual threshold used.
- Only then, optionally update the per-channel mean/variance:
  `mean += alpha * delta`;
  `variance = (1-alpha) * (variance + alpha * delta**2)`.
- With `freeze_on_alarm=True` (default), suspicious channels are not used
  to update their own reference. This is a declared diagnostic algorithm
  policy; callers can explicitly configure a different policy.

This is a transparent adaptive baseline, **not** a calibrated physical
failure probability or guaranteed concept-drift solution. A learning
algorithm's state, threshold state and policy are different entities.
Remainder thresholds and model states must not be silently conflated.

## JSON checkpoint and recovery

```python
import json
from torch_like.diagnostics import StreamingThreshold

stream = StreamingThreshold(
    algorithm_name="DetectorX", mode="adaptive_ewma",
    initial_mean=(0.0, 0.0), initial_std=(1.0, 1.0), sigma=3.0,
)
# stream.decide_chunk(precomputed_scores, start_index=0)
checkpoint = json.loads(json.dumps(stream.state_dict()))

recovered = StreamingThreshold(
    algorithm_name="DetectorX", mode="adaptive_ewma",
    initial_mean=(0.0, 0.0), initial_std=(1.0, 1.0), sigma=3.0,
)
recovered.load_state_dict(checkpoint)
# The next chunk must start at recovered.observations.
```

The checkpoint is a detached JSON-compatible mapping with schema version,
full explicit configuration, processed observation count and current
state. Restore validates all fields and is atomic; invalid snapshots cannot
partially overwrite a running instance. Load only caller-trusted checkpoint
data. Checkpoints do not claim authenticity or provenance verification.

## Contract limitations

- This module is *not* a generic detector checkpoint; it checkpoints the
  thresholding component. Future algorithms may implement the same
  `state_dict/load_state_dict/reset` pattern explicitly.
- Continuous sequence identity and monotonic timestamps remain the
  caller's responsibility; an absolute sample index checks only continuity.
- No interpolation, automatic missing-value fill, sample-level reconstruction
  from window scores, clustering of alarms or event postprocessing.
- One stream object is single-writer and should not be shared for concurrent
  mutation. Separate concurrent streams need separate instances.
- Published `torch-like-algorithms 0.2.0` remains unchanged until a new,
  versioned stable release is reviewed with installed-wheel tests.
