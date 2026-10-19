# R3 — Model-neutral anomaly event extraction

The first diagnostic-workflow increment converts **predicted anomaly scores**
into events while retaining their source algorithm, window ID, channel (if
available), threshold, score and declared temporal resolution. It does not
infer a fault label or root cause from an anomaly alarm.

## Explicit score resolution

- `window`, shape (N,): exactly one score for an entire input window.
  Output events deliberately contain **no** time-of-onset, offset or duration.
- `window_channel`, shape (N,C): one window-level score per channel; an
  explicit score layout is required.
- `time`, shape (N,L): a time-resolved scalar anomaly score; requires an
  explicit score layout because the two-dimensional shape is ambiguous.
- `time_channel`, shape (N,L,C): separate time-resolved scores per channel.

Time- and channel-resolved dimensions must exactly match the associated
SignalBatch. Scores must use the anomaly_detection/anomaly_score contract
with a **predeclared threshold**; uncalibrated PCA results fail closed.

## Grouping semantics

`min_active_points` specifies the minimum number of true threshold
exceedances in an episode. `gap_tolerance` permits up to the configured
number of inactive samples **between** active samples; tolerated inactive
points do not count toward the minimum. Episode ends use exclusive sample
indices, but `first_time_s` and `last_time_s` represent actual observation
times, not an invented end-exclusive event instant.

Explicit timestamps are used directly when present. Without timestamps,
point positions are nominally mapped to index / sample_rate_hz.
For window-level scores, debounce and gap options are invalid because the
source did not provide time-resolved evidence.

Episodes never merge across independent input windows. This is deliberate:
caller windows can overlap, represent different devices, or have different
start times. A separate stateful fusion layer with group/time provenance is
required before cross-window event linkage.

## Provenance and leakage boundary

Event extraction operates on prediction results only; no fault truth, test
label, training-set statistics or threshold optimization is involved.
It uses the threshold already carried by DiagnosticResult and never edits
the input arrays. Event data is useful for alert reporting; isolation,
decision fusion, expert rules, and root-cause logic remain future stages.

Tests verify channel-specific debounce, point-count semantics, explicit
timestamps, no invented window onset, layout disambiguation, strict
dimensionality, separate-window isolation, and fail-closed behavior.

## R3a evaluation bridge

`AnomalyEvaluation.as_diagnostic_result()` converts the already sealed
window-level scores and chosen threshold into a typed `DiagnosticResult`.
Use `extract_anomaly_events(report.as_diagnostic_result(), sealed_test.signals)`
to surface the **window-only** alarms, retaining source sample IDs.
This conversion does not refit the detector, repeat the test evaluation,
create a new threshold or fabricate point-level onset timestamps.

The window event export is a reporting artifact, not a new opportunity
to tune the threshold based on sealed test outcomes.
