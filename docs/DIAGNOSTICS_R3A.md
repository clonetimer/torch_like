# R3a — Auditable Anomaly Detection Workflow

## Goal

Provide one **executable, narrow vertical slice** of industrial diagnosis:
healthy reference acquisition -> detector fit -> optionally calibrate on a
separate development validation set -> score the sealed test windows
exactly once -> report decisions and confusion matrix.

This workflow is for **window-level anomaly detection**. Fault type,
isolation, root cause, multi-fault coupling, and RUL estimation remain
separate tasks. This slice introduces no competing machine-learning runtime.

## Split requirements

`AnomalyPartition(signals, recording_ids, labels)` requires
`SignalBatch.sample_ids` for every window and a recording ID for every
window. Recording IDs refer to stable original recordings or device runs;
they must **not** be freshly generated independently per split. Binary
labels apply to *whole windows*, not time points or channels. Healthy
reference labels, when provided, must contain only zero.

`AnomalySplit(reference, test, validation=None)` rejects any overlap of
recording IDs or sample IDs between partitions. This protects against
splitting overlapping windows from a single recording into train and
test when recording IDs are honest. It does **not** cryptographically
verify the provenance, guarantee no duplicate physical content under
fraudulently renamed recordings, or detect related entities across
different runs. A future manifest and grouped time-boundary audit are
needed to establish stronger provenance.

## Threshold selection

`AnomalyDiagnosticWorkflow(detector, threshold_policy="native")`
preserves the native detector's finite threshold (Z-score, CUSUM,
EWMA, Isolation Forest). Validation labels are not needed.

With `threshold_policy="validation_f1"`, the workflow uses **only**
labeled validation windows from disjoint recordings to select the
threshold maximizing window-level F1 over a finite quantile grid.
The test labels do not participate in threshold fitting. Exact
tie-breaking prefers the higher threshold/fewer false alarms.

For time/channel-level scores, the declared workflow aggregation is
**maximum over time and channels** for each window, leaving one
score per input window. The aggregation is not a point-level metric
and does not claim event onset timestamps.

## Sealed evaluation semantics

`run(split)` marks the workflow as consumed before any model
execution. A second `run` is rejected, even when a first attempt
fails. The implementation calls `detector.fit` only with healthy
reference signals. Optional calibration only sees validation windows.
The final test partition is passed only once to `detector.score`
and never to `fit` or threshold optimization.

The resulting immutable `AnomalyEvaluation` stores sample IDs,
recording IDs, anomaly scores, Boolean decisions, TP/TN/FP/FN,
precision, recall, F1 and false-alarm rate. False-alarm rate is
`null` when there are no healthy test windows. Zero-denominator
precision, recall and F1 use the conventional zero value.

`AnomalyEvaluation.to_dict` provides JSON-compatible content for
audit logging. Evaluation labels are intentionally omitted from the
report payload; the confusion counts are aggregate evaluation results.

## Example

    from torch_like.diagnostics import (
        AnomalyDiagnosticWorkflow, AnomalyPartition, AnomalySplit,
        SklearnWindowAnomalyDetector, SignalBatch,
    )

    # reference_signals and sealed_test_signals must carry original sample IDs
    # and use the same channel names, physical units and sample rate.
    # Provide original recording IDs, not arbitrary per-split tokens.
    train = AnomalyPartition(reference_signals, recording_ids=train_run_ids)
    test = AnomalyPartition(sealed_test_signals, recording_ids=test_run_ids, labels=test_labels)
    result = AnomalyDiagnosticWorkflow(
        SklearnWindowAnomalyDetector(random_state=0)
    ).run(AnomalySplit(reference=train, test=test))
    record = result.to_dict()

`reference_signals`, IDs and test labels above are caller inputs;
this snippet is deliberately non-executable without user data.

## Acceptance and non-goals

- No train/test groups or sample IDs overlap.
- Native and validation-derived thresholds never read sealed-test labels
  before scoring.
- Scoring preserves the sample leading axis; score reduction is explicit.
- Strict one-time evaluation per workflow instance and serializable audit.
- Regression tests use both deterministic detectors and native sklearn.
- Full repository quality, coverage (>=83%), wheel and DDP CI are required.

**Limitations:** Per-window precision/recall/F1 are not event-level
detection metrics, a guarantee of generalization, or a production
false-alarm certification. Time-causal streaming state, event matching,
rolling-origin temporal splits and physical trace provenance remain on
the roadmap.
