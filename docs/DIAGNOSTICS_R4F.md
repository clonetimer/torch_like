# R4f — Sealed fault-classification cluster uncertainty

R4f adds `bootstrap_fault_generalization` as a **strictly post-hoc** tool for
R4c's already sealed `FaultGeneralizationEvaluation`. No raw signals, per-window
truth labels, model fitting, inference, threshold selection or model search are
permitted in this computation.

## Example

```python
from torch_like.diagnostics import bootstrap_fault_generalization

# evaluated: already completed FaultGeneralizationWorkflow.run(split).
recording_result = bootstrap_fault_generalization(
    evaluated,
    cluster_by="recording",
    n_resamples=2000,
    confidence=0.95,
    seed=17,
    min_clusters=3,
    min_class_clusters=2,
)
print(recording_result.to_dict())

# Only use this when the original R4c protocol truly held out devices.
device_result = bootstrap_fault_generalization(
    evaluated,
    cluster_by="device",
    n_resamples=2000,
    confidence=0.95,
    seed=17,
)
```

## What is resampled?

The unit is a **whole original recording** (default) or **whole physical
device**, never an individual overlapping time window. Each bootstrap draw
samples the same number of clusters with replacement; for each draw, the
recording-/device-level confusion matrices are added, giving a new integer
confusion table under the same fixed fault-class vocabulary. Accuracy and
`macro_f1_all_classes` are recomputed from that table.

Point estimates are exactly the *sealed original* evaluation's
window-weighted accuracy and macro-F1, not an average of per-recording scores.
The interval endpoints are the empirical quantiles at
`(1-confidence)/2` and `(1+confidence)/2`. Report results as **percentile
cluster-bootstrap intervals conditional on the observed source clusters**;
they are not population-wide guarantees.

Every label must occur in at least `min_class_clusters` independent source
clusters (default 2), and each analysis requires at least `min_clusters`
independent clusters (minimum 3). These are minimal engineering safeguards, not
sufficient statistical power for serious industrial claims. Report the actual
cluster counts. Missing classes in *bootstrap replicates* are not thrown away:
the fixed-universe macro-F1 gives them zero, and the machine-readable report
counts their frequency. Reject overall classes represented by too few true
independent sources, rather than hiding their uncertainty in averages.

## Recording versus device

- **Recording resampling** treats original recordings as the independent units
  for a **recording-conditional** uncertainty summary. It is not evidence that
  the classifier transfers to unseen devices, especially if all recordings
  are from one device.
- **Device resampling** is only permitted for R4c `holdout="device"` or
  `holdout="both"` and at least 3 independent held-out devices with
  sufficient class-bearing devices. Resampling the aggregate device confusion
  preserves the dependence of multiple recordings on the same device.
- Device/condition/pair/recording marginal confusion tables must all reconcile
  with the sealed overall confusion; actual source IDs, per-recording predicted
  class totals, count coverage and distinctness are audited. Duplicate or
  untrusted source IDs fail closed when detectable.
- The original R4c report contains only marginal domain slices; it is not
  a cryptographic provenance certificate and cannot prove that physical source
  identities weren't relabeled or duplicated upstream. This method deliberately
  does **not** offer an unverified hierarchical device→recording bootstrap.

## Scientific interpretation and limits

The metric is **window-weighted**, even when whole source clusters are sampled;
long recordings contribute more windows in each draw. A cluster-weighted
recording-average metric would be a different estimand and is not silently
substituted.

The percentile bootstrap does not adjust for model-selection bias, poor domain
coverage, distribution shift, unlabeled faults, or unreliable ground truth.
Sparse/imbalanced fault classes and small cluster counts can create extremely
unstable or degenerate intervals; check per-class source counts, missing-class
replicate counts and the full held-out domain table before citing any interval.
There is no claim of causal root-cause inference, calibrated severity,
precision of source-probability mapping or real-world transfer validation.

## Test and merge requirements

- A long-recording counterexample distinguishes record-bootstrap intervals
  from misleadingly narrow independent-window intervals.
- Synthetic tiny source tables test deterministic RNG, no-variance intervals,
  sparse sources, absent bootstrap classes, device constraints, corrupted
  source counts and duplicate identities.
- End-to-end integration verifies **zero extra fit/predict** calls after sealed
  evaluation and JSON-compatible reporting.
- Full locked Ruff, API stability, wheel, >=83% branch coverage and CPU/Gloo DDP
  checks must pass on the final PR head and be checked again after merge.

Next: independently collected multi-device telemetry trials with a physical
provenance manifest and genuinely held-out domain benchmarks.
