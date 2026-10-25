# Diagnostics interop P1/P2 — AAS lessons without platform coupling

This is an **algorithm-library** change. It does not implement AAS storage,
dataset governance, dashboards, scheduling, experiment approval, label truth,
automatic calibration or application alarm policies.

## P1: machine-readable algorithm constructor parameters

\`DiagnosticAlgorithmSpec.parameters\` is opt-in. Legacy algorithms without
parameter declarations retain their existing keyword-argument behavior.

\`\`\`python
import json
from torch_like.diagnostics import DiagnosticRegistry, register_statistical_detectors

registry = DiagnosticRegistry()
register_statistical_detectors(registry)
descriptor = registry.get_spec("EWMADetector").to_dict()
print(json.dumps(descriptor, ensure_ascii=False, indent=2))
detector = registry.create("EWMADetector", alpha=0.2)
# registry.create("EWMADetector", alpha=0) -> ValueError, prior to factory creation
\`\`\`

Descriptors include names, primitive kinds, optional required/default fields,
inclusive or exclusive numeric bounds, choices and descriptions. Validation
does **not** coerce strings to numbers, booleans to integers, or guess parameter
types. The actual algorithm constructor is still responsible for additional
data-dependent numerical constraints. New parameter descriptors must exactly
match the constructor's accepted parameter names.

## P2: score is not an alarm decision

\`DiagnosticResult\` remains a raw typed algorithm result. A separate
\`DiagnosticDecision\` represents an **explicit** final decision:

- \`score_threshold\`: an anomaly \`DiagnosticResult\` plus an explicit scalar
  or same-shape threshold array. A threshold is never fitted on inference
  values. The result retains per-sample/per-time/per-channel resolution.
- \`native\`: algorithm-created binary anomaly labels or multiclass fault
  labels, optionally carrying the original diagnostic result. A generic
  threshold is forbidden.
- \`label\`: direct algorithm labels, without a competing score or generic
  threshold.

\`\`\`python
import numpy as np
from torch_like.diagnostics import DiagnosticDecision, DiagnosticResult

raw = DiagnosticResult(
    "anomaly_detection", "anomaly_score",
    np.array([[0.1, 3.0], [4.0, 0.5]]), "MyDetector",
)
decision = DiagnosticDecision.from_threshold(raw, 2.0)
assert decision.labels.tolist() == [[0, 1], [1, 0]]

classification = DiagnosticDecision.from_labels(
    task="fault_classification",
    algorithm_name="MyClassifier",
    labels=np.array([0, 2, 1]),
    label_names=("normal", "fault_a", "fault_b"),
)
\`\`\`

No implicit max/mean pooling, anomalous/nonzero collapse of multiclass
labels, posterior probability invention or decision override is performed.
Applications still choose training references, score calibration samples,
threshold policies and evaluation metrics.

## Compatibility and release boundary

This is an additive full-framework API proposal. Existing \`SignalBatch\`,
\`DiagnosticResult\`, \`DiagnosticPipeline\`, RUL model contracts and native
PyTorch/sklearn estimators are not redefined. P1 schemas participate only
when declared. P2 decisions remain optional and do not change pipeline's
existing \`DiagnosticResult\`-only composition.

The **already published** \`torch-like-algorithms 0.2.0\` artifacts remain
unchanged. The standalone package build profile is not promoted by this PR;
any subsequent stable release must receive a **new version**, refreshed
allowlist/exports and installed-wheel tests. In particular, this branch's
P2 result is not yet available through \`torch_like_stable\`.

## Deferred phases

- **P3:** stateful streaming threshold/estimator contracts, checkpoint
  snapshots, restoration and segment-boundary behavior. No silent carry-over
  from independent CUSUM/EWMA windows.
- **P4:** compare AAS adaptations against upstream models numerically and
  replace copies with version-pinned adapters only after equivalence tests.
- **P5:** optional versioned native C ABI for multichannel windows, typed
  output, state, errors and cross-platform compatibility. Do not transplant
  AAS's one-dimensional \`double[N]\` ABI as the core model interface.
