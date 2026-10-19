# R1 — Industrial Diagnostics Contract Foundation

The R1 diagnostic namespace adds task-neutral contracts on top of the existing torch_like framework. It does not rewrite the PyTorch model registry, subclass sklearn estimators, or claim production diagnosis capabilities.

## Public types and responsibilities

- SignalBatch: explicit N×L×C sample/time/channel dimensions, acquisition sampling frequency in Hz, optional physical units, channel names, sample identifiers and timestamps in seconds; nonfinite signal handling is a deliberate input policy.
- DiagnosticResult: sample-leading prediction with declared task, output semantics, threshold (when applicable), algorithm provenance and class labels. It is not a carrier for ground-truth fault labels.
- DiagnosticAlgorithmSpec: task, backend, output kind, supported operations, input kind and online support declaration.
- DiagnosticRegistry: collision-safe factory lookup with native object return. Does not force sklearn estimators to masquerade as nn.Module or vice versa.

The registry intentionally starts empty; it is not evidence that any as-yet-unregistered detector has been numerically validated.

## Backend and leakage boundaries

- Input data must be valid before diagnostic inference; 2-D arrays require an explicit single-sequence conversion.
- A sample rate is never inferred from sequence length. Future frequency-domain algorithms must use Hz derived from this metadata.
- Time axes are strictly increasing within each sample. NaN is only accepted when allow_missing=True, indicating that a separate cleaning stage is needed.
- Output contracts keep predictions separate from training labels and test-set supervision.
- Native backends are instantiated lazily by registered factories. Their native fit/predict signatures are not changed.
- R1 adds no new root torch_like exports, so the existing protected baseline remains stable.

## Acceptance criteria

- Regression tests cover shape and rate validation, missing data, time monotonicity, class probability and RUL semantics, duplicate registry names, and native sklearn/PyTorch factories.
- Existing full quality and DDP gates remain mandatory.
- R2 may introduce numeric statistical detectors and feature operators, but each requires a reference calculation, synthetic anomalous signals and leakage-safe threshold tests.

## Next work

R2: calibrated Z-score, EWMA, CUSUM, PCA T-squared and SPE/Q; frequency features with physical sampling rate; separate train-only threshold estimation and inference scoring.
