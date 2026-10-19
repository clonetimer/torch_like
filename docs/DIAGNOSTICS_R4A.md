# R4a — Supervised single-label fault classification

## Task and boundary

R4a identifies one fault **type** per input window. It does not infer a physical fault source, independently verify a fault cause, or model concurrent/multi-label faults.

The module reuses the preexisting native scikit-learn estimator catalog and the R2c window features. It introduces no replacement machine-learning backend. The InceptionTime and ROCKET diagnosis models remain available in their existing catalog; their verified classifier adapters are separate work.

## Fixed class vocabulary

FaultLabelSchema declares ordered class labels and a named healthy/normal class. All classes must be represented in the training set, and any unknown sealed-test class causes an explicit error rather than being silently treated as a known fault. A single class label applies to each window. The meaning and ontology of fault types are the responsibility of the supplied data manifest.

## Fitting and prediction

SklearnWindowFaultClassifier defaults to native RandomForestClassifier, and also supports native SVC with probability=True, or an injected estimator exposing fit and predict_proba.

Training first extracts channel-major statistical features and passes only **training labels** to native fit. Predictions never receive ground truth labels. Native class-index ordering is remapped to the fixed schema, then validated for finite normalized [0,1] class probabilities before returning DiagnosticResult in (samples, classes) form. A separate predict method emits categorical fault labels.

Fitted inputs and inference data must match in channel names/order, units, sampling frequency and window length. This is closed-world categorical classification, not calibrated confidence that a physical subsystem caused the observed event. SVC probability estimates in particular are not a substitute for external calibration evidence.

## Split and sealed evaluation

FaultClassificationPartition attaches each sample ID and original recording ID to a window, alongside its label. FaultClassificationSplit requires recording- and sample-disjoint training, optional validation and sealed test sets. The same original run must not be renamed as a new group when creating a split.

FaultClassificationWorkflow performs exactly one fit on training windows and one predict_proba call on sealed test windows, then constructs a confusion matrix, support, accuracy and macro-F1. Per-window true labels are used only **after** the sealed predictions for metric construction. The workflow instance cannot be reused. The optional validation partition is reserved for future independent model selection and does not implicitly affect the final workflow.

Evaluation confusion rows are true classes and columns are predicted classes. Report output includes probability vectors, predictions, original sample/recording IDs and aggregate counts, but does not export individual sealed ground-truth labels. Per-class support must be inspected; a single averaged score can hide poorly represented fault classes.

## Current limits

- Independent recording IDs prevent obvious train/test window overlap, but do not cryptographically establish that physical data were not renamed or duplicated.
- This is window-level single-label classification. It does not support unseen classes, multi-fault mixtures, open-set rejection, temporal event segmentation or causal fault isolation.
- Class probabilities are classifier predictions, not validated engineering severity or fault-source likelihoods.
- Cross-device, cross-operating-condition and external dataset generalization remain to be demonstrated through separate controlled experiments.

## Acceptance

Tests verify fixed label order, native RF/SVC compatibility, class-probability validity, metadata guards, no inference refit, run-group split leakage prevention, one-shot sealed scoring, confusion matrix and macro-F1. Full repository Quality (>=83% branch coverage, audit and wheel checks) and CPU/Gloo DDP workflows must pass before merge.

Next: R4b fault-source isolation should require an explicit engineering mapping and distinguish model evidence from causal evidence. R4c adds independent development model selection and neural-model adapters.
