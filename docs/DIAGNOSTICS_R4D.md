# R4d — Native PyTorch fault classifiers under sealed evaluation

R4d connects existing **native** `torch_like.models.diagnosis.InceptionTime` and
`ROCKET` to the R4a classification and R4c device/condition holdout workflows.
It does not replace `torch.nn.Module`, change native model classes, or claim
paper reproduction and real-device generalization.

## Usage

```python
from torch_like.diagnostics import (
    FaultClassificationWorkflow,
    TorchWindowFaultClassifier,
)

# split is an already validated FaultClassificationSplit.
classifier = TorchWindowFaultClassifier(
    "InceptionTime",
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    random_state=0,
)
report = FaultClassificationWorkflow(classifier).run(split)
assert classifier.model.__class__.__name__ == "InceptionTime"
```

For R4c, pass the same classifier to `FaultGeneralizationWorkflow` with a
`FaultGeneralizationSplit`; no adapter-specific test access is needed.
Select `model_name="ROCKET"` for the repository's native ROCKET-style random
convolutional feature model. `model_kwargs` is an optional dictionary of
native model constructor settings; alternatively provide a preconstructed
`torch.nn.Module` through `model` (never both).

## Contract and scientific boundaries

- **Development only:** `fit(reference, labels, schema)` consumes training
  windows and their labels, requires every predefined class, and runs minibatch
  Adam/cross-entropy for an explicit epoch count. This first adapter is
  CPU-only; no silent device transfer to GPU is performed.
- **Label-free inference:** `predict_proba(batch)` runs under
  `torch.inference_mode()` with `eval()`, softmaxes ordered logits and
  returns `DiagnosticResult(fault_classification, class_probability)`. The
  classifier instance is single-fit; a caller must create a new instance for
  another training run. `predict` returns categorical names.
- **Physical input contract:** fixed time-window length, channel identity and
  order, units and Hz sample rate; no automatic resampling, imputation, unit
  conversion or normalization. Nonfinite values fail closed.
- **Sealed evaluation:** the preexisting R4a/R4c workflows enforce original
  recording and sample ID disjointness and run sealed test prediction once.
  The adapter neither performs validation-based search nor reads test labels.
- **Evidence:** softmax values are model predictions, **not calibrated**
  probabilities of physical fault sources or verified root causes. No
  independent domain robustness experiment, multiseed comparison, or
  independent-recording uncertainty interval is claimed here.
- **ROCKET naming:** the native repository ROCKET model has learned linear
  classifier weights on fixed random kernels. It is not claimed to replicate
  every variant and training procedure in published ROCKET benchmarks.

## Acceptance

R4d requires negative tests for missing classes, invalid model outputs,
metadata mismatch, NaN/missing-data rejection, no-grad inference, immutability
of parameters under inference, preserved native model types, independent
domain holdout and single sealed evaluation. Merge only with the existing
Quality (including branch coverage, wheel and API governance) and CPU/Gloo DDP
checks passing on the **final** PR head.

Next engineering work should separately implement validation-only model
selection (multiple seeds and fair budgets), recording-cluster uncertainty and
real data generalization evidence. No sealed-test metric may select or retune
a model.
