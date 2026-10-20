# R4e — Validation-only multi-seed diagnostic classifier selection

R4e is a **selection protocol**, not a claim of model quality or real-device
generalization. It adds `FaultCandidateSpec` and `FaultSelectionWorkflow`
around existing native sklearn and PyTorch diagnostic classifiers and the
R4a/R4c sealed evaluation workflows.

## End-to-end example

```python
from torch_like.diagnostics import (
    FaultCandidateSpec,
    FaultSelectionWorkflow,
)

# split: already validated FaultClassificationSplit or FaultGeneralizationSplit.
# Its validation partition is mandatory and physically disjoint from the test.
candidate_rf = FaultCandidateSpec(
    identifier="rf_100",
    backend="sklearn",
    algorithm="RandomForestClassifier",
    seeds=(2, 5, 11),
    model_parameters=(("n_estimators", 100),),
    budget_unit="trees",
    budget_value=100,
)
candidate_inception = FaultCandidateSpec(
    identifier="inception_cpu",
    backend="torch_like",
    algorithm="InceptionTime",
    seeds=(2, 5, 11),
    model_parameters=(("num_blocks", 3),),
    training_parameters=(("epochs", 12), ("batch_size", 32)),
    budget_unit="epochs",
    budget_value=12,
)
workflow = FaultSelectionWorkflow(
    (candidate_rf, candidate_inception),
    metric="macro_f1",
    min_class_windows=1,
    min_validation_recordings=1,
)
result = workflow.run(split)
audit = result.to_dict()
# audit["selection"]: validation-only scores, eligibility and winner
# audit["sealed_test"]: one fresh model's one-shot test evaluation
```

The supported closed-world candidates are native RandomForestClassifier/SVC
through `SklearnWindowFaultClassifier`, and native InceptionTime/ROCKET through
`TorchWindowFaultClassifier`. This layer does not provide automatic hyperparameter
search, estimator source mutation, calibration, class discovery, or GPU scheduling.
Custom builders can be injected with the `factory(spec, seed)` callback; such
builders must still implement the existing fit/predict_proba diagnostic contract.

## Execution and leakage boundaries

1. Freeze the candidate specifications, nominal budgets, immutable scalar/flat-tuple
   constructor arguments, unique candidate identifiers and **at least two distinct
   seeds** per candidate. `random_state` and model identity are controlled by the
   specification, not overwritten by raw keyword arguments.
2. Reuse the R4a recording/sample-disjoint split, and, for R4c inputs, the verified
   device/condition/pair domain holdout. Reject missing validation, insufficient
   declared class support, insufficient independent validation recordings, and
   insufficient R4c validation domain recordings **before fitting any candidate**.
   Defaults of one recording per domain/class are minimal validity checks and
   **not adequate statistical evidence** of generalization; raise these minima
   for real experiments.
3. Each candidate/seed gets a **fresh classifier**, fits exclusively on labeled
   training windows and predicts only validation windows. Only validation
   `macro_f1` or `accuracy` is used to rank configurations. No sealed test
   signals, predictions or labels enter the selection loop.
4. Report every seed's validation metrics or a typed failure. A candidate is
   eligible only if **all** its predeclared seeds succeed. Choose maximum mean
   validation metric, then lower seed-to-seed standard deviation, then lexical
   candidate identifier. Do **not** cherry-pick a seed based on validation:
   the first predeclared seed is used for the final fresh model.
5. Instantiate the selected model anew and call the original R4a or R4c
   workflow for one training fit and **one sealed test prediction**. The
   selector is single-use and no candidate can reopen the sealed test.
   Training remains train-only; validation observations are used solely for
   ranking and are not silently merged into the final training subset.
6. The report intentionally stores aggregate validation scores and support,
   *not individual validation probabilities, test ground-truth labels, or
   repeated test rankings*. The sealed report keeps the original R4a/R4c
   contract and associated limitations.

## Scientific and engineering caveats

- `trees`, `epochs` and `fits` are **nominal settings, not comparable compute
  budgets**. R4e does not claim runtime, energy, samples seen, memory use or
  model-family fairness. Cross-family rankings are exploratory until measured
  resources and budget policies are made commensurate.
- Multiple seeds reduce sensitivity to random initialization; windows sharing
  an original recording are correlated, so ordinary window-based scores are not
  independent physical repeats or uncertainty intervals.
- Selection on validation itself incurs selection bias; publish sealed-test
  metrics only after all candidates are frozen, and use an external physical
  holdout when claiming transfer/generalization.
- No open-world rejection, multivariate fault-source posterior, fault propagation
  or causal root-cause inference is implied by softmax class probabilities.
- Factory injection is caller-controlled code. The library guarantees that its
  *own* ranking loop never passes sealed test partitions into candidate methods;
  it cannot enforce that arbitrary user code has no external data access.

## Verification gates

Ruff pinned at 0.17.1; API governance; full pytest with >=83% branch coverage;
wheel metadata; CPU/Gloo DDP. The final PR head must pass all. Tests include
train/validation/test provenance leakage, R4c holdout leakage, insufficient
support, failure isolation, deterministic tie/seed rules, native sklearn and
PyTorch execution and the single sealed inference call.

Next: R4f original-recording-cluster uncertainty bounds and genuine externally
held-out physical device benchmarks.
