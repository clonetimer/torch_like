# R5b/R5c — Direct diagnostic composition and native RUL adapter

This is an **algorithm library**. No data-source audit, physical provenance,
train/validation/test partition mandate, workflow database or experiment platform.
Users supply all input signals, labels, model parameters, time bases and
engineering decisions. The library checks only numerical/interface constraints.

## 1. Explicit diagnostic composition

`DiagnosticPipeline` executes individually named, **already fitted** diagnostic
models. Use `DiagnosticStage(name, algorithm, method, source)` to declare each
operation. With `source="signals"` (default), the stage sees the original
`SignalBatch`. With `source="<preceding stage>"`, the stage sees that
upstream typed `DiagnosticResult`, for example the existing
`FaultSourceEvidenceMapper.predict`. Each step may invoke a native `score`,
`predict`, or `predict_proba` method and must return `DiagnosticResult`.
It never invokes `fit` or guesses whether a later step wants raw time-series,
class probabilities, an anomaly score or a health index.

```python
from torch_like.diagnostics import (
    DiagnosticPipeline, DiagnosticStage, GaussianHealthIndex,
    ZScoreDetector, FaultSourceEvidenceMapper,
    SklearnWindowFaultClassifier, FaultLabelSchema,
)

# reference, labeled_windows, live_windows are caller-owned SignalBatch values.
baseline = GaussianHealthIndex().fit(reference)
anomaly = ZScoreDetector().fit(reference)
labels = FaultLabelSchema(("normal", "bearing_fault"))
classification = SklearnWindowFaultClassifier().fit(labeled_windows, labels_train, labels)
# user_rules: a caller-created FaultSourceRulebook
candidate_sources = FaultSourceEvidenceMapper(user_rules)

pipeline = DiagnosticPipeline(
    DiagnosticStage("anomaly", anomaly, method="score"),
    DiagnosticStage("health", baseline, method="score"),
    DiagnosticStage("fault", classification, method="predict_proba"),
    DiagnosticStage("source_support", candidate_sources, method="predict", source="fault"),
)
results = pipeline.predict(live_windows)
print(results["health"].values)          # HI (higher means reference-like)
print(results["fault"].label_names)      # declared class order
print(results["source_support"].values) # expert-rule support, NOT causal posterior
```

Results preserve each algorithm's **unmodified semantics**, shape, threshold
and optional class names. A `DiagnosticBundleResult` exposes read-only mapping
and JSON-friendly `to_dict`. **No silent mean/max fusion of heterogeneous
scores, no implied diagnostic source probability, and no fixed alarm-debounce
policy**. If an application needs a fusion/decision layer, it must define an
explicit, task-specific algorithm and its calibration.

## 2. Uniform RUL calls

`RULRegressor(native_torch_model, unit="cycles"|"hours"|"seconds")` wraps
an original `torch.nn.Module`, preserving its class, weights and optimizer
interfaces. `create_rul_regressor` constructs installed named native models:

```python
from torch_like.diagnostics import create_rul_regressor

rul = create_rul_regressor(
    "TCNRUL", unit="cycles", negative_policy="clip",
    enc_in=2, seq_len=128, d_model=16, e_layers=2,
)
rul.fit(train_windows, caller_rul_targets, epochs=5, batch_size=16)
forecast = rul.predict_with_unit(live_windows)
print(forecast.values, forecast.unit)  # nonnegative (N,), explicit unit
print(rul.model)                       # original native TCNRUL
```

- `fit`: caller-supplied `SignalBatch` (N,L,C) and `targets` (N,),
  plus batch size, epochs, learning rate, Adam MSE/MAE/Huber loss and optional
  time marker `x_mark` with shape (N,L,M). **No data splitting, test scoring,
  selected failure threshold or automatic labels.**
- `predict`: returns `DiagnosticResult(task="rul_regression",
  output_kind="rul_value")` for use in a `DiagnosticPipeline`.
  `predict_with_unit` returns `RULForecast(result, unit)`, because time
  and cycle units must never be silently conflated.
- `negative_policy="reject"` (default) errors on native network negative
  outputs; `negative_policy="clip"` explicitly truncates negative estimates
  to zero. This is **output handling**, not a trained nonnegative RUL head or
  physical validity guarantee. Large unrealistically high values are never
  silently clipped.
- `ready=False` (default) prevents prediction from randomly initialized
  models; `fit` or explicit `mark_ready()` after external model restoration
  enables inference. `mark_ready` does *not* check the quality of weights.
- Matching fitted channel names, physical units, sampling rate and model
  temporal/channel dimensions is required; there is no authenticity audit.

The full `torch_like` distribution has **seven** registered RUL model types:
`THGNN`, `InformerRUL`, `AutoformerRUL`, `LogTransRUL`, `TCNRUL`,
`PatchTSTRUL`, and `TransformerRUL`. Use `list_rul_models()` to inspect
the models physically available. The algorithm-only stable distribution
deliberately includes the **three self-contained** backbones
(`TCNRUL`, `PatchTSTRUL`, `TransformerRUL`), not the other four's
forecasting/graph dependencies. An unknown or uninstalled model never falls
back to an unrelated surrogate.

This is a standardized **interface**, not an assertion that all models are
scientifically interchangeable or numerically equivalent. In particular,
some forecasting-derived RUL models require application-specific time markers
and valid constructor arguments; callers remain responsible for selecting
the model and supplying appropriate time/cycle RUL labels.

## 3. Stable distribution compatibility

New interfaces live in `torch_like.diagnostics` in the full framework and
`torch_like_stable.diagnostics` in the standalone slim wheel.
`torch-like-algorithms` advances to **v0.2.0** to preserve the immutability
of published v0.1.0 source and wheel contents. Original RULModel, sklearn,
InceptionTime and ROCKET interfaces are not changed.

The package build allowlist excludes AutoML, experiment workflows, datasets
and unrelated model domains. Quality, pinned Ruff, public API, installed-wheel
smoke, and DDP CI remain release gates.
