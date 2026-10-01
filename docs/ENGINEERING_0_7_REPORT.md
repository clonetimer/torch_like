# torch_like 0.7.0 Engineering Report

## 1. Objective

Version 0.7.0 continues the engineering convergence introduced in 0.4.0-0.6.0. The goal of this release is to remove duplicated array/DataLoader glue and duplicated metric definitions without pretending that forecasting, graph, generative, reinforcement-learning, and sklearn workloads share one universal `fit(X, y)` contract.

The resulting layering is:

```text
raw arrays / task data
        │
        ▼
   DataAdapter
        │
     DataLoader
        │
        ▼
   BatchAdapter
        │
   AdaptedBatch
   /    |     \
model  target  raw
batch           batch
   │      │
   ▼      ▼
 model  MetricEvaluator
   │      │
   └── Trainer ── checkpoint/events
```

## 2. Main changes

### 2.1 DataAdapter layer

Added `torch_like.data.DataAdapter`, `DataLoaderConfig`, and `TensorDataAdapter`.

`TensorDataAdapter`:

- accepts NumPy arrays or tensors;
- preserves input dtype instead of silently coercing labels/features;
- supports deterministic DataLoader shuffle through an explicit seed;
- supports index selection so callers can materialize train/validation/test loaders from one canonical dataset;
- is reused by the ordinary PyTorch AutoML backend instead of reimplementing `TensorDataset` / `DataLoader` construction there.

### 2.2 BatchAdapter and AdaptedBatch

Added `torch_like.training.AdaptedBatch` with explicit fields:

- `model_batch` — object consumed by the model contract;
- `targets` — supervised target used by a step/evaluator;
- `sample_count` — optional aggregation weight;
- `raw_batch` — original loader output for specialized task logic.

Added `BatchAdapter`, `IdentityBatchAdapter`, `CallableBatchAdapter`, `DefaultBatchAdapter`, and `ForecastingBatchAdapter`.

The default adapter is deliberately conservative. It automatically adapts common, unambiguous supervised `(X, y)` batches, while graph, recommendation, multimodal, generative, and RL workloads remain explicit when their semantics cannot be inferred safely.

`ForecastingBatchAdapter` is an important special case: it creates decoder input from known context plus zero future placeholders and stores the future target separately. This prevents target leakage while still exposing the correct supervised target to `step_fn` and evaluation.

### 2.3 Dataset-level evaluation

Added the lazy `torch_like.evaluation` namespace with:

- `Metric` protocol;
- `FunctionalMetric`;
- `MetricEvaluator`;
- `classification_metrics`;
- `regression_metrics`;
- `forecasting_metrics`;
- `multi_target_regression_metrics`;
- `metric_direction`.

This fixes a correctness issue in the legacy `metric_fns` path. In 0.6.x, a metric function produced one scalar per batch and Trainer averaged the batch values by sample count. That is not generally equivalent to computing a metric over the complete validation dataset for nonlinear statistics such as F1 and R².

The new `metrics=` API accumulates predictions/targets and computes each metric once at epoch end over the complete validation set. The legacy `metric_fns` API remains available for compatibility and keeps its documented batch-mean semantics.

### 2.4 Trainer integration

`torch_like.training.Trainer` now supports:

- model-aware automatic `BatchAdapter` resolution for unambiguous supervised models;
- existing `loss_fn(model, model_batch)` behavior without semantic changes;
- new `step_fn(model, adapted_batch)` for tasks that need targets/raw context while keeping model inputs separate;
- dataset-level `metrics=` plus an optional `prediction_fn`;
- legacy callable `batch_adapter` through `CallableBatchAdapter`.

No universal loss is guessed for point forecasting or other tasks that intentionally do not expose a standardized loss contract.

### 2.5 AutoML convergence

The ordinary PyTorch AutoML backend now reuses `TensorDataAdapter` for indexed DataLoader materialization.

Task metric formulas previously duplicated in `automl/sklearn_backend.py` now delegate to `torch_like.evaluation`, so direct evaluation and AutoML use one set of definitions.

The forecasting runner now also uses `ForecastingBatchAdapter` for its normal train/evaluate path and delegates RMSE/MAE calculation to `forecasting_metrics`. Existing private runner conversion helpers are retained only for compatibility with benchmark utilities.

`torch_like.training` and `torch_like.evaluation` no longer import sklearn metrics eagerly; sklearn is loaded only when one of the sklearn-backed convenience metric functions is invoked.

AutoML continues to own task-specific preprocessing, loss selection, candidate search, protocol auditing, sealed-test handling, and artifact orchestration. Those responsibilities are not pushed into the generic Trainer.

### 2.6 Trial-process hardening

AutoML trials already execute in a fresh Python interpreter. 0.7.0 further hardens this boundary:

- the child environment applies `ResourceSpec.max_cpu_workers` to common OpenMP/BLAS thread controls before NumPy/scikit-learn/LightGBM/PyTorch imports;
- child stderr is written to a temporary file rather than a `PIPE`, so pipe EOF inherited by short-lived descendants is not part of the parent/child completion contract;
- process-group timeout cleanup remains in place.

A regression test verifies the native-thread environment contract.

During qualification, long pytest commands executed directly through the container command transport could intermittently appear to stall around mixed sklearn/PyTorch tasks. The same continuous 38-test AutoML chain executed outside that transport completed 38/38 in 43.20 seconds. A complete single-process run with plugin autoload disabled also completed and reported 528 passed plus one expected stale-public-API-baseline failure. After the compatible baseline update, the API stability suite passed. This indicates that the observed transport behavior should not be represented as a deterministic torch_like trial failure.

## 3. Public API policy

0.7.0 adds `torch_like.evaluation` as a protected namespace while intentionally avoiding another large set of root-level convenience aliases.

Recommended access is:

```python
import torch_like as tl

loader = tl.data.TensorDataAdapter().loader(...)
adapter = tl.training.ForecastingBatchAdapter()
evaluator = tl.evaluation.MetricEvaluator(...)
```

Public API audit after convergence:

- protected namespaces: 9;
- protected symbols: 888;
- model catalog: 165 models;
- lazy export indexes: current.

The final API transition contains no removals and no replacements. `ForecastingBatchAdapter` gained optional `label_len` and `pred_len` constructor overrides; the transition checker classifies this as compatible. The committed baseline was rewritten only after validating that transition.

## 4. Compatibility

Existing behavior retained:

- native PyTorch modules remain native `nn.Module` objects;
- native sklearn estimators remain native sklearn objects;
- existing structured domain Batch objects remain valid;
- existing callable `batch_adapter` remains accepted;
- existing `loss_fn` behavior remains accepted;
- existing `metric_fns` remains accepted with legacy batch-mean semantics;
- 0.5 deployment artifacts continue to use `save_model` / `load_model`;
- 0.6 training checkpoints remain a separate lifecycle concept;
- 0.6 AutoML artifact/bundle behavior remains compatible.

## 5. Verification

### 5.1 Repository tests

Final pytest collection contains **531 tests** (521 inherited from 0.6.0 plus 10 new 0.7 regressions).

All **531 unique tests passed on the final code and final API/catalog baselines**, using deliberately separated pytest processes so long-lived NumPy/scikit-learn/PyTorch native runtime state cannot contaminate later AutoML trials:

- representative model/domain APIs: **266/266**;
- API stability, contracts, registry, catalog and core engineering: **86/86**;
- AutoML protocol/examples/backend-support/candidate/capability/evidence tests: **58/58**;
- runner, task-data, training runtime/adapters/events and run-event tests: **73/73**;
- `tests/test_automl.py`: **48/48** (`DataTests` 13, `BackendTests` 19, `TaskTests` 16; the TaskTests were split around the mixed-backend case).

A single uninterrupted 531/531 pytest parent process is intentionally **not** the release claim. During qualification, very long parent processes showed sensitivity to accumulated native thread/subprocess state, while the same unique tests consistently passed when isolated into bounded groups. This release therefore records the split-process qualification as the authoritative result rather than conflating command-transport timeout with a framework assertion failure.

The final API transition contains no breaking removals. The compatible optional `ForecastingBatchAdapter(label_len=..., pred_len=...)` parameters are included in the committed baseline, which protects **888 public symbols**.

### 5.2 Static/contract checks

Passed:

- `python -m compileall`;
- public API audit;
- model catalog audit;
- lazy export index audit.

`ruff` is not installed in the qualification environment. The report therefore does not claim a Ruff-clean repository. Historical style debt remains outside this release's behavioral scope.

### 5.3 Distribution verification

The final wheel is built with the environment's installed setuptools using `pip wheel --no-deps --no-build-isolation` because the qualification environment has no network access for build isolation.

Installation verification is performed from outside the source tree with `pip --target`. Smoke coverage includes:

- package version and import path;
- 165-model discovery;
- `TensorDataAdapter` and `Trainer`;
- dataset-level metrics;
- `ForecastingBatchAdapter` target separation and explicit `step_fn`;
- `save_model` / `load_model` prediction round-trip;
- native sklearn estimator creation/training/prediction.

## 6. Known limitations

0.7.0 intentionally does not claim support for:

- DDP/FSDP or distributed sampler/metric reduction;
- multi-optimizer or multi-phase objectives;
- generic RL environment rollout;
- universal graph/recommendation/multimodal/generative Batch inference;
- bounded-memory streaming implementations for every metric.

`FunctionalMetric` currently retains complete prediction and target arrays on CPU until epoch-end computation. This is correct for the target use cases in 0.7.0 but can become memory-heavy on very large validation sets. Streaming/reducible metric state is a natural next step.

## 7. Recommended next milestone

A reasonable 0.8.0 scope is **Streaming/Distributed Evaluation + Runner Convergence**:

1. define reducible metric state (`update/merge/compute`) for bounded-memory metrics;
2. add sampler/state contracts needed before DDP/FSDP can be claimed;
3. migrate dataset runners to public DataAdapter/BatchAdapter/Evaluator primitives where semantics match;
4. keep specialized multi-optimizer/RL/generative loops explicit until correct lifecycle contracts exist;
5. add a release-quality test launcher that isolates optional pytest plugins from framework qualification.

The project should not respond to remaining heterogeneity by adding more root aliases or by forcing every model through one `fit(X, y)` facade.
