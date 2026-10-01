# API Contracts and Runtime Adapter / API 合同与运行时适配层

`torch_like 0.5+` adds a machine-readable contract layer above the native PyTorch and scikit-learn object models. The goal is orchestration consistency, not forcing every domain into one artificial `fit(X, y)` API.

## Design boundary / 设计边界

- Native torch_like models remain `torch.nn.Module` instances.
- Curated sklearn algorithms remain the original sklearn estimator classes and instances.
- Existing domain APIs such as `make_forecast_batch`, `loss_step`, `generate`, and `sample` are preserved.
- The contract layer describes task semantics, input fields, output semantics, backend, domain, and supported operations.
- `ModelRuntime` is optional. It delegates to the native object and checks capabilities before dispatch.
- Heterogeneous domains such as generative modeling and reinforcement learning can be marked `partial`; the library does not claim a universal operation where one cannot be made correct.

## Describe an algorithm / 查看算法合同

```python
import torch_like as tl

spec = tl.describe("PatchTST")
print(spec.domain)                 # forecasting
print(spec.task.kind)              # time_series_forecasting
print(spec.operations)             # ('predict',)
print(spec.task.input.batch_type)   # torch_like.models.forecasting.ForecastingBatch
print(spec.to_dict())              # plain JSON-serializable metadata
```

The same entrypoint covers the curated sklearn compatibility catalog:

```python
spec = tl.get_model_spec("RandomForestClassifier")
assert spec.backend == tl.BackendKind.SKLEARN
assert spec.task.kind == tl.TaskKind.CLASSIFICATION
assert spec.operations == ("fit", "predict")
```

`get_model_spec(name)` imports only the requested torch_like implementation. It does not bootstrap an entire domain catalog.

## Runtime adapter / 统一运行时适配

```python
import torch
import torch_like as tl

model = tl.InceptionTime(
    input_dim=3,
    num_classes=5,
    seq_len=32,
    d_model=16,
    num_blocks=3,
    bottleneck_dim=4,
    kernel_sizes=(3, 5, 7),
)

runtime = tl.as_runtime(model)
batch = runtime.make_batch(
    x=torch.randn(4, 32, 3),
    labels=torch.tensor([0, 1, 2, 3]),
)
logits = runtime.predict(batch)
loss = runtime.loss(batch)
```

For causal sequence models, `runtime.predict(batch)` maps the common operation to `next_token_logits(batch)`. For sklearn, `runtime.fit(...)`, `runtime.predict(...)`, `runtime.transform(...)`, and `runtime.fit_predict(...)` preserve the estimator's native semantics.

Unsupported operations fail before dispatch:

```python
runtime = tl.as_runtime(tl.create("DLinear", ...))
assert not runtime.supports("fit")
# runtime.fit(...) -> UnsupportedOperationError
```

This is deliberate: PyTorch training is optimizer/loop driven, while sklearn training is estimator `fit` driven. A single fake implementation would hide materially different lifecycle semantics.

## Persistence / 保存与恢复

```python
model = tl.create("DLinear", ...)
tl.save_model(model, "artifacts/dlinear")
restored = tl.load_model("artifacts/dlinear")
```

A torch_like artifact contains:

```text
artifacts/dlinear/
├── model.json       # schema, contract, class path, constructor config
└── weights.pt       # state_dict only
```

Curated sklearn estimators use the same `model.json` manifest and store `estimator.joblib`.

`save_model` writes to a temporary sibling directory and commits it only after the artifact is complete. Existing targets are rejected unless `overwrite=True` is explicit.

The constructor-config encoder supports JSON primitives, tuples, `pathlib.Path`, NumPy arrays/scalars, and torch tensors/dtypes/devices. Unsupported constructor objects fail explicitly instead of silently producing an unrecoverable artifact.

> Security: sklearn/joblib artifacts use pickle semantics. Only load model directories from trusted sources.

## Contract status / 合同状态

`ContractStatus.STANDARDIZED` means the declared orchestration operations have a common dispatch path. `ContractStatus.PARTIAL` means the task is intentionally heterogeneous and callers must inspect `operations` and model-specific notes.

In 0.5.0, the built-in catalog resolves contracts for all 165 indexed models. Most models use standardized contracts; generative and reinforcement-learning families remain partial where conditional inputs/action semantics differ.

## AutoML integration / AutoML 集成

AutoML `CandidateSpec.model_spec()` now exposes the same shared contract when a candidate belongs to the torch_like or curated sklearn catalogs. The PyTorch AutoML backend constructs models through the shared registry after checking the candidate task against `TaskKind`; it no longer keeps a second hard-coded class import table.

Task adapters still own task-derived constructor arguments such as `num_classes`, `seq_len`, or `pred_len`. This separation is intentional:

```text
ModelSpec          -> what the model accepts/returns and what operations it supports
CandidateSpec      -> which AutoML tasks/search grids may select it
Task adapter       -> how dataset/task metadata becomes constructor/batch arguments
Backend runner     -> optimizer/training/evaluation lifecycle
```

That boundary lets new neural-network families grow without turning the central factory into a dataset-specific training engine.


## Training lifecycle / 训练生命周期

Starting in 0.6.0, PyTorch optimization lifecycle is provided by `torch_like.training.Trainer`. The contract/runtime layer still answers **what operations a model supports**; the Trainer owns **how repeated optimization is executed**. AutoML task adapters can supply an explicit loss function for task contracts that intentionally do not standardize one. See [`TRAINING_RUNTIME.md`](TRAINING_RUNTIME.md).

This keeps the boundaries separate:

```text
ModelSpec / ModelRuntime  -> task semantics and capability dispatch
Trainer                   -> optimization, validation, early stop, checkpoint/resume
AutoML task adapter       -> dataset-specific preprocessing and objective selection
```
