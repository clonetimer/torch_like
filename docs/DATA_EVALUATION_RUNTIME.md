# Data and Evaluation Runtime / 数据与评估运行时

`torch_like 0.8.0` builds on the 0.7 shared adapters between raw datasets, domain batch contracts, training, evaluation, and AutoML. The goal is to remove repeated glue code without flattening heterogeneous model semantics.

`torch_like 0.8.0` 延续 0.7 的数据、domain batch、训练、评估和 AutoML 共享适配层，并增加可合并的流式指标状态。目标是消除重复胶水代码，而不是把不同领域强行压成同一种输入输出。

## Layers / 分层

```text
arrays / tensors / task datasets
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
model_batch targets raw_batch
       │       │
       ▼       ▼
    Model   Evaluator
       │       │
       └── Trainer ── checkpoint/events
```

### DataAdapter

`torch_like.data.TensorDataAdapter` builds deterministic `TensorDataset` / `DataLoader` objects from numpy arrays or tensors while preserving dtype and optional sample indices.

```python
import torch_like as tl

adapter = tl.data.TensorDataAdapter()
loader = adapter.loader(
    x,
    y,
    config=tl.data.DataLoaderConfig(
        batch_size=32,
        shuffle=True,
        seed=42,
    ),
)
```

The same adapter is used by the normal PyTorch AutoML backend, so split indexing and DataLoader construction are no longer reimplemented there.

The forecasting experiment runner also consumes the shared `ForecastingBatchAdapter` and `forecasting_metrics` primitives. Legacy private runner helpers remain for benchmark compatibility, but the normal training/evaluation path no longer maintains a second decoder-input or RMSE/MAE implementation.

### BatchAdapter

`torch_like.training.BatchAdapter` converts one loader output into a domain model batch. `AdaptedBatch` keeps four concerns explicit:

- `model_batch`: object consumed by `predict()` / `loss_step()`;
- `targets`: supervised target used by dataset-level evaluation or explicit training steps;
- `sample_count`: optional explicit aggregation weight;
- `raw_batch`: original loader output for task-specific logic.

Common supervised `(X, y)` loaders are adapted automatically when the model contract is unambiguous. Ambiguous domains such as graph, recommendation, multimodal, generative, and reinforcement learning continue to require explicit adapters unless their loader already returns the domain batch object.

### ForecastingBatchAdapter

Forecasting data deserves an explicit adapter because decoder input and future target are not the same thing. `ForecastingBatchAdapter` understands the dictionaries produced by `torch_like.data.task_data`, constructs a decoder input containing known context plus zero future placeholders, and keeps the future target separate.

```python
adapter = tl.training.ForecastingBatchAdapter()

trainer = tl.Trainer(
    model,
    tl.TrainerConfig(epochs=10),
    batch_adapter=adapter,
    step_fn=lambda current, batch: torch.nn.functional.mse_loss(
        current.predict(batch.model_batch),
        batch.targets,
    ),
)
```

This avoids future-target leakage into decoder inputs while still making targets available to the training/evaluation layer.

## Dataset-level metrics / 数据集级指标

`0.6.x` supported `metric_fns`, which return one scalar per batch and are averaged by sample count. That remains supported for compatibility, but it is not mathematically exact for nonlinear metrics such as F1 and R².

`0.7.0` introduced `MetricEvaluator` and the exact buffered `FunctionalMetric`. `0.8.0` keeps that fallback for arbitrary functions and adds mergeable streaming metrics for common cases: `ClassificationMetric`, `RegressionMetric`, `ForecastingMetric`, `MeanMetric`, and `StandardDeviationMetric`. Streaming metrics retain sufficient statistics rather than complete prediction/target arrays.

```python
trainer = tl.Trainer(
    model,
    config,
    metrics={
        "accuracy": lambda y_true, y_pred: (y_true == y_pred).mean(),
    },
    prediction_fn=lambda current, batch: current.predict(batch).argmax(-1),
)
```

For a metric function, argument order is `(y_true, y_pred)`.

For large validation sets, prefer a builtin streaming metric when its semantics match the task:

```python
evaluator = tl.evaluation.MetricEvaluator({
    "rmse": tl.evaluation.RegressionMetric("rmse"),
    "r2": tl.evaluation.RegressionMetric("r2"),
})

for prediction, target in validation_batches:
    evaluator.update(prediction, target)

print(evaluator.compute())
```

`MetricEvaluator.state_dict()` returns mergeable metric state. Another evaluator can combine it with `merge_state()` and compute once after all shards are merged. This is different from averaging already-computed rank metrics, which is generally wrong for F1/R². `FunctionalMetric` also exposes state for correctness, but its state remains buffered and therefore `fully_streaming` is `False`.

```python
left.merge_state(right.state_dict())
metrics = left.compute()
```

The forecasting and RUL experiment runners now use streaming metrics. The detection comparison path computes score mean/std without retaining every score unless `collect_scores=True` is explicitly requested.

Importing `torch_like.training` or `torch_like.evaluation` does not eagerly import scikit-learn. The sklearn metric functions are resolved only when the corresponding convenience metric is actually called, keeping PyTorch-only training startup isolated from sklearn.

Shared metric primitives include:

- `classification_metrics(...)`;
- `regression_metrics(...)`;
- `forecasting_metrics(...)`;
- `multi_target_regression_metrics(...)`;
- `metric_direction(...)`.

AutoML now delegates these formulas to `torch_like.evaluation`, so sklearn and PyTorch candidate paths consume the same metric definitions.

## Compatibility boundary / 兼容边界

- Existing callable `batch_adapter=lambda raw: ...` remains accepted through `CallableBatchAdapter`.
- Existing `metric_fns={name: lambda model, batch: scalar}` remains supported as a legacy batch-mean metric API.
- Existing `loss_fn(model, batch)` semantics remain unchanged.
- New `step_fn(model, adapted_batch)` is available when a task needs both a domain `model_batch` and separately retained targets.
- Native sklearn estimators still use native sklearn `fit/predict`; `TensorDataAdapter` and `Trainer` do not wrap sklearn into `nn.Module`.
- DDP/FSDP, multi-optimizer objectives, RL environment rollout, and highly model-specific generative training remain outside the generic Trainer contract.
