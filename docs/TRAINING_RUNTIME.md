# 0.9.0 更新

本文原有单进程 API 保持兼容。新增 DDP 的使用方式、数据合同和未支持边界见 [DDP_RUNTIME.md](DDP_RUNTIME.md)。

# Training Runtime / 训练运行时

`torch_like 0.8.0` provides a small public training runtime for native PyTorch models. It centralizes optimization, validation, early stopping, gradient clipping, checkpoint/resume, device placement, AMP, and lifecycle events without replacing domain task APIs.

`torch_like 0.8.0` 延续并收敛轻量公共训练运行时。它统一优化器、验证、早停、梯度裁剪、checkpoint/resume、设备迁移、AMP 和生命周期事件，但不替换各领域原有的任务 Batch 和 `loss_step` 语义。

## Boundary / 边界

The runtime is intentionally **not** a universal `fit(X, y)` facade:

- torch_like neural models remain ordinary `torch.nn.Module` objects;
- sklearn estimators continue to use native sklearn `fit`/`predict`;
- domain batch objects such as `TabularBatch`, `SequenceBatch`, and `RULBatch` remain the training contract;
- tasks without a standardized loss contract, including point forecasting, must supply an explicit `loss_fn` or `step_fn`;
- `BatchAdapter` keeps model input and evaluation target separate rather than flattening domain batches;
- 0.8.0 `Trainer` remains single-process/single-device. `DistributedContext` and sampler/metric/checkpoint contracts prepare rank semantics, but DDP/FSDP execution is not silently claimed as supported.

## Standard task training / 标准任务训练

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch_like as tl

model = tl.FTTransformer(
    input_dim=8,
    output_dim=3,
    d_model=32,
    n_heads=4,
    n_layers=2,
    d_ff=64,
)

x = torch.randn(128, 8)
y = torch.randint(0, 3, (128,))
loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

trainer = tl.Trainer(
    model,
    tl.TrainerConfig(
        epochs=10,
        device="auto",
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=1e-2,
        grad_clip_norm=1.0,
        checkpoint_dir="runs/example",
    ),
)
result = trainer.fit(loader)
```

When the model contract advertises `loss`, `Trainer` delegates to `as_runtime(model).loss(batch)`. For unambiguous supervised tuple batches, 0.8.0 automatically creates the corresponding domain batch; callers can still pass an explicit `BatchAdapter` when input semantics are richer.

## Explicit training step / 显式训练步骤

Deterministic forecasting models expose prediction semantics but do not claim one universal training loss. With 0.8.0, forecasting task-data dictionaries can use `ForecastingBatchAdapter`; `step_fn` receives an `AdaptedBatch`, so the domain model input and future target remain separate:

```python
import torch.nn.functional as F

model = tl.DLinear(
    enc_in=4,
    dec_in=4,
    c_out=4,
    seq_len=96,
    label_len=48,
    pred_len=24,
)

trainer = tl.Trainer(
    model,
    tl.TrainerConfig(epochs=20, device="auto"),
    batch_adapter=tl.training.ForecastingBatchAdapter(),
    step_fn=lambda current, batch: F.mse_loss(
        current.predict(batch.model_batch),
        batch.targets,
    ),
)
```

`loss_fn(model, batch)` remains supported with its 0.6 semantics for existing integrations. `step_fn(model, adapted_batch)` is the preferred form when the task needs targets or other raw-batch context that must not be inserted into the model input. MSE, MAE, Huber, likelihood objectives, quantile loss, and domain-specific constraints are deliberately not guessed by the framework.

## Validation metrics / 验证指标

Validation loss is aggregated by sample count. Dataset-level metrics introduced in 0.7 remain available, while `0.8.0` adds streaming/mergeable builtin metrics that are computed once over the complete validation set instead of averaging per-batch nonlinear metrics:

```python
trainer = tl.Trainer(
    model,
    config,
    metrics={
        "accuracy": lambda y_true, y_pred: (y_true == y_pred).mean(),
    },
    prediction_fn=lambda current, batch: current.predict(batch).argmax(-1),
)
result = trainer.fit(train_loader, val_loader)
print(result.history[-1].validation_metrics)
```

The older `metric_fns={name: lambda model, batch: scalar}` API remains available for compatibility and is still interpreted as a batch-mean metric. Prefer `metrics=` for F1, R², event metrics, and other statistics that must be computed over the whole dataset. See [`DATA_EVALUATION_RUNTIME.md`](DATA_EVALUATION_RUNTIME.md).

## Checkpoint and resume / 断点恢复

With `checkpoint_dir` set, the runtime writes:

```text
checkpoint_dir/
├── last_checkpoint.pt
└── best_checkpoint.pt
```

A training checkpoint contains:

- model state;
- optimizer state;
- scheduler state;
- AMP scaler state;
- completed epoch and history;
- early-stopping state;
- torch CPU/CUDA RNG state;
- DataLoader generator state when the loader exposes one;
- the Trainer configuration used to create the checkpoint.
- checkpoint schema v2 process metadata (`rank`, `local_rank`, `world_size`);
- explicit `rng_state_scope="rank_local"` metadata.

Schema v1 checkpoints written by 0.6/0.7 remain loadable and are interpreted as single-process checkpoints. Loading a schema-v2 checkpoint under a different rank/world-size context fails explicitly rather than restoring the wrong rank-local RNG state.

Resume uses the configured epoch count as the **total target**:

```python
trainer = tl.Trainer(model, tl.TrainerConfig(epochs=20, checkpoint_dir="runs/example"), ...)
result = trainer.fit(
    train_loader,
    val_loader,
    resume_from="runs/example/last_checkpoint.pt",
)
```

If the checkpoint completed epoch 8, the call continues from epoch 9 through epoch 20. The runtime restores the loader generator state when available, so epoch-boundary resume preserves shuffled sample order.

`tl.load_training_checkpoint(...)` requires an already-constructed model because a training checkpoint is lifecycle state, not a self-describing deployment artifact. Use `tl.save_model(...)` / `tl.load_model(...)` for portable model artifacts.

## Early stopping and restore-best / 早停与最佳模型

```python
config = tl.TrainerConfig(
    epochs=100,
    early_stopping_patience=8,
    min_delta=1e-4,
    restore_best=True,
    checkpoint_dir="runs/example",
)
```

Validation loss is monitored when a validation loader is provided; otherwise training loss is monitored. `restore_best=True` restores only model weights at the end of fitting. It does not rewind the optimizer after training has finished.

## AMP and devices / AMP 与设备

`device="auto"` selects CUDA when available and CPU otherwise. `amp=True` enables CUDA autocast and gradient scaling only on CUDA; on CPU it remains disabled rather than changing CPU numerical semantics.

## Lifecycle events / 生命周期事件

The shared event bus is available as:

```python
from torch_like.training import training_events

dispose = training_events.on(lambda name, **payload: print(name, payload))
try:
    trainer.fit(train_loader, val_loader)
finally:
    dispose()
```

Events currently include `training_start`, `epoch_end`, and `training_end`. AutoML uses the same event bus and the same `Trainer`, so direct training and AutoML do not maintain separate lifecycle implementations.

## AutoML integration / AutoML 集成

From 0.6.0, the normal torch AutoML backend owns only task-specific preparation and objective adaptation. Epoch iteration, optimizer/scheduler stepping, gradient safety, early stopping, checkpointing, and events are delegated to `torch_like.training.Trainer`.

Detection-specific backends still have specialized optimization routines where their objectives require multiple optimizers or model-specific phases. Those are intentionally not forced through the generic Trainer until a correct multi-optimizer contract exists.

## 0.7 data/evaluation convergence / 0.7 数据与评估收敛

`TensorDataAdapter`, `BatchAdapter`, `AdaptedBatch`, and `MetricEvaluator` are documented in [`DATA_EVALUATION_RUNTIME.md`](DATA_EVALUATION_RUNTIME.md). AutoML uses the shared tensor loader adapter and shared metric formulas; direct training and AutoML therefore no longer maintain separate array-to-loader and task-metric implementations.

## Distributed-ready contracts / 分布式准备合同

`torch_like.training.DistributedContext` reads or carries standard `rank / world_size / local_rank` identity without initializing a process group. `DistributedSamplerConfig` plus `create_distributed_sampler()` delegate data partitioning to PyTorch's standard `DistributedSampler` rather than reimplementing distributed sampling.

```python
adapter = tl.data.TensorDataAdapter()
dataset = adapter.dataset(x, y)
context = tl.training.DistributedContext(rank=0, world_size=2, local_rank=0)
sampler = tl.training.create_distributed_sampler(
    dataset,
    context,
    tl.training.DistributedSamplerConfig(shuffle=True, seed=42),
)
loader = adapter.loader_from_dataset(
    dataset,
    config=tl.data.DataLoaderConfig(batch_size=32, shuffle=False),
    sampler=sampler,
)
```

A future distributed trainer must additionally own process-group initialization, model wrapping, gradient synchronization, sampler `set_epoch`, rank-local RNG restore, metric-state communication, and primary-rank artifact writes. 0.8.0 exposes the contracts needed for that work but intentionally does not claim the lifecycle is complete.
