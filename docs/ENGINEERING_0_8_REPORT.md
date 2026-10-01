# torch_like 0.8.0 Engineering Report

## Scope

0.8.0 continues the engineering convergence of 0.4-0.7. It does not add a fake distributed-training switch. The release focuses on three boundaries that must be correct before DDP/FSDP can be considered: memory-bounded metric reduction, rank/shard metadata, and runner reuse of framework evaluation primitives.

## Streaming and mergeable metrics

`torch_like.evaluation` now includes:

- `ClassificationMetric` for accuracy, macro precision/recall/F1, and multilabel micro F1;
- `RegressionMetric` for MAE/MSE/RMSE/uniform-average R²;
- `ForecastingMetric` for MAE/RMSE/SMAPE and per-horizon MAE/RMSE;
- `MeanMetric` and `StandardDeviationMetric` for scalar streams;
- `MergeableMetric` state contract;
- `MetricEvaluator.state_dict()`, `load_state_dict()`, `merge_state()`, `merge()`, and `fully_streaming`.

The builtin metrics retain sufficient statistics only. Arbitrary `(y_true, y_pred)` functions continue to use `FunctionalMetric`, which buffers complete CPU arrays so exact semantics are preserved rather than approximated by batch averaging.

Metric states are merged before `compute()`. The implementation never averages already-computed shard F1/R² values.

## Runner convergence

- forecasting runner evaluation now uses `ForecastingMetric` and no longer concatenates all predictions/targets;
- RUL runner evaluation now uses `RegressionMetric` and no longer concatenates all predictions/targets;
- detection comparison uses streaming mean/std in its normal reporting path and only retains per-sample scores when explicitly requested.

Dataset-specific runners remain outside the protected core framework API.

## Distributed-ready contracts, not distributed execution

`torch_like.training` adds:

- `DistributedContext(rank, world_size, local_rank)`;
- `DistributedSamplerConfig`;
- `create_distributed_sampler()`, delegating sharding to PyTorch `DistributedSampler`;
- `TensorDataAdapter.loader_from_dataset(..., sampler=...)` for explicit sampler integration.

`Trainer` accepts a context internally but rejects `world_size > 1` with `NotImplementedError`. Process-group initialization, DDP/FSDP wrapping, gradient synchronization, rank-local checkpoint coordination, metric communication, and primary-rank artifact policy remain future work.

## Checkpoint schema v2

Training checkpoints now record:

- `distributed_context` with global rank, local rank, and world size;
- `rng_state_scope = "rank_local"`.

The loader continues to accept schema-v1 checkpoints from 0.6/0.7 as single-process state. Schema-v2 rank/world-size mismatch fails explicitly.

## Compatibility

- 0.7 callable metrics remain accepted through `FunctionalMetric`;
- 0.7 custom metrics implementing only `reset/update/compute` remain valid for ordinary evaluation;
- merge/state operations require the additive `MergeableMetric` contract;
- top-level `torch_like` was intentionally not expanded with distributed aliases; new distributed contracts live under `torch_like.training`;
- sklearn is still not imported by `import torch_like`, `import torch_like.training`, or `import torch_like.evaluation`.

## Validation

The final validation is performed in isolated pytest groups because a single very long-lived pytest parent can accumulate native sklearn/PyTorch thread-pool state and make the mixed AutoML test stall. Each unique test is nevertheless executed in the final source state.

- model/domain API: 266/266;
- core/API/contracts/governance: 98/98;
- AutoML support/protocol/candidates/events: 57/57;
- runner/task-data/training: 71/71;
- `tests/test_automl.py`: 48/48, with the mixed-backend case isolated;
- total unique tests: 540/540.

Governance/build validation:

- protected public API symbols: 901;
- model catalog: 165 models;
- public API baseline current;
- model catalog audit current;
- `compileall` passes;
- wheel build and source-tree-external installation smoke are required before release packaging.

## Remaining limitations

0.8.0 does not implement DDP/FSDP, multi-optimizer/alternating objectives, RL rollout orchestration, or general streaming implementations for every arbitrary user metric. `FunctionalMetric` intentionally remains O(validation-set-size) memory. Long-lived mixed AutoML pytest processes can still be sensitive to native runtime state; production trials continue to execute in isolated Python interpreters.
