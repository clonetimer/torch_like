# torch_like 0.6.0 Engineering Report

## Scope

Version 0.6.0 continues the engineering convergence started in 0.4.0 and 0.5.0. The main goal is to remove the remaining duplicated single-optimizer PyTorch training lifecycle from higher layers and expose one reusable, capability-aware training runtime without changing native `nn.Module` or sklearn semantics.

This release does **not** claim a production distributed trainer. It deliberately targets single-process/single-device training and keeps specialized multi-optimizer or multi-phase algorithms on explicit domain-specific paths.

## Main changes

1. Added the public `torch_like.training` package.
   - `Trainer`
   - `TrainerConfig`
   - `EpochResult`
   - `TrainingResult`
   - `train_model`
   - `save_training_checkpoint`
   - `load_training_checkpoint`
   - recursive `move_to_device` / `infer_batch_size`
   - shared `training_events`

2. Added a reusable single-optimizer lifecycle.
   - CPU/CUDA/`auto` device resolution
   - Adam, AdamW, SGD
   - cosine and plateau schedulers
   - optional CUDA AMP
   - gradient clipping and non-finite loss/gradient rejection
   - validation loss and optional metric aggregation
   - early stopping / `min_delta`
   - restore-best semantics

3. Added atomic training checkpoints and epoch-boundary resume.
   - model state
   - optimizer state
   - scheduler state
   - AMP scaler state
   - completed epoch/history
   - early-stopping state
   - CPU/CUDA torch RNG state
   - DataLoader generator state when exposed
   - trainer configuration
   - package version and model class provenance

4. Preserved the distinction between deployment artifacts and lifecycle checkpoints.
   - `save_model` / `load_model`: self-describing model artifact for inference/deployment
   - `save_training_checkpoint` / `load_training_checkpoint`: optimizer/lifecycle state for an already-constructed model

5. Refactored the ordinary PyTorch AutoML backend to reuse `torch_like.training.Trainer`.
   - AutoML still owns task-specific preprocessing, target scaling and objective selection.
   - Epoch iteration, optimizer/scheduler stepping, gradient safety, early stopping, checkpointing and lifecycle events now come from the shared Trainer.
   - Existing artifact filenames `last_checkpoint.pt` and `best_checkpoint.pt` are preserved.

6. Preserved historical AutoML bundle compatibility.
   - `load_best()` accepts the new 0.6 training-checkpoint schema.
   - the historical pre-0.6 checkpoint shape remains supported.

7. Moved the canonical training event bus to `torch_like.training.events`.
   - `torch_like.automl.events.training_events` reuses the same singleton.
   - direct training and AutoML therefore share event semantics.

8. Added the training namespace to the API-stability policy.
   - protected public symbols increased from 850 to 868.
   - deterministic API baseline and model-catalog audit were regenerated.
   - model catalog remains unchanged at 165 built-in models.

9. Added `docs/TRAINING_RUNTIME.md` and updated architecture, user guide, roadmap, packaging examples and README boundaries.

## Design decisions

### No fake universal `fit(X, y)`

The public trainer does not turn neural models into sklearn estimators. Standardized task models use their existing batch contract and `loss_step`; raw DataLoader output can be converted through `batch_adapter`.

Tasks that intentionally have no universal training objective, especially point forecasting, must provide an explicit `loss_fn(model, batch)`. This avoids silently selecting MSE where MAE, Huber, likelihood, quantile or domain-specific objectives may be required.

### No false DDP claim

0.6.0 is single-process/single-device. Distributed training is not implemented behind a misleading flag. A future DDP/FSDP layer should be added only after the checkpoint, sampler, metric-reduction and process lifecycle contracts are explicitly defined.

### Specialized objectives remain specialized

Some anomaly, generative and reinforcement-learning algorithms require multiple optimizers, alternating phases or algorithm-specific rollout state. Those paths are not forced through the generic Trainer until a correct multi-optimizer contract exists.

## Verification

The release was validated by grouped regression rather than one monolithic `pytest` invocation.

### Engineering/API groups

- `tests/test_training_runtime.py`
- `tests/test_contracts_runtime.py`
- `tests/test_engineering_api.py`
- `tests/test_lazy_imports.py`
- `tests/test_registry.py`

Result: **47 passed**.

### Stability/events/backend groups

- `tests/test_api_stability.py`
- `tests/test_model_catalog_audit.py`
- `tests/test_training_events.py`
- `tests/test_run_events.py`
- `tests/test_backend_runners.py`

Result: **37 passed**.

### Representative model-domain regression

Twelve model/API groups covering forecasting, detection, graph/task interfaces, vision, multimodal, generative, tabular, recommendation, audio, reinforcement and sequence/Mamba2 behavior.

Result: **266 passed**.

### Real AutoML torch paths

Five selected real execution paths covering in-process torch backend, RUL constraint behavior, multilabel tabular, full tabular leaderboard + `load_best()`, and RUL leaderboard contract.

Result: **5 passed**.

These selected groups are disjoint, for **355 selected regression tests passed**. This is not presented as a full-suite run.

### Deterministic governance checks

- lazy export indexes: current
- public API baseline: **868 protected symbols**, current
- model catalog audit: **165 models**, current
- Python `compileall`: passed

### Wheel / installed-state smoke test

The environment has no network access and did not have the `build` module installed. The first isolated build attempt therefore failed while trying to download build dependencies. The wheel was then built correctly with the already-installed compatible `setuptools 82.0.1` using:

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

The resulting `torch_like-0.6.0-py3-none-any.whl` was installed with `pip --target` into a clean directory and imported from outside the source tree.

Installed-state checks:

- imported package path points to the isolated target directory;
- package version: `0.6.0`;
- built-in model discovery: `165`;
- public `Trainer` available;
- one real FTTransformer training epoch completed;
- `last_checkpoint.pt` loaded through `load_training_checkpoint`;
- checkpoint reports `kind=training_checkpoint` and `library_version=0.6.0`;
- `save_model` / `load_model` roundtrip produced identical predictions.

## Known limitations / remaining work

1. Distributed DDP/FSDP lifecycle is not implemented.
2. Generic multi-optimizer / alternating-phase training is not standardized yet.
3. Dataset-specific `runner/` scripts still contain local experiment loops and remain outside the core public API.
4. Detection-specific AutoML algorithms with specialized training phases still use dedicated backend routines.
5. `ruff` is not installed in the current execution environment, so no claim of repository-wide Ruff cleanliness is made. New training files were checked for >120-character lines and all modified Python files passed `compileall`.
6. Resume is defined at epoch boundaries. Exact mid-batch recovery is not provided.
7. Training checkpoints are trusted local artifacts for lifecycle recovery; they are not a substitute for the self-describing `save_model` deployment format.

## Recommended next milestone

The next engineering step should be **0.7.0 Data/Trainer Adapter Convergence**, not another broad directory reshuffle. The main target is to standardize how task datasets become domain batch objects and how metrics are reduced, then decide whether a correct multi-optimizer contract can cover anomaly/generative/RL training without weakening task semantics.

A sensible sequence is:

1. `DataAdapter` / `BatchAdapter` protocol with static shape/dtype validation.
2. metric registry and train/validation/test evaluator separation.
3. explicit checkpoint compatibility policy and migration tests.
4. multi-optimizer training plan protocol only for domains that can share it correctly.
5. DDP/FSDP design after sampler, event, metric and checkpoint semantics are proven in single-process mode.
