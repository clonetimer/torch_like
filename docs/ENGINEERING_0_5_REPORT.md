# torch_like 0.5.0 Engineering Report

## Objective

0.4.0 solved package layout, lazy import, registry discovery, and native sklearn interoperability. The remaining engineering gap was runtime contract fragmentation: every domain had useful local batch/predict/loss helpers, but orchestration code could not inspect them through one stable API, persistence was not self-describing, and AutoML still maintained its own class-resolution table.

## Changes

1. Added `torch_like.contracts` with immutable `FieldSpec`, `InputSpec`, `OutputSpec`, `TaskSpec`, and `ModelSpec` descriptors plus backend/task/status enums.
2. Added `get_model_spec()` / `describe()` for names, classes, and instances across built-in torch_like models and the curated sklearn catalog.
3. Added `get_model_class()` so class resolution can stay lazy and avoid instantiating a model or importing a full domain catalog.
4. Added optional `ModelRuntime` / `as_runtime()` for capability-checked `make_batch`, `predict`, `loss`, generation/sampling/action dispatch, and native sklearn fit/transform paths.
5. Added `save_model()` / `load_model()` with a self-describing manifest, torch `state_dict` storage, sklearn joblib storage, explicit overwrite control, and constructor-config type encoding.
6. Connected AutoML `CandidateSpec` to the shared `ModelSpec` contract and changed the PyTorch AutoML builder to instantiate through the shared model registry rather than duplicated hard-coded class imports.
7. Added fail-closed handling for partial contracts. Example: ControlNet requires a model-specific `control` input and therefore does not advertise the generic `predict(batch)` operation.
8. Bumped package version to 0.5.0, protected the new contract namespace in the public-API baseline, and added contract API documentation.

## Validation performed

The user did not require a full repository test run for this iteration, so validation focused on the changed architecture and representative integration paths.

- Contract/runtime/serialization tests: **17 passed**.
- Engineering regression selection covering contracts, registry, lazy imports, engineering API, task interfaces, backend runners, scoped registration, capabilities, and API stability: **122 passed**.
- Representative model/domain API regression suite covering audio, detection, graph, forecasting, generative, multimodal, recommendation, reinforcement, tabular, vision, Mamba/sequence, and shared task interfaces: **266 passed**.
- Representative AutoML real execution paths, including torch backend, multilabel tabular, tabular leaderboard contract, and RUL leaderboard contract: **5 passed**.
- Contract resolution audit: **165/165** indexed built-in models resolved to `ModelSpec` without error; **148** are marked `standardized` and **17** are intentionally marked `partial` (primarily generative/RL semantics that cannot honestly share one generic contract).
- Public API audit: current at package version **0.5.0**, protecting **850** symbols.
- Model catalog audit: current with **165** indexed built-in models.
- Python bytecode compilation completed successfully for the package and new contract tests.
- Wheel build completed successfully with the environment's installed build tooling using `--no-build-isolation` because the container has no network access for downloading an isolated build environment.
- The built wheel was installed into a clean target directory and imported from outside the source tree. Version discovery, 165-model registry discovery, `describe()`, DLinear save/load prediction round-trip, and sklearn `RandomForestClassifier` runtime prediction all succeeded.

### Static-style tooling boundary

`ruff` is not installed in the current environment, so no claim of a successful Ruff run is made. The newly added contract inspection module was manually checked for >120-character lines after formatting and is clean, and `compileall` succeeds. The repository still contains pre-existing long lines in older modules; those are legacy style debt and are not represented as fixed by this release.

## Remaining boundaries

- `ModelRuntime` is an orchestration adapter, not a universal training engine. PyTorch optimizers/training loops and sklearn estimator fitting remain intentionally distinct.
- Vision outputs, conditional generative models, and RL action semantics remain model-specific where a single common semantic would be misleading.
- AutoML still needs task adapters for converting dataset metadata into task-specific constructor and batch arguments. The class-resolution duplication has been removed, but dataset/task adaptation cannot be eliminated safely.
- Persistence is designed for trusted local artifacts. sklearn/joblib artifacts rely on pickle semantics and must not be loaded from untrusted sources.
- A future release should address the remaining training/data lifecycle layer (trainer protocol, device/dtype movement, checkpoint/resume policy, metrics/events) rather than forcing additional semantics into `ModelSpec`.
