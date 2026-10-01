# torch_like Project Structure / 项目结构

中文：本文说明 `torch_like` 的目录分层和边界。目标是防止模型框架、实验脚本、数据处理和文档混在一起。

English: This document explains the directory layering and boundaries of `torch_like`. The goal is to keep the model framework, experiment scripts, data processing, and documentation separate.

## 目标 / Goal

中文：`torch_like` 的核心目标是成为本地 PyTorch 风格模型库，而不是训练工程集合。

English: The core goal of `torch_like` is to be a local PyTorch-style model library, not a collection of training projects.

## 分层 / Layering

### 1. Core framework / 核心框架层

Paths / 路径:

- `torch_like/models/`
- `torch_like/classical/`
- `torch_like/nn/`
- `torch_like/utils/`
- `torch_like/data/`
- `torch_like/evaluation/`
- `torch_like/training/`
- `torch_like/automl/`
- `governance/task_adapter_approvals.json`

中文：这里放可复用 API、模型定义、任务 batch、数据适配、数据集级评估、训练生命周期、registry 和基础工具。

English: This layer contains reusable APIs, model definitions, task batches, data adapters, dataset-level evaluation, training lifecycle, registry utilities, and basic helpers.

中文：`torch_like/automl/` 是可选编排子层，只通过公开 registry 和任务协议调用模型；它可以训练、比较和导出管线，但不得在运行时修改模型源代码或绕过模型治理。

English: `torch_like/automl/` is an optional orchestration sublayer that consumes public registry and task contracts. It may train, compare, and export pipelines, but may not mutate model source code at runtime or bypass model governance.

中文：`torch_like/data/` 负责可复用 DataLoader 物化；`torch_like/training/` 负责单优化器训练生命周期和 BatchAdapter；`torch_like/evaluation/` 负责跨完整数据集计算的指标。三者不得相互复制任务公式或数据转换。

English: `torch_like/data/` owns reusable DataLoader materialization, `torch_like/training/` owns the single-optimizer lifecycle and BatchAdapter contracts, and `torch_like/evaluation/` owns whole-dataset metric computation. These layers must not duplicate task formulas or data conversion logic.

AutoML module boundaries / AutoML 模块边界:

- `data.py`: shared dataset dispatch and tabular/sequence preparation.
- `esa_data.py`: ESA-ADB archive reading, resampling, labels, and time-window preparation.
- `anomaly_backend.py`: anomaly candidate training/scoring adapters and artifact export.
- `anomaly_catalog.py`: registry-driven detection-model discovery and audited leaderboard defaults.
- `anomaly_metrics.py`: point-wise and approximate event-wise anomaly metrics and validation-only threshold selection.
- `candidate_registry.py`: task compatibility, backend metadata, and audited candidate search defaults.
- `sklearn_backend.py`: classical machine-learning execution and candidate-plan expansion.
- `bundle.py`: trusted-local best-model loading, hash verification, preprocessing, and inference.
- `task.py`: immutable-task validation, trial resume, failure isolation, ranking, and sealed-test execution.

中文：数据集专属读取逻辑、模型训练适配、指标计算和排行榜编排必须分层；`task.py` 不直接实现 ESA 文件解析或异常模型结构。

中文：`governance/task_adapter_approvals.json` 将任务输出头/骨干适配与新论文模型准入分开管理，
同时保留机器可检查的来源、风险、测试和复审信息。

English: `governance/task_adapter_approvals.json` separates task-head/backbone adaptations from new
paper-model admissions while retaining machine-checkable source, risk, test, and review metadata.

Rules / 规则:

- 中文：模型必须继承 `nn.Module` 或 `BaseModel` 的任务子类。
- English: Models must inherit from `nn.Module` or a task-specific subclass of `BaseModel`.
- 中文：不要在模型文件中写数据集专属训练流程。
- English: Do not put dataset-specific training workflows inside model files.
- 中文：新增模型要接入对应 domain 的 `__init__.py` 和 registry。
- English: New models should be exported through the domain `__init__.py` and registered in the catalog.

### 2. Auxiliary experiment layer / 辅助实验层

Paths / 路径:

- `runner/`
- `examples/`

中文：runner 用于验证模型能否训练、评估、比较，但不是核心框架 API。

English: Runners validate whether models can be trained, evaluated, and compared, but they are not the core framework API.

中文：`examples/forecasting_rolling_validated.yaml` 是验证级排行榜模板，用于展示 `trust.level: validated`、三种子和 rolling-origin 封存测试的组合，不是固定基准结果。
English: `examples/forecasting_rolling_validated.yaml` is a validated leaderboard template showing `trust.level: validated`, three seeds, and rolling-origin sealed testing; it is not a fixed benchmark result.

Rules / 规则:

- 中文：runner 可以依赖数据加载和训练循环。
- English: Runners may depend on data loading and training loops.
- 中文：runner 不应该定义模型结构。
- English: Runners should not define model architectures.
- 中文：如果某段逻辑对所有模型通用，应优先抽到 `torch_like/data/`、`torch_like/training/` 或 `torch_like/evaluation/`；只有实验编排逻辑才保留在 `runner/common.py`。
- English: If logic is shared across models, prefer `torch_like/data/`, `torch_like/training/`, or `torch_like/evaluation/`; keep only experiment-orchestration helpers in `runner/common.py`.

### 3. Verification layer / 验证层

Path / 路径:

- `tests/`

中文：测试关注框架合同，不追求论文指标复现。

English: Tests focus on framework contracts, not paper-score reproduction.

中文：`torch_like/automl/protocol.py` 在训练前审计实际生成的 split；每次排行榜运行将结果保存为 `protocol_audit.json`。

English: `torch_like/automl/protocol.py` audits each materialized split before training; every leaderboard run stores the result as `protocol_audit.json`.

中文：同一模块还审计排行榜评估计划；`evaluation_trust_audit.json` 记录信任等级、种子覆盖、trial 唯一性和统一预算检查。

English: The same module also audits the leaderboard evaluation plan; `evaluation_trust_audit.json` records trust level, seed coverage, trial uniqueness, and uniform-budget checks.

Minimum coverage / 最低覆盖:

- 中文：直接实例化。
- English: Direct instantiation.
- 中文：registry 构造。
- English: Registry construction.
- 中文：`predict` / `fit_step` / `score_batch` 输出形状。
- English: Output shapes for `predict`, `fit_step`, and `score_batch`.
- 中文：顶层 lazy import。
- English: Top-level lazy imports.

### 4. Documentation layer / 文档层

Path / 路径:

- `README.md`
- `docs/`

中文：所有说明文档采用中英文双语格式。

English: All user-facing documentation should be bilingual in Chinese and English.

中文：`docs/AUTOML_BENCHMARK_TRUST.md` 维护排行榜 smoke/development/validated 分层、真实任务覆盖和晋升规则。
English: `docs/AUTOML_BENCHMARK_TRUST.md` maintains leaderboard smoke/development/validated tiers, real-task coverage, and promotion rules.

### 5. Generated artifact layer / 生成产物层

Paths / 路径:

- `runner/results/`
- future checkpoint or benchmark outputs

中文：生成产物不应成为框架 API 的一部分。

English: Generated artifacts should not become part of the framework API.

### 6. Packaging layer / 打包分发层

Paths / 路径:

- `pyproject.toml`
- `tools/build_package.ps1`
- `tools/verify_package_install.ps1`
- `dist/` (generated and ignored / 生成并忽略)

中文：打包层把核心 `torch_like/` 包构建为可由其他 Python 环境安装的 wheel。runner、测试、示例和本地数据不进入分发包。

English: The packaging layer builds the core `torch_like/` package into a wheel installable by other Python environments. Runners, tests, examples, and local data are excluded from distributions.

### 7. Model governance layer / 模型治理层

Paths / 路径:

- `governance/model_admission_policy.json`
- `torch_like/models/catalog_schema.py`
- `torch_like/models/catalog_schema_v2.json`
- `governance/model_catalog_audit.json`
- `governance/api_stability_policy.json`
- `governance/api_deprecations.json`
- `governance/public_api_baseline.json`
- `governance/model_registry_baseline.txt`
- `docs/MODEL_CANDIDATE_POOL.md`
- `docs/MODEL_CATALOG_AUDIT.md`
- `docs/CATALOG_SCHEMA_V2.md`
- `docs/API_STABILITY.md`
- `docs/MODEL_EVIDENCE.md`
- `docs/ROADMAP.md`
- `tools/audit_model_catalog.py`
- `tools/audit_public_api.py`
- `tools/validate_candidate_pool.py`

中文：治理层决定模型是否值得进入框架，不包含模型实现。政策文件是评分与状态转换的机器可读依据；候选池保存证据、同行比较、风险和复审台账；验证器在 CI 中阻止绕过准入流程或使用过期结论。

English: The governance layer decides whether a model merits framework admission; it does not contain model implementations. The policy file is the machine-readable source for scoring and transitions, the candidate pool records evidence, peer comparisons, risks, and review deadlines, and the validator blocks bypasses and stale decisions in CI.

中文：模型目录审计从实时 registry 确定性生成 JSON 和双语 Markdown，记录旧集成等级、schema v2 运行时字段、API 存在性和未评估证据。CI 会拒绝 registry 与已提交审计不一致的变更。schema 政策位于核心包内，因此源码环境与安装后的 wheel 使用同一规则。

English: The model-catalog audit deterministically generates JSON and bilingual Markdown from the live registry, recording legacy integration levels, runtime schema-v2 fields, API presence, and unassessed evidence. CI rejects registry changes that leave the committed audit stale. The schema policy ships in the core package so source checkouts and installed wheels use the same rules.

中文：公共 API 基线保护顶层、models、nn、data 和 automl 的公开导出及调用签名。兼容新增可以更新基线；删除、重命名、替换和破坏性签名变化必须经过机器可检查的弃用周期。

English: The public-API baseline protects exports and call signatures in the top-level, models, nn, data, and automl namespaces. Compatible additions may update the baseline; removals, renames, replacements, and breaking signature changes require a machine-checkable deprecation lifecycle.

中文：注册表基线记录治理制度启用前已经存在的模型。任何不在基线中的新 registry 条目，都必须对应候选池中带有 `implemented` 决策和真实 registry 名称的批准记录。

English: The registry baseline records models that predate enforcement. Any new registry entry outside that baseline must map to an approved candidate-pool record carrying an `implemented` decision and its real registry name.

## Registry Boundary / 注册边界

中文：registry 只注册可复用模型，不注册训练脚本、数据集脚本或一次性实验。

English: The registry should only register reusable models, not training scripts, dataset scripts, or one-off experiments.

Current domains / 当前域:

- `forecasting`
- `detection`
- `rul`
- `graph`
- `sequence`
- `vision`
- `multimodal`
- `generative`
- `diagnosis`
- `tabular`
- `recommendation`
- `audio`
- `reinforcement`

## Import Boundary / 导入边界

中文：PyTorch-native 模型提供 domain import、兼容的顶层短入口和 registry/factory 构造；sklearn 算法通过独立的 `torch_like.classical` 暴露原生 estimator，不伪装成 `nn.Module`。

English: PyTorch-native models support domain imports, compatible top-level shortcuts, and registry/factory construction. sklearn algorithms use the separate `torch_like.classical` namespace and remain native estimators rather than pretending to be `nn.Module` objects.

```python
import torch_like as tl
model = tl.create("PatchTST", domain="forecasting", ...)
baseline = tl.create("RandomForestRegressor", backend="sklearn", ...)
```

```python
from torch_like.models.forecasting import PatchTST
from torch_like.classical import RandomForestRegressor
```

中文：每个模型 domain 的 `__init__.py` 只承担 lazy facade；完整元数据登记放在私有 `_catalog.py`。新增模型不得重新把所有实现强导入到 `__init__.py`。

English: Each model-domain `__init__.py` is only a lazy facade; full metadata registration lives in private `_catalog.py`. New models must not reintroduce eager imports of every implementation into `__init__.py`.

## Data Boundary / 数据边界

中文：`torch_like/data/` 提供通用数据协议；数据集专属解析优先放在 runner 或外部项目。

English: `torch_like/data/` provides generic data protocols. Dataset-specific parsing should stay in runners or external projects.

## Practical Design Rule / 实用设计规则

中文：如果一个文件的主要价值是“定义模型结构”，放进 `torch_like/models/`；如果主要价值是“跑某个实验”，放进 `runner/`；如果主要价值是“解释如何使用”，放进 `docs/`。

English: If a file mainly defines model architecture, put it under `torch_like/models/`. If it mainly runs an experiment, put it under `runner/`. If it explains usage, put it under `docs/`.
