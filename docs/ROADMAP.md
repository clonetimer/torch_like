# torch_like Roadmap / 长期路线图

中文：`torch_like` 的长期目标是成为建立在 PyTorch 之上的、本地优先、接口统一、证据驱动的多领域模型框架。用户既可以像使用普通 `torch.nn.Module` 一样直接组合模型，也可以通过可选的 AutoML 层，从数据和任务声明得到可复现的候选排名与可信本地模型包。

English: `torch_like` aims to become a local-first, interface-consistent, evidence-driven, multi-domain model framework built on PyTorch. Users should be able to compose models like ordinary `torch.nn.Module` objects or use the optional AutoML layer to turn explicit datasets and task specifications into reproducible rankings and trusted-local model bundles.

## Non-goals / 非目标

- 不重新实现 Tensor、autograd、优化器、分布式运行时或 CUDA 内核。 / Do not reimplement tensors, autograd, optimizers, distributed runtimes, or CUDA kernels.
- 不把 sklearn 模型伪装成 `nn.Module`。 / Do not disguise sklearn estimators as `nn.Module` models.
- 不把 runner 提升为核心框架 API。 / Keep runners outside the core framework API.
- 不以模型数量或 smoke 分数作为质量目标。 / Do not use model count or smoke scores as quality targets.
- 不把接口测试通过解释为论文复现或生产可用。 / Do not equate passing contract tests with paper reproduction or production readiness.
- AutoML 不在运行时修改模型源码或绕过模型治理。 / AutoML must not mutate model source code or bypass model governance.

## Architecture / 架构主线

| Layer / 层级 | Responsibility / 职责 |
|---|---|
| PyTorch foundation / PyTorch 基础 | Tensor, autograd, devices, compilation, distributed and kernel execution / 张量、自动微分、设备、编译、分布式和内核执行 |
| Model framework / 模型框架 | Model classes, task batches, direct imports and registry construction / 模型类、任务 Batch、直接导入和 registry 构造 |
| Task protocols / 任务协议 | Classification, regression, forecasting, RUL and anomaly semantics / 分类、回归、预测、寿命和异常检测语义 |
| AutoML orchestration / AutoML 编排 | Data profiling, leakage-aware splits, candidates, ranking and bundles / 数据画像、防泄漏划分、候选、排名和模型包 |
| Governance and delivery / 治理与交付 | Admission evidence, audits, benchmarks, compatibility, packaging and documentation / 准入证据、审计、基准、兼容性、打包和文档 |

## Milestones / 里程碑

### M1 Framework Convergence / 框架收敛

Target / 目标: `0.4.x`

- 审计全部 registry 模型的能力、旧 `integration_level`、API 信号和证据缺口。 / Audit capabilities, legacy integration levels, API signals, and evidence gaps for every registry model.
- 将实现一致性、实现范围、执行方式和成熟度拆成正交字段。 / Separate fidelity, implementation scope, execution profile, and maturity into orthogonal fields.
- 定义公共 API 稳定性基线和弃用规则。 / Define the public-API stability baseline and deprecation policy.
- 新模型继续执行证据驱动准入，不设置数量目标。 / Continue evidence-driven admission without a model-count target.

Exit criteria / 退出条件:

- registry 审计可确定性生成并由 CI 检查。 / The registry audit is deterministic and CI-enforced.
- 所有旧 `integration_level` 均有无歧义迁移映射。 / Every legacy integration level has an unambiguous migration mapping.
- 每个公共模型都有明确成熟度状态；未知状态必须显式标为 `unassessed`。 / Every public model has an explicit maturity state, with unknowns marked `unassessed`.
- 公共 API 变更必须经过兼容性检查和弃用周期。 / Public API changes require compatibility checks and a deprecation window.

Current status / 当前状态:

- 已完成确定性 model catalog 审计与 CI 漂移门禁。 / Deterministic model-catalog audit and CI drift gate are complete.
- 已完成 schema v2 兼容运行时字段与 wheel 分发验证。 / Additive schema-v2 runtime fields and wheel distribution verification are complete.
- 已完成公共 API 基线、弃用台账、transition validator 和 CI 门禁。 / Public-API baseline, deprecation ledger, transition validator, and CI gate are complete.
- 待完成按 domain 登记真实 maturity/evidence。 / Domain-level maturity/evidence registration remains pending.

### M2 Trustworthy Task Platform / 可信任务平台

Target / 目标: `0.5.x`–`0.6.x`

- 补齐多目标回归的真实端到端任务。 / Add a real end-to-end multi-target regression task.
- 为 AutoML 增加任务相关候选路由和公平训练预算。 / Add task-aware candidate routing and fair training budgets.
- 优化顺序固定为预处理、训练超参、模型超参、校准/集成、入选后剪枝。 / Prioritize preprocessing, training hyperparameters, model hyperparameters, calibration/ensembling, and post-selection pruning in that order.
- 自动结构修改保持实验状态，不进入默认生产搜索。 / Keep automatic architecture mutation experimental and outside default production search.

Exit criteria / 退出条件:

- 每个声明支持的任务至少有一个真实数据协议和一个强基线。 / Every declared task has at least one real-data protocol and one strong baseline.
- 所有排行榜任务执行防泄漏划分、多种子聚合、失败隔离和一次密封测试。 / All leaderboards enforce leakage-aware splits, multi-seed aggregation, failure isolation, and one-shot sealed testing.
- 任务产物可以通过 `load_best()` 在可信本地环境恢复。 / Task artifacts can be restored through `load_best()` in a trusted local environment.

### M3 Evidence and Benchmark Platform / 证据与基准平台

Target / 目标: `0.7.x`

- 区分 contract、numerical、task、performance 和 paper-reproduction 五类证据。 / Separate contract, numerical, task, performance, and paper-reproduction evidence.
- 为每个核心模型族维护代表性 CPU/CUDA 延迟、吞吐、内存和数值一致性基准。 / Maintain representative CPU/CUDA latency, throughput, memory, and numerical-consistency benchmarks for each core family.
- 自动生成模型卡，记录适用范围、限制、依赖、许可证来源和证据。 / Generate model cards covering scope, limits, dependencies, license provenance, and evidence.

Exit criteria / 退出条件:

- `benchmarked` 和 `production_ready` 均必须引用可复核运行产物。 / `benchmarked` and `production_ready` states reference auditable run artifacts.
- 性能优化具有基线、容差、目标设备和回退路径。 / Performance changes declare a baseline, tolerance, target device, and fallback.
- 过期证据和性能退化能够阻止质量门禁。 / Stale evidence and tracked performance regressions can fail quality gates.

### M4 Stable Delivery / 稳定交付

Target / 目标: `0.8.x`–`0.9.x`

- 建立 Python/PyTorch/设备兼容矩阵和隔离安装验证。 / Establish a Python/PyTorch/device compatibility matrix and isolated installation verification.
- 稳定 checkpoint、配置、预处理器和 `ModelBundle` 格式。 / Stabilize checkpoint, configuration, preprocessor, and `ModelBundle` formats.
- 完善离线 wheelhouse、产物哈希和不可信文件边界。 / Complete offline wheelhouse, artifact hashing, and untrusted-file boundaries.

### M5 Stable API / 稳定 API

Target / 目标: `1.0`

`1.0` 不是模型数量里程碑。它要求公共 API 跨多个次版本保持稳定，全部公共模型具有明确成熟度，核心任务具备真实基准，AutoML 防泄漏与复现机制稳定，并且打包与离线部署通过独立环境验证。

`1.0` is not a model-count milestone. It requires public APIs to remain stable across multiple minor releases, every public model to have explicit maturity, core tasks to have real benchmarks, AutoML leakage/reproducibility controls to be stable, and packaging plus offline deployment to pass isolated-environment verification.

## Current Path / 当前路径

```text
Long-term framework goal / 长期框架目标
└── M1 Framework convergence / 框架收敛
    ├── Deterministic model catalog audit / 确定性模型目录审计
    ├── Catalog schema v2 migration / catalog schema v2 迁移
    ├── Public API stability baseline / 公共 API 稳定性基线
    └── Deprecation policy / 弃用策略
```

中文：里程碑只能在其退出条件有代码、测试、benchmark 或人工审计证据时关闭。完成一个子任务不等于完成所属里程碑。

English: A milestone closes only when its exit criteria have code, tests, benchmarks, or human-review evidence. Completing one child task does not complete its parent milestone.

M1 evidence status / M1 证据状态：结构化证据登记和自动校验已完成，当前覆盖 forecasting、detection、RUL 的 19 个单数据集实测模型；未覆盖模型继续显式保持 `unassessed`。M1 的治理基础已具备，后续成熟度扩展按领域逐项进行，不批量晋升。

Structured evidence registration and automated validation are complete. Nineteen single-dataset benchmarked models across forecasting, detection, and RUL are covered; all others explicitly remain `unassessed`. The M1 governance foundation is in place, while future maturity expansion remains domain-by-domain rather than bulk promotion.

M2 current status / M2 当前状态：排行榜任务现在会在训练前生成并强制通过 `protocol_audit.json`，对普通划分、group-CV 和 rolling-origin 执行实际索引、组和时间边界审计；同时生成 `evaluation_trust_audit.json`，把 `development`/`validated` 信任等级、多种子覆盖和统一预算检查纳入可审计协议。下一项是把真实任务基准按信任等级分层整理，并补齐 `validated` 示例配置。

Leaderboard tasks now generate and enforce `protocol_audit.json` before training, auditing materialized indices, groups, and time boundaries for standard splits, group-CV, and rolling-origin. They also generate `evaluation_trust_audit.json`, making `development`/`validated` trust levels, multi-seed coverage, and uniform-budget checks auditable. Real-task benchmark tiers are documented in `docs/AUTOML_BENCHMARK_TRUST.md`, validated templates exist for forecasting and XJTU-SY RUL, `tools/validate_automl_examples.py` keeps example configs checkable in CI, and `tools/validate_automl_runs.py` checks completed run artifacts. The next M2 step is to register selected completed validated runs as model/task evidence rather than manually curating claims.
