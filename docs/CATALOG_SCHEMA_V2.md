# Model Catalog Schema v2 / 模型目录 Schema v2

中文：本文定义 M1 阶段的 catalog schema v2 兼容实现。registry 已以加法方式返回 v2 字段，不会删除现有 `integration_level`，也不会根据模型名称自动授予成熟度或 benchmark 等级。

English: This document defines the M1 catalog schema-v2 compatibility implementation. The registry now returns additive v2 fields without removing `integration_level`, and no model receives maturity or benchmark status from its name.

## Baseline Evidence / 基线证据

首版确定性审计记录于 [`MODEL_CATALOG_AUDIT.md`](MODEL_CATALOG_AUDIT.md)：

- 13 个公开 domain、165 个注册模型。 / 13 public domains and 165 registered models.
- 165 个类均继承 `torch.nn.Module`。 / All 165 classes inherit `torch.nn.Module`.
- 11 种旧 `integration_level` 值全部具有显式迁移映射。 / All 11 legacy `integration_level` values have explicit migration mappings.
- 165 个模型的 maturity 和 benchmark evidence 均保持 `unassessed`。 / Maturity and benchmark evidence remain `unassessed` for all 165 models.
- `predict` 和 task-loss 入口不是跨域统一要求；应由 domain contract 决定。 / `predict` and task-loss entry points are domain-specific rather than universal requirements.

## Problem / 问题

旧 `integration_level` 同时编码四类信息：

1. 是否接近论文结构。 / Fidelity to the described architecture.
2. 实现的是完整模型、核心机制还是子集。 / Full, core, or subset implementation scope.
3. 使用 dense、portable scan 或 LSH 等何种执行方式。 / Dense, portable-scan, LSH, or standard execution.
4. `reference` 一词有时表达实现边界，但容易被误读为成熟度。 / `reference` sometimes expresses implementation scope but can be mistaken for maturity.

复合字符串不利于查询、自动治理和兼容演进。例如 `paper_aligned_portable_subset` 无法直接回答模型是否经过 benchmark，也无法与 `paper_aligned_core` 独立比较执行方式。

Compound strings are difficult to query, govern, and evolve. For example, `paper_aligned_portable_subset` does not say whether a model has benchmark evidence and cannot independently compare execution profile with `paper_aligned_core`.

## Target Fields / 目标字段

| Field / 字段 | Meaning / 含义 | Initial values / 首批取值 |
|---|---|---|
| `implementation_fidelity` | 结构实现与公开架构说明的关系 / relationship to the published architecture | `framework_oriented`, `paper_aligned`, `reference` |
| `implementation_scope` | 实现覆盖范围 / implementation coverage | `full`, `core`, `subset`, `mlstm_only` |
| `execution_profile` | 主要执行路径 / primary execution path | `standard`, `dense`, `portable`, `portable_scan`, `lsh` |
| `maturity` | 框架内验证成熟度 / framework validation maturity | `unassessed`, `contract_validated`, `benchmarked`, `production_ready` |
| `benchmark_evidence` | 任务基准覆盖 / task-benchmark coverage | `unassessed`, `smoke_only`, `single_dataset`, `multi_dataset` |
| `evidence_record` | 支撑成熟度或 benchmark 晋升的记录 / record supporting maturity or benchmark promotion | `null` or a non-empty record identifier / `null` 或非空记录标识 |

`integration_level` 在迁移期继续作为只读兼容字段。唯一迁移映射随核心 wheel 分发，位于 [`../torch_like/models/catalog_schema_v2.json`](../torch_like/models/catalog_schema_v2.json)。

`integration_level` remains a read-only compatibility field during migration. The canonical mapping ships in the core wheel at [`../torch_like/models/catalog_schema_v2.json`](../torch_like/models/catalog_schema_v2.json).

## Evidence Rules / 证据规则

- `contract_validated`：必须有该模型或其明确共享契约的构造、前向、梯度、配置往返和边界测试。 / Requires construction, forward, gradient, configuration round-trip, and boundary tests for the model or an explicitly shared contract.
- `benchmarked`：除 contract 证据外，必须有真实数据、正式超参、协议说明、运行 ID、环境和结果文件。 / Additionally requires real data, regular hyperparameters, protocol documentation, run ID, environment, and result artifacts.
- `production_ready`：除 benchmark 证据外，必须有受支持环境、序列化/恢复、资源上限、已知失败模式和回退路径。 / Additionally requires supported environments, serialization/restoration, resource bounds, known failure modes, and fallback paths.
- smoke 只证明兼容性，不足以晋升到 `benchmarked`。 / Smoke execution proves compatibility only and cannot promote a model to `benchmarked`.
- 单元测试数量、论文引用量和模型名称都不能单独触发晋升。 / Test count, citation count, and model names cannot independently trigger promotion.

## Domain Contracts / 领域契约

所有模型必须支持 `nn.Module` 对象模型、`config()`、`from_config(...)` 和 `param_count()`。其余接口按 domain 定义：

All models must support the `nn.Module` object model, `config()`, `from_config(...)`, and `param_count()`. Remaining interfaces are domain-specific:

- 监督任务模型通常需要 `predict(...)` 与 `loss_step(...)`/`fit_step(...)`。 / Supervised task models generally require prediction and task-loss entry points.
- 表征模型可以提供 embedding/feature 接口，而不强制定义分类损失。 / Representation models may expose embedding or feature APIs without a classification loss.
- 生成模型需要采样或生成入口，其训练目标可能不是统一 `loss_step`。 / Generative models require sampling or generation APIs and may not share one loss signature.
- 强化学习模型暴露 Q、policy、value 或算法损失，不强制使用普通分类预测。 / Reinforcement models expose Q, policy, value, or algorithm losses rather than ordinary classification prediction.

因此，审计中的 API signal 只能用于发现待审查项，不能跨 domain 自动判定合格或失败。

API signals in the audit are therefore review indicators, not cross-domain automatic pass/fail criteria.

## Migration Stages / 迁移阶段

1. 当前已完成：确定性审计、映射策略和漂移门禁。 / Completed: deterministic audit, mapping policy, and drift gate.
2. 当前已完成：以加法方式返回 v2 派生字段，保留 `integration_level`。 / Completed: additive v2 derived fields while retaining `integration_level`.
3. `0.4.x`：按 domain 审计并登记真实 maturity/evidence，不允许批量自动晋升。 / Audit domains and record real maturity/evidence without bulk automatic promotion.

当前已建立结构化证据登记：19 个在 ETTh1、ESA-ADB Mission1 或 XJTU-SY 上完整执行的模型标记为 `benchmarked + single_dataset`，其余模型保持 `unassessed`。具体边界见 [`MODEL_EVIDENCE.md`](MODEL_EVIDENCE.md)。

A structured evidence registry is now active: 19 models fully executed on ETTh1, ESA-ADB Mission1, or XJTU-SY are marked `benchmarked + single_dataset`; all remaining models stay `unassessed`. See [`MODEL_EVIDENCE.md`](MODEL_EVIDENCE.md) for boundaries.
4. `0.5.0`：内置 catalog 原生写入 v2 字段，旧字段由兼容层生成。 / Built-in catalogs write native v2 fields and the compatibility layer derives the legacy field.
5. `1.0` 之前：评估是否弃用旧字段；若弃用，至少跨两个次版本发出警告。 / Before `1.0`, evaluate legacy-field deprecation with warnings across at least two minor releases.

## Acceptance Gates / 验收门禁

- 新增旧复合值必须先更新唯一映射政策、审计报告和测试。 / A new compound legacy value requires policy, audit, and test updates first.
- 未映射值使 CI 失败。 / Unmapped values fail CI.
- registry 变化但未重新生成审计时 CI 失败。 / Registry changes with a stale audit fail CI.
- maturity 晋升必须引用独立证据记录。 / Maturity promotion must reference a separate evidence record.
- v2 字段不得改变模型构造、前向输出或 checkpoint。 / V2 metadata must not change construction, forward outputs, or checkpoints.
