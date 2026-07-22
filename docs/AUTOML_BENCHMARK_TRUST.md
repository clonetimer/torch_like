# AutoML Benchmark Trust / AutoML 基准可信分层

中文：本文件定义 `torch_like.automl` 真实任务排行榜的可信分层。它解决的是“这个排行榜结果能被怎样使用”的问题，而不是“某个模型是否复现论文 SOTA”。模型成熟度仍由 model catalog evidence 管理。

English: This document defines trust tiers for real-task `torch_like.automl` leaderboards. It answers how a leaderboard result may be used, not whether a model reproduces paper SOTA. Model maturity remains governed by model-catalog evidence.

## Trust Tiers / 可信层级

| Tier / 层级 | Required config / 必要配置 | Required artifacts / 必要产物 | Valid use / 合理用途 |
|---|---|---|---|
| `smoke` | Small budget, often single seed / 小预算，通常单种子 | `config.json`, `plan.json`, `protocol_audit.json`, `leaderboard.csv` | Check ingestion, adapter compatibility, and failure isolation / 检查读取、适配器兼容和失败隔离 |
| `development` | Default `trust.level: development`; sealed test required by the split protocol / 默认 `trust.level: development`；划分协议必须有封存测试 | Smoke artifacts plus `evaluation_trust_audit.json` and sealed-test record / smoke 产物加评估信任审计和封存测试记录 | Compare candidates during framework development / 框架开发中的候选比较 |
| `validated` | `trust.level: validated` and at least 3 `search.evaluation_seeds` / `trust.level: validated` 且至少 3 个评估种子 | Development artifacts plus complete seed coverage and budget checks / development 产物加完整种子覆盖和预算检查 | Formal local comparisons under one declared protocol / 一个明确协议下的正式本地比较 |

中文：`validated` 仍然不是论文级 SOTA 声明。它只说明在当前数据、划分、候选集、预算和指标下，排行榜比较过程经过可审计约束。

English: `validated` is still not a paper-level SOTA claim. It only means the comparison was audited under the declared data, split, candidate pool, budget, and metric protocol.

## Static Example Checks / 示例静态检查

中文：`tools/validate_automl_examples.py` 会在不读取外部数据、不训练模型的情况下检查 `examples/*.yaml`。它保证示例能被 `load_config()` 解析，`task_id` 不重复，`validated` 或 `benchmark` 命名与 `trust.level` 一致，验证级示例至少有三个评估种子，并且 `max_trials` 足以让显式列出的算法至少各获得一次机会。

English: `tools/validate_automl_examples.py` checks `examples/*.yaml` without reading external data or training models. It guarantees that examples load through `load_config()`, task ids are unique, `validated` or `benchmark` naming agrees with `trust.level`, validated examples declare at least three evaluation seeds, and `max_trials` can give every explicitly listed algorithm at least one chance.

## Completed Run Checks / 已完成运行检查

中文：`tools/validate_automl_runs.py` 校验已经生成的排行榜运行目录。它要求核心产物完整，`protocol_audit.json` 与 `evaluation_trust_audit.json` 通过，`plan.json` 与 `config.json` 的任务和信任等级一致，`leaderboard.csv` 不包含 `test_*` 指标列，`sealed_test_metrics.json` 在完成状态下包含最终测试指标。对于 `validated` 运行，它还要求至少三个种子、入榜完成候选的 `completed_seed_count == seed_count`，以及完成的一次性封存测试。

English: `tools/validate_automl_runs.py` validates generated leaderboard run directories. It requires core artifacts, passed `protocol_audit.json` and `evaluation_trust_audit.json`, matching task and trust level between `plan.json` and `config.json`, no `test_*` metric columns in `leaderboard.csv`, and final test metrics in a completed `sealed_test_metrics.json`. For `validated` runs, it additionally requires at least three seeds, `completed_seed_count == seed_count` for completed ranked candidates, and one completed sealed test.

```powershell
python tools/validate_automl_runs.py runs/forecasting-rolling-validated
python tools/validate_automl_runs.py --skip-legacy runs
python tools/validate_automl_runs.py --allow-empty --skip-legacy runs
```

## Current Real-Task Coverage / 当前真实任务覆盖

| Task / 任务 | Example / 示例 | Current tier / 当前层级 | Notes / 说明 |
|---|---|---|---|
| Tabular classification / 表格分类 | `examples/tabular_classification_leaderboard.yaml` | `development-ready` | Uses CSV placeholders and three seeds, but users must provide data before a real validated run / 使用 CSV 占位路径和三种子，真实 validated 运行需用户提供数据 |
| Time-series classification / 时序分类 | `examples/time_series_classification_leaderboard.yaml` | `smoke/development` | FordA-style long CSV contract; current example uses one seed / FordA 长表 CSV 协议；当前示例为单种子 |
| Time-series forecasting / 时序预测 | `examples/forecasting_rolling_leaderboard.yaml` | `development` | ETTh1 rolling-origin protocol, single-seed by default / ETTh1 rolling-origin 协议，默认单种子 |
| Time-series forecasting validated / 验证级时序预测 | `examples/forecasting_rolling_validated.yaml` | `validated template` | Adds `trust.level: validated` and three seeds / 增加验证级信任声明和三种子 |
| RUL regression / 寿命预测 | `examples/xjtu_sy_rul_group_cv_benchmark.yaml` | `validated template` | Group-CV plus sealed bearing test with three seeds; expensive on full local data / group-CV 加封存 bearing 测试和三种子；完整本地数据上运行成本较高 |
| Anomaly detection / 异常检测 | `examples/esa_mission1_anomaly_leaderboard.yaml` | `development` | Real ESA-lite approximation; validation period remains event-sparse / 真实 ESA-lite 近似；验证期事件仍较少 |

## Promotion Rules / 晋升规则

中文：一个真实任务结果从 `development` 晋升为 `validated`，至少需要满足：

- 配置显式声明 `trust.level: validated`。
- `search.evaluation_seeds` 至少 3 个且无重复。
- `protocol_audit.json` 和 `evaluation_trust_audit.json` 均为 `passed`。
- `leaderboard.csv` 中入榜候选的 `completed_seed_count == seed_count`。
- `sealed_test_metrics.json` 只在最佳管线选定后产生一次。
- 报告必须说明数据路径、数据指纹、代码版本、依赖版本、候选集、预算和主指标。

English: Promoting one real-task result from `development` to `validated` requires:

- Explicit `trust.level: validated`.
- At least three unique `search.evaluation_seeds`.
- Both `protocol_audit.json` and `evaluation_trust_audit.json` have `status: passed`.
- Ranked candidates in `leaderboard.csv` have `completed_seed_count == seed_count`.
- `sealed_test_metrics.json` is produced only once after selecting the best pipeline.
- The report states data path, data fingerprint, code version, dependency versions, candidate pool, budget, and primary metric.

## Non-Goals / 非目标

中文：以下内容不由 `validated` 自动保证：

- 复现论文官方实验。
- 覆盖多个公开数据集。
- 证明模型生产可用。
- 证明候选池包含该任务所有强模型。
- 自动选择最优预处理或最优超参空间。

English: `validated` does not automatically guarantee:

- Reproduction of official paper experiments.
- Coverage of multiple public datasets.
- Production readiness.
- Inclusion of every strong model for the task.
- Globally optimal preprocessing or hyperparameter spaces.
