# Model Evidence Registry / 模型证据登记

## Purpose / 目的

中文：`torch_like` 使用 [`torch_like/models/model_evidence.json`](../torch_like/models/model_evidence.json) 记录框架内真实执行过的模型基准。证据记录决定 catalog 的 `maturity`、`benchmark_evidence` 和 `evidence_record`，防止根据论文名、引用量或 smoke test 自动提升成熟度。

English: `torch_like` uses [`torch_like/models/model_evidence.json`](../torch_like/models/model_evidence.json) to record benchmarks actually executed inside the framework. Records drive `maturity`, `benchmark_evidence`, and `evidence_record`, preventing promotion based only on paper names, citations, or smoke tests.

## Meaning / 含义

- `benchmarked + single_dataset`：模型在一个真实数据集上完成了有验证集和独立测试边界的本地实验。/ The model completed a local real-dataset experiment with validation and separate test boundaries.
- 它不表示论文复现、SOTA、跨数据集稳健性或生产可用。/ It does not mean paper reproduction, SOTA, cross-dataset robustness, or production readiness.
- `unassessed` 不表示模型无效，只表示当前仓库没有足够的登记证据。/ `unassessed` does not mean invalid; it means the repository lacks sufficient registered evidence.

## Current Coverage / 当前覆盖

| Domain / 领域 | Dataset / 数据集 | Models / 模型数 | Evidence boundary / 证据边界 |
|---|---|---:|---|
| Forecasting / 预测 | ETTh1 | 3 | Rolling-origin, one seed, framework protocol |
| Detection / 异常检测 | ESA-ADB Mission1 | 9 | Six-channel ESA-lite v3, one seed |
| RUL / 寿命预测 | XJTU-SY | 7 | One condition; single split or group-CV |

共有 19 个不同模型获得本地 benchmark 标记，其他模型保持 `unassessed`。/ Nineteen distinct models carry local benchmark evidence; all others remain `unassessed`.

## Query API / 查询接口

```python
import torch_like as tl

info = tl.get_model_info("DLinear", domain="forecasting")
record = tl.get_evidence_record(info["evidence_record"])
rul_records = tl.list_evidence_records(domain="rul")
```

返回值是防御性副本，调用方修改不会污染全局 registry。/ Returned values are defensive copies and cannot mutate the global registry.

## Promotion Rules / 晋升规则

1. 必须登记任务、数据集、协议、模型列表、配置、报告、随机种子和局限。/ Record task, dataset, protocol, models, config, report, seed, and limitations.
2. 配置与报告必须是仓库内存在的相对路径。/ Config and report must be existing repository-relative paths.
3. `evaluated_models` 必须属于记录指定的 domain。/ Every evaluated model must belong to the declared domain.
4. 单数据集证据最多晋升为 `benchmarked + single_dataset`。/ Single-dataset evidence can promote at most to `benchmarked + single_dataset`.
5. `production_ready` 需要独立的多数据集、回归、兼容性和运维证据，本轮没有任何模型达到该等级。/ `production_ready` requires separate multi-dataset, regression, compatibility, and operational evidence; no model currently qualifies.

Run the deterministic gate after changes / 修改后运行确定性门禁：

```powershell
D:\Project\base\base\Scripts\python.exe tools\audit_model_catalog.py --write
D:\Project\base\base\Scripts\python.exe tools\audit_model_catalog.py --check
```
