# Public API Stability / 公共 API 稳定性

中文：本政策从当前 `0.3.x` 基线开始保护用户可见 API。项目仍处于 alpha 阶段，但“alpha”不再意味着可以无记录地删除、重命名或改变公共调用方式。

English: This policy protects user-visible APIs from the current `0.3.x` baseline onward. The project remains alpha, but alpha no longer permits undocumented removal, renaming, or breaking calling-convention changes.

## Protected Scope / 保护范围

机器基线覆盖以下命名空间的 `__all__` 导出、符号身份和公开调用签名：

The machine baseline covers `__all__` exports, symbol identity, and public call signatures in:

- `torch_like`
- `torch_like.models`
- `torch_like.nn`
- `torch_like.data`
- `torch_like.automl`

模型 registry 另外由模型目录审计和候选治理保护。`runner/`、以下划线开头的内部符号、局部实现函数、计算图内部结构和测试辅助代码不属于稳定 API。

The model registry is additionally protected by model-catalog auditing and admission governance. Runners, underscore-prefixed internals, local implementation functions, computation-graph internals, and test helpers are not stable APIs.

## Compatibility Rules / 兼容规则

无需弃用流程的兼容变化：

Compatible changes that do not require deprecation:

- 新增公共符号。 / Adding a public symbol.
- 新增具有默认值的可选参数。 / Adding an optional parameter with a default.
- 修复不改变声明接口的内部错误。 / Fixing internals without changing the declared interface.
- 增加文档、测试、性能路径或 metadata。 / Adding documentation, tests, performance paths, or metadata.

必须经过弃用流程的变化：

Changes requiring deprecation:

- 删除或重命名公共符号。 / Removing or renaming a public symbol.
- 替换符号所指向的公共对象。 / Replacing the public object behind a symbol.
- 删除参数、改变参数 kind、增加必填参数或修改默认值。 / Removing parameters, changing parameter kind, adding required parameters, or changing defaults.
- 改变公开返回结构且无法保持兼容。 / Incompatibly changing a public return structure.

## Deprecation Lifecycle / 弃用周期

1. 在 [`../governance/api_deprecations.json`](../governance/api_deprecations.json) 登记符号、变化类型、替代项、原因、版本和日期。 / Register the symbol, change type, replacement, reason, versions, and dates.
2. 保留旧 API，并通过 `warn_deprecated(...)` 发出 `FutureWarning`。 / Keep the old API and emit `FutureWarning` through `warn_deprecated(...)`.
3. 文档和示例切换到新 API。 / Move documentation and examples to the new API.
4. 至少经过两个 minor 版本且不少于 90 天。 / Wait at least two minor releases and 90 calendar days.
5. 删除时把台账状态改为 `removed`，提升版本并重新生成基线。 / At removal, mark the ledger record `removed`, bump the version, and regenerate the baseline.

当前机器政策位于 [`../governance/api_stability_policy.json`](../governance/api_stability_policy.json)，确定性基线位于 [`../governance/public_api_baseline.json`](../governance/public_api_baseline.json)。

The machine policy is [`../governance/api_stability_policy.json`](../governance/api_stability_policy.json), and the deterministic baseline is [`../governance/public_api_baseline.json`](../governance/public_api_baseline.json).

## Developer Workflow / 开发流程

检查当前 API：

Check the current API:

```powershell
& 'D:\Project\base\base\Scripts\python.exe' tools\audit_public_api.py --check
```

完成兼容新增后更新基线：

Update the baseline after a compatible addition:

```powershell
& 'D:\Project\base\base\Scripts\python.exe' tools\audit_public_api.py --write
```

`--write` 不是绕过门禁的命令。它会先比较旧基线与新 API；未登记、未到版本、未到日期或未标记 `removed` 的破坏性变化会被拒绝。

`--write` is not a bypass. It compares the old baseline with the new API first and rejects breaking changes that are unregistered, too early by version or date, or not marked `removed`.

## Versioning Boundary / 版本边界

- Patch 版本只能包含兼容修复。 / Patch releases contain compatible fixes only.
- Minor 版本可以增加 API，并承载弃用警告。 / Minor releases may add APIs and carry deprecation warnings.
- Major 版本可以完成已经满足弃用周期的破坏性迁移。 / Major releases may complete breaking migrations that satisfied the deprecation lifecycle.
- 安全或严重正确性问题如果无法等待普通周期，必须先提升机器政策版本并记录人工裁决，不能静默改写基线。 / Security or severe correctness emergencies that cannot wait require a machine-policy version bump and recorded human decision; the baseline must not be silently rewritten.

## Evidence Boundary / 证据边界

API 稳定只说明调用契约受到保护，不表示模型已经 benchmark、论文复现或生产验证。模型成熟度继续由 catalog schema v2 和独立证据记录管理。

API stability protects calling contracts only. It does not imply benchmarking, paper reproduction, or production validation. Model maturity remains governed by catalog schema v2 and separate evidence records.
