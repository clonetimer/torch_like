# R4c — 跨设备与跨工况故障分类泛化评估协议

## 适用范围

R4c 只支持**封闭类别集合、单标签**的故障类型分类，不宣称根因隔离或未知故障识别。它保留 R4a 的分类模型、训练协议与一次性封存测试；新增物理设备和工况身份，专门用于防泄漏的分组划分以及测试后分域统计。

- `FaultDomainPartition`：对 R4a `FaultClassificationPartition` 补充每个窗口的 `device_ids` 和 `condition_ids`。单个原始记录必须属于一个确定的设备/工况组合。
- `FaultGeneralizationSplit`：复用 R4a 样本 ID 和原始记录 ID 的不交叉检查，并对训练、可选验证和封存测试施加额外的域不交叉约束。
- `FaultGeneralizationWorkflow`：训练仅访问训练窗口及其故障标签；调用 R4a 工作流在封存测试上推理**一次**；只有获取测试预测后，才使用测试真值构建指标。

## 四种可审计的 holdout

| `holdout` | 要求 |
| --- | --- |
| `device` | train / validation / test 的设备 ID 两两不交叉 |
| `condition` | 运行工况 ID 两两不交叉 |
| `both` | 设备 ID 与工况 ID 都分别两两不交叉 |
| `pair` | 仅要求 (设备 ID, 工况 ID) 组合不交叉，允许单独设备或工况重复 |

`pair` 只能用于评估未见过的设备-工况**组合**，不能报告为对未见过设备、未见过工况的泛化能力。验证域也须遵守同一隔离协议，防止用封存测试域进行模型筛选。

## 报告语义

`FaultGeneralizationEvaluation.to_dict()` 同时给出整体分类结果，以及按设备、工况、设备/工况组合、原始记录四类维度切片的混淆矩阵。每个切片均报告独立记录数、窗口数、各类别真值支持数、缺失真值类别、accuracy、`macro_f1_all_classes`、`macro_f1_present_classes`。

固定类别集合上的 `macro_f1_all_classes` 规定未出现类别得分为零；`macro_f1_present_classes` 只平均实际出现的真值类别，**不能**直接对类别组成不同的切片比较高低。

少于 `min_domain_windows` 个窗口（默认 5）的切片被标记为 `sufficient_window_support=False`。这只是最基本的样本数提醒，不代表误差条、置信区间或统计显著性。多个窗口来自同一运行记录时，它们具有相关性，不能被当作独立设备重复实验。

## 使用示意

```python
from torch_like.diagnostics import (
    FaultClassificationPartition,
    FaultDomainPartition,
    FaultGeneralizationSplit,
    FaultGeneralizationWorkflow,
    FaultLabelSchema,
    SklearnWindowFaultClassifier,
)

schema = FaultLabelSchema(("normal", "bearing", "imbalance"))
# train_signals/test_signals: SignalBatch，包含唯一 sample_ids
train = FaultDomainPartition(
    FaultClassificationPartition(train_signals, train_recording_ids, train_labels),
    train_device_ids,
    train_condition_ids,
)
test = FaultDomainPartition(
    FaultClassificationPartition(test_signals, test_recording_ids, test_labels),
    test_device_ids,
    test_condition_ids,
)
split = FaultGeneralizationSplit(schema, train, test, holdout="device")
evaluation = FaultGeneralizationWorkflow(
    SklearnWindowFaultClassifier(random_state=0)
).run(split)
report = evaluation.to_dict()
```

## 数据和实验边界

1. 所有设备、工况及原始记录 ID 必须是可信且稳定的物理来源标识；框架不可能靠字符串验证物理数据是否被重命名或复制。
2. 如果同一连续原始记录包含多种工况，须上游明确处理、登记其**共同物理来源**，不能把同一原始记录拆分重命名后跨训练/测试使用。
3. 故障类别须预定义且全部在训练数据中出现；未知类别识别是另一项任务。
4. 采样率、传感器名称/次序、单位和时间窗长度仍须与模型训练相符；任何域适配、数据归一化只能在开发数据上拟合。
5. 本阶段交付的是**评估协议和真实可执行代码**，不是已经完成多设备真实数据实验。统计泛化证据需要多个独立设备、足量独立记录、不同工况、模型选择隔离与不确定性分析。
6. R4c 暂不自动搜索超参数，也不假定 InceptionTime / ROCKET 神经网络已接入这条诊断评估接口。

合并条件：GitHub Quality（包括锁定 Ruff、API 检查、Wheel 与 >=83% 测试覆盖率）和 CPU/Gloo DDP 均通过。
