# Forecasting Model Evaluation / 时序预测模型评估

## Scope / 范围

This evaluation validates the leaderboard mechanism on the official ETTh1 hourly dataset from the
[ETDataset repository](https://github.com/zhouhaoyi/ETDataset). It is a framework admission experiment, not a
reproduction of conventional ETT paper tables: windows use a non-overlapping 24-step stride, expanding-window
validation, and a sealed latest-period test.

本评估使用 [ETDataset 官方仓库](https://github.com/zhouhaoyi/ETDataset)的 ETTh1 小时数据验证排行榜机制。
这是框架准入实验，不是常见 ETT 论文表格的原样复现：窗口采用不重叠的 24 步 stride、扩展窗口验证和最新
时间段封存测试。

## Protocol / 协议

- Target / 目标：`OT`
- Historical context / 历史长度：96
- Forecast horizon / 预测长度：24
- Samples / 样本：721
- Development / 开发集：577
- Sealed test / 封存测试：144 windows, `2018-02-02 23:00:00` to `2018-06-25 23:00:00`
- Validation / 验证：3 expanding-window folds
- Neural training / 神经网络训练：up to 50 epochs, inner-time early stopping, AdamW, MSE, gradient clipping

The first execution exposed and rejected an invalid scale comparison: inputs were standardized while forecasting
targets remained on the raw scale. The backend now fits a target scaler only on each inner training split, stores it
with the preprocessor, and inverse-transforms predictions before metrics. A string-timestamp report edge case was also
fixed. These failures demonstrate why a second task domain was necessary for mechanism validation.

首次执行暴露并否决了一个错误尺度比较：输入已标准化，但 forecasting 目标仍是原尺度。后端现在只用每个
inner training split 拟合目标 scaler，将其与预处理器共同保存，并在计算指标前逆变换预测；同时修复了字符
串时间戳报告边界。它们说明引入第二类任务对机制验证是必要的。

## Results / 结果

Results ranked by mean development RMSE / 按开发集 RMSE 均值排名：

| Rank | Pipeline | RMSE mean | RMSE std | MAE | SMAPE |
|---:|---|---:|---:|---:|---:|
| 1 | DLinear | 2.392502 | 0.510729 | 1.806833 | 0.236419 |
| 2 | LastValue | 2.397636 | 0.429710 | 1.741336 | 0.224730 |
| 3 | PatchTST | 2.567530 | 0.421581 | 1.955113 | 0.244418 |
| 4 | TSMixer | 10.653166 | 5.639855 | 9.952792 | 0.684278 |

DLinear was selected and retrained for 34 epochs. Its sealed-test metrics were MAE `1.308023`, RMSE `1.792204`,
R² `0.733094`, and SMAPE `0.224629`. Per-horizon MAE/RMSE were also stored; error generally increased with horizon.

DLinear 最终使用 34 个 epoch 重训，封存测试指标为 MAE `1.308023`、RMSE `1.792204`、R² `0.733094`、
SMAPE `0.224629`。同时保存逐 horizon MAE/RMSE，误差总体随预测距离增加。

## Admission Decision / 准入决定

- `LastValue` remains an AutoML sanity baseline, not a public `nn.Module` model.
- DLinear, PatchTST, and TSMixer already exist in `torch_like`; this experiment validates adapters, not new admission.
- DLinear's gain over LastValue is too small to support a superiority claim.
- PatchTST and TSMixer do not pass a local value gate under this configuration; no new forecasting model is added.
- Before model admission claims, run standard split-compatible benchmarks, multiple seeds, and fair family-level search.

- `LastValue` 保留为 AutoML 合理性基线，不作为公共 `nn.Module` 模型。
- DLinear、PatchTST、TSMixer 已存在于 `torch_like`；本实验验证的是适配器，而不是新增准入。
- DLinear 相对 LastValue 的增益过小，不支持优越性结论。
- PatchTST 和 TSMixer 在当前配置下未通过本地价值门禁，因此不新增 forecasting 模型。
- 在作出模型准入结论前，还需运行兼容标准划分的基准、多随机种子和公平的模型家族搜索。
