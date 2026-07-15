# XJTU-SY Model Evaluation / XJTU-SY 模型评估

## Decision / 结论

中文：截至 2026-07-15，没有发现一个同时具备公开可核验实现、与本项目输入协议一致、并在统一多 bearing 协议下独立复现的单一“XJTU-SY SOTA”模型。因此本轮不强制新增论文模型。框架保留已有 RUL 结构族，新增候选必须先通过协议匹配和本地验证门禁。

English: As of 2026-07-15, no single “XJTU-SY SOTA” candidate was found with a verifiable public implementation, matching input protocol, and independent reproduction under one multi-bearing protocol. No paper model is forced into the framework in this round. Existing RUL families remain, and new candidates must first pass protocol and local-validation gates.

## Why Scores Are Not Directly Comparable / 为什么分数不可直接比较

Published studies vary in train/test bearings, operating conditions, raw-signal versus handcrafted health indicators, censoring assumptions, target normalization, and whether a validation set exists. A smaller reported RMSE is not evidence of a better reusable model when these protocols differ.

论文在训练/测试 bearing、工况、原始信号或人工健康指标、删失假设、标签归一化和验证集设置方面均不一致。协议不同的 RMSE 数值不能直接证明某个模型更适合作为通用框架模型。

## Candidate Review / 候选评审

| Candidate | Evidence and protocol | Framework decision |
|---|---|---|
| MDSCT | End-to-end raw vibration; multiscale depthwise convolution, channel attention, patching, and ProbSparse attention. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11481647/) | Observe. No verified official code was found. A clean-room multiscale-patch composition failed the local validation gate and was removed. |
| Pathformer-TCLSTM | Wavelet denoising, EMD, time/frequency/time-frequency features, TSDAE feature selection, then Pathformer-TCLSTM. [Primary paper](https://papers.phmsociety.org/index.php/ijphm/article/view/4297) | Defer. Most value lies in a dataset-specific preprocessing pipeline rather than a reusable model class. |
| HIWSAN | Cross-condition domain adaptation with health-indicator weighting and subdomain alignment. [Primary paper](https://www.mdpi.com/1424-8220/25/15/4536) | Defer until `domain_adaptation_rul` is an explicit task contract. It is not a drop-in scalar RUL regressor. |
| xLSTM-Transformer RUL | Uses ISOMAP health indicators plus time index and a length-10 feature sequence; published in 2026. [Primary paper](https://www.mdpi.com/1424-8220/26/5/1578) | Observe. The framework already contains xLSTM and Transformer components, but the paper pipeline does not match the current raw-window adapter. |
| RULSurv | Censoring-aware survival formulation; reports CoxPH and Random Survival Forest results. [Primary paper](https://arxiv.org/abs/2405.01614) | Valuable future task, but requires event/censoring labels and survival metrics rather than the current scalar-regression contract. |
| THGNN | Designed for heterogeneous multi-sensor temporal graphs and evaluated on N-CMAPSS. [Primary paper](https://arxiv.org/abs/2405.04336) | Keep as an existing graph-RUL adapter, but do not call it an XJTU-SY SOTA model. Two vibration channels provide weak graph structure. |

## Local Regular Experiment / 本地常规实验

Protocol / 协议:

- Dataset: XJTU-SY condition `35Hz12kN`
- Samples: 5 bearings, 40 acquisitions per bearing, 2 windows per acquisition, 400 windows total
- Input: 1024 points, horizontal and vertical vibration channels
- Label: normalized acquisition-position RUL
- Split: bearing-level group split, 3 train / 1 validation / 1 test, seed 42
- Training: Adam, learning rate 0.001, batch size 32, 100 epochs, CUDA
- Dataset fingerprint: `df0cb277a8f501582fdde3c648a119a78916071a7c1a74d83cb3dc871bb0c213`

Results ranked by validation RMSE / 按验证 RMSE 排名:

| Rank | Model | Parameters | Validation RMSE | Test RMSE |
|---:|---|---:|---:|---:|
| 1 | TCNRUL | 9,025 | 0.1717 | 0.3783 |
| 2 | LogTransRUL | 329,091 | 0.2396 | 0.1812 |
| 3 | LightGBMRegressor, flatten, 500 trees | N/A | 0.2421 | 0.2294 |
| 4 | InformerRUL | 329,091 | 0.2546 | 0.1757 |
| 5 | PatchTSTRUL | 13,249 | 0.2575 | 0.2886 |
| 6 | THGNN | 23,201 | 0.2682 | 0.2404 |
| 7 | TransformerRUL | 41,505 | 0.2839 | 0.1930 |
| 8 | AutoformerRUL | 328,963 | 0.6386 | 0.2945 |

中文：7 个常规模型均完成，无 OOM、NaN、超时或不兼容。TCNRUL 是该验证 bearing 上的首选，但在测试 bearing 上明显退化；测试集排序与验证集排序不一致，说明 bearing 间分布差异很大。这个结果支持“先做多 fold 协议”，不支持宣称 TCN 为 XJTU-SY SOTA。

English: All seven regular trials completed without OOM, NaN, timeout, or incompatibility. TCNRUL wins on the validation bearing but degrades sharply on the test bearing. The validation and test rankings disagree, showing substantial bearing-to-bearing shift. This supports a multi-fold protocol, not a claim that TCN is XJTU-SY SOTA.

Six additional LightGBM pipelines compared basic statistics, vibration statistics, and raw flattening under the same
split and seed. Raw flattening with 500 trees and 15 leaves ranked best for LightGBM. Vibration statistics performed
worse than both flattening and basic statistics, demonstrating why preprocessing must be ranked as part of the complete
pipeline rather than fixed globally.

另有 6 条 LightGBM 管线在相同划分和种子下比较基础统计、振动统计和原始展平。500 棵树、15 leaves 的
原始展平方案最佳。振动统计反而弱于展平和基础统计，说明数据预处理必须作为完整算法管线参与排行榜，
不能在任务开始前武断固定。

## Group-CV And Sealed-Test Experiment / 多 Fold 与封存测试实验

The improved leaderboard protocol reserves `35Hz12kN/Bearing1_2` before candidate execution. The other four
bearings rotate as validation groups. Candidate subprocesses receive only the 320 development samples; the selected
pipeline is then retrained on all development samples and evaluated once on the 80 sealed-test samples.

改进后的排行榜协议在候选执行前封存 `35Hz12kN/Bearing1_2`，其余 4 个 bearing 轮流作为验证组。候选子进程
只接收 320 个 development 样本；选出最佳管线后，才在全部 development 样本上重训，并对 80 个封存测试
样本评估一次。

Configuration / 配置：
[`examples/xjtu_sy_rul_group_cv_benchmark.yaml`](../examples/xjtu_sy_rul_group_cv_benchmark.yaml)

Results ranked by mean validation RMSE / 按验证 RMSE 均值排名：

| Rank | Pipeline | Validation RMSE mean | RMSE std |
|---:|---|---:|---:|
| 1 | LightGBM, flatten, 300 trees | 0.219608 | 0.073763 |
| 2 | LightGBM, flatten, 500 trees | 0.221392 | 0.071789 |
| 3 | LightGBM, basic statistics, 300 trees | 0.226629 | 0.066764 |
| 4 | LightGBM, basic statistics, 500 trees | 0.228207 | 0.069253 |
| 5 | LightGBM, vibration statistics, 300 trees | 0.258742 | 0.075711 |
| 6 | LightGBM, vibration statistics, 500 trees | 0.261177 | 0.077503 |
| 7 | LogTransRUL, 100 epochs | 0.272320 | 0.090175 |
| 8 | TCNRUL, 100 epochs | 0.276964 | 0.109085 |

The selected 300-tree flatten pipeline achieved sealed-test MAE `0.201379`, RMSE `0.234063`, and R² `0.365250`.
All eight candidates completed without OOM, NaN, timeout, or incompatibility. This experiment supports keeping
LightGBM as a strong RUL AutoML baseline; it does not justify adding a new XJTU-SY-specific public model class.

最终选中的 300-tree flatten 管线在封存测试集上取得 MAE `0.201379`、RMSE `0.234063` 和 R² `0.365250`。
8 个候选均未发生 OOM、NaN、超时或不兼容。该结果支持把 LightGBM 保留为强 RUL AutoML 基线，但不足以
支持新增 XJTU-SY 专用公共模型类。

Compared with the earlier single split, the mechanism now prevents per-candidate test inspection, aggregates
bearing-level validation mean/std, and stores the final test separately from the leaderboard. Remaining gaps are
predefined complete fold protocols, nested CV for larger hyperparameter searches, cross-seed statistics, and task
contracts for health indicators, domain adaptation, and censored survival.

与此前单次划分相比，当前机制已阻止逐候选查看测试结果，汇总 bearing 级验证均值和标准差，并把最终测试
与排行榜分离保存。尚未解决的是完整预定义 fold 协议、大规模超参搜索所需的嵌套交叉验证、多随机种子统计，
以及健康指标、域适应和删失生存分析任务合同。

## Neural Training Fairness Recheck / 神经网络公平训练复验

The neural candidates were rerun with inner-bearing early stopping, best-checkpoint restoration, AdamW, Huber loss,
gradient clipping, a plateau scheduler, and a sigmoid unit-interval output. Outer validation bearings remained scoring
only. The final TCN epoch count was the median best epoch from the four folds and was retrained on all development data.

神经网络候选增加 inner-bearing 早停、最佳 checkpoint 恢复、AdamW、Huber、梯度裁剪、Plateau 调度和
sigmoid 单位区间输出后重新运行；outer validation bearings 仍然只负责评分。最终 TCN 使用 4 个 fold 最佳
epoch 的中位数，并在全部 development 数据上重训。

Configuration / 配置：
[`examples/xjtu_sy_rul_neural_fairness.yaml`](../examples/xjtu_sy_rul_neural_fairness.yaml)

| Model | Previous validation RMSE | Fair-training RMSE | Fair-training std |
|---|---:|---:|---:|
| TCNRUL | 0.276964 | 0.255420 | 0.048874 |
| LogTransRUL | 0.272320 | 0.276922 | 0.060736 |

The selected TCN used four final epochs and achieved sealed-test MAE `0.214773`, RMSE `0.247994`, and R² `0.287443`.
Training controls materially reduced TCN fold variance and improved its mean, but it still did not beat the selected
LightGBM sealed-test RMSE `0.234063`. LogTrans variance also fell, while its mean did not improve. The remaining gap is
therefore not solved by adding more epochs or another local-window neural architecture; the next RUL capability should
model acquisition-level degradation sequences.

最终 TCN 使用 4 个 epoch，在封存测试集取得 MAE `0.214773`、RMSE `0.247994` 和 R² `0.287443`。训练控制
显著降低了 TCN 的 fold 方差并改善均值，但仍未超过 LightGBM 的封存测试 RMSE `0.234063`。LogTrans 方差
同样下降，但均值没有改善。因此剩余差距不能靠继续增加 epoch 或另一个局部窗口神经网络解决；下一项 RUL
能力应当建模 acquisition 级退化序列。

## Rejected Experimental Composition / 被否决的实验组合

A non-paper-exact multiscale convolution plus patch Transformer composition was tested before admission. Its stronger configuration used 148,036 parameters and achieved validation RMSE 0.2210 and test RMSE 0.2908. It did not beat the 9,025-parameter TCN on the admission metric, so its public API, registry entry, AutoML mapping, and approval record were removed.

一个非论文原样复现的多尺度卷积加 Patch Transformer 组合在准入前接受了测试。较强配置含 148,036 个参数，验证 RMSE 为 0.2210、测试 RMSE 为 0.2908；它没有在准入指标上超过仅 9,025 参数的 TCN，因此其公共 API、registry、AutoML 映射和审批记录均已撤销。

## Next Evidence Gate / 下一证据门禁

Before adding another XJTU-SY-specific architecture:

在新增下一个 XJTU-SY 专用结构前：

1. Use the implemented `group_cv` and sealed-test protocol as the minimum local gate.
2. Add predefined complete folds when reproducing a published protocol.
3. Separate raw-window regression from health-indicator sequence pipelines.
4. Run identical preprocessing, folds, seeds, budgets, and metrics.
5. Require a meaningful mean-fold gain or a clear capability gain such as censoring or cross-condition adaptation.
6. Keep paper-inspired compositions out of the public registry when they fail the gate.
