# ESA-ADB 异常检测评估 / ESA-ADB Anomaly Detection Evaluation

## 数据集定位 / Dataset Positioning

中文：ESA Anomaly Dataset 是由 ESA、ESOC、Airbus Defence and Space 与 KP Labs 协作整理的真实卫星遥测异常数据集。数据来自三个匿名化任务；官方 ESA-ADB 基准使用 Mission1 和 Mission2，Mission3 因异常数量少、异常较简单且通信缺口较多而未纳入正式基准。数据集许可证为 CC BY 3.0 IGO。

English: The ESA Anomaly Dataset contains real, anonymized spacecraft telemetry curated by ESA, ESOC, Airbus Defence and Space, and KP Labs. The official ESA-ADB benchmark uses Mission1 and Mission2. Mission3 is excluded from the benchmark because its anomalies are fewer and more trivial and its telemetry contains many communication gaps. The dataset is licensed under CC BY 3.0 IGO.

- Dataset: <https://doi.org/10.5281/zenodo.12528696>
- Paper: <https://arxiv.org/abs/2406.17826>
- Official benchmark code: <https://github.com/kplabs-pl/ESA-ADB>

## Mission1 结构 / Mission1 Structure

中文：Mission1 包含 76 个遥测通道，其中 58 个是需要输出异常判断的 target channels，18 个是只提供上下文的 non-target channels。任务还包含 698 种 telecommands。原始通道分别保存为 ZIP 压缩的 pandas Pickle；`labels.csv` 给出事件 ID、通道和闭区间起止时间，`anomaly_types.csv` 给出 Anomaly、Rare Event 和 Communication Gap 等类别。

English: Mission1 contains 76 telemetry channels: 58 target channels evaluated for anomaly detection and 18 non-target context channels. It also contains 698 telecommands. Each raw channel is stored as a ZIP-compressed pandas Pickle. `labels.csv` contains event IDs, channels, and closed timestamp ranges, while `anomaly_types.csv` describes categories such as Anomaly, Rare Event, and Communication Gap.

官方统计 / Official statistics:

| Item | Mission1 |
|---|---:|
| Duration | 14 anonymized years |
| Data points | 774,856,895 |
| Annotated points | 1.80% |
| Annotated events | 200 |
| Anomalies | 118 |
| Rare nominal events | 78 |
| Communication gaps | 4 |

## 官方协议 / Official Protocol

中文：官方把时间线前半作为训练集、后半作为测试集，Mission1 因而具有 84 个月训练数据；训练段最后 3 个月作为验证集。验证和测试都严格晚于训练数据。训练数据并非纯正常数据，异常也会出现在训练和验证中。官方预处理以 30 秒间隔使用 zero-order hold 重采样，并要求阈值、标准化参数和 contamination 等只能在训练/验证阶段确定。

English: The official protocol uses the first half of each mission for training and the second half for testing, yielding 84 training months for Mission1. The last three training months form the validation set. Validation and test samples are strictly later than training samples. Training is not anomaly-free. Official preprocessing uses 30-second zero-order-hold resampling, and thresholds, normalization parameters, and contamination estimates must be determined without future test data.

## torch_like 首轮协议 / First torch_like Protocol

中文：首轮实验使用官方建议的 Mission1 lightweight subset，即 `channel_41` 至 `channel_46`。它们全部属于 subsystem 5、group 8 且都是 target channels。为控制本地资源，首轮仅选择下列时间段，因此结果用于验证框架机制，不能直接与官方 14 年全量结果比较：

English: The first experiment uses the official Mission1 lightweight subset, `channel_41` through `channel_46`. All six are target channels from subsystem 5 and group 8. The following bounded periods are used to fit local resources. Consequently, these results validate the framework mechanism and are not directly comparable with the full 14-year official benchmark.

| Split | Time range |
|---|---|
| Train | 2004-01-01 to 2006-09-30 |
| Validation | 2006-10-01 to 2006-12-31 |
| Sealed test | 2007-01-01 to 2007-12-31 |

- Resampling: 30 seconds, zero-order hold.
- Window: 64 points, approximately 32 minutes.
- Stride: 30 points, approximately 15 minutes.
- Positive categories: `Anomaly` and `Rare Event`, matching the official baseline setting in which existing algorithms do not explicitly distinguish them.
- Threshold: one global threshold selected only on validation scores.
- Ranking: corrected event-wise F0.5, emphasizing precision and false-alarm control.

## 指标边界 / Metric Boundary

中文：`torch_like` 首版在统一重采样网格上计算 point metrics、corrected event-wise F0.5、alarming precision 和 channel event F0.5。它保留 ESA-ADB 的优先级思想，但事件由重采样后的连续标签片段近似恢复，尚未复现官方基于原始事件 ID、纳秒时间范围、subsystem 和 affiliation 的完整评估器。因此报告中的指标必须标记为 ESA-lite approximation。

English: The first implementation computes point metrics, corrected event-wise F0.5, alarming precision, and channel-event F0.5 on a uniformly resampled grid. It follows ESA-ADB's operational priorities, but reconstructs events approximately from contiguous resampled labels and does not yet reproduce the complete official evaluator based on event IDs, nanosecond time ranges, subsystems, and affiliation metrics. Reported values must therefore be labeled as ESA-lite approximations.

## 候选模型 / Candidates

- `GlobalSTD`: transparent statistical baseline.
- `IsolationForest`: per-channel window-feature isolation baseline.
- Registry-discovered `torch_like` models: `AnomalyTransformer`, `DCDetector`, `GDN`, `MEMTO`, `MTADGAT`, `OmniAnomaly`, `RTDetector`, `TranAD`, and `USAD`.

中文：两个传统方法只作为锚点。深度学习候选从 `torch_like` 的 detection registry 自动发现，不再由 ESA 数据适配器硬编码模型名单。

中文：候选模型只接收 train 和 validation。排行榜确定最佳候选后，框架才重新执行该候选并打开 sealed test。测试标签不会参与标准化、模型训练、阈值搜索或模型选择。

English: Candidate trials receive only train and validation data. After ranking, only the selected candidate is rerun against the sealed test. Test labels are not used for normalization, model fitting, threshold search, or model selection.

## 本地结果 / Local Results

运行配置 / Run configuration: [`examples/esa_mission1_anomaly_leaderboard.yaml`](../examples/esa_mission1_anomaly_leaderboard.yaml)

- Resampled rows: 4,207,680.
- Windows: 140,250 = 96,382 train + 8,830 validation + 35,038 sealed test.
- Event IDs in selected channels and ranges: 18 train, 3 validation, 13 test.
- All five candidates completed. The earlier RTDetector smoke overflow was fixed by bounding its learned log-scale before exponentiation.

Validation ranking / 验证集排名:

| Rank | Candidate | Corrected event F0.5 | Point precision | Point recall | Point F1 | False alarm rate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TranAD | 0.499954 | 0.962963 | 0.085526 | 0.157100 | 0.000114 |
| 2 | IsolationForest | 0.227131 | 0.222222 | 0.019737 | 0.036254 | 0.000684 |
| 3 | GlobalSTD | 0.075632 | 0.464286 | 0.042763 | 0.078313 | 0.001709 |
| 4 | USAD | 0.045322 | 0.037037 | 0.003289 | 0.006042 | 0.002962 |
| 5 | RTDetector | 0.002258 | 0.001779 | 0.003289 | 0.002309 | 0.063340 |

Selected TranAD sealed test / 入选 TranAD 封存测试:

| Metric | Value |
|---|---:|
| Corrected event F0.5 | 0.777746 |
| Event precision corrected | 0.874949 |
| Event recall | 0.538462 |
| Point precision | 0.997821 |
| Point recall | 0.524989 |
| Point F1 | 0.687998 |
| False alarm rate | 0.000058 |
| Channel event F0.5 | 0.631579 |
| Average precision | 0.721259 |
| ROC-AUC | 0.889368 |

中文：结果证明排行榜机制已经能在真实大规模异常数据上完成压缩包读取、预定义时间划分、异常分数训练、验证集阈值搜索、失败隔离、验证排名和一次性封存测试。它不证明 TranAD 在完整 ESA-ADB 上达到 SOTA：时间范围被限制，步长为 15 分钟，指标是 uniform-grid 近似，而且只运行一个随机种子和一组轻量超参数。

English: These results demonstrate that the leaderboard mechanism can execute archive ingestion, predefined chronological splitting, anomaly-score training, validation-only threshold search, failure isolation, validation ranking, and one-shot sealed testing on a large real dataset. They do not establish TranAD as SOTA on full ESA-ADB: the time range is bounded, scores are emitted every 15 minutes, metrics are uniform-grid approximations, and only one seed and one lightweight hyperparameter setting were evaluated.

## Registry-driven framework run / Registry 驱动的框架模型实验

中文：第二轮使用同一数据划分，将训练预算提高到 10 epochs，并测试两个锚点和 detection registry 中全部九个模型。11 个候选全部完成，没有 OOM、NaN 或接口不兼容。验证集按校正事件 F0.5 排名前五如下。

English: The second run kept the same data split, increased the budget to 10 epochs, and evaluated two anchors plus all nine models in the detection registry. All 11 candidates completed without OOM, NaN, or interface failures. The top five validation results by corrected event F0.5 are:

| Rank | Candidate | Corrected event F0.5 | Point precision | Point recall | Point F1 | False alarm rate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | DCDetector | 0.357003 | 0.222222 | 0.019737 | 0.036254 | 0.000456 |
| 2 | AnomalyTransformer | 0.277693 | 0.407407 | 0.036184 | 0.066465 | 0.000342 |
| 3 | TranAD | 0.277693 | 0.851852 | 0.075658 | 0.138973 | 0.000342 |
| 4 | IsolationForest | 0.227131 | 0.222222 | 0.019737 | 0.036254 | 0.000684 |
| 5 | MTADGAT | 0.146933 | 0.148148 | 0.013158 | 0.024169 | 0.000911 |

Selected DCDetector sealed test / 入选 DCDetector 封存测试：

| Metric | Value |
|---|---:|
| Corrected event F0.5 | 0.555475 |
| Event precision corrected | 0.624872 |
| Event recall | 0.384615 |
| Point precision | 0.987879 |
| Point recall | 0.485786 |
| Point F1 | 0.651299 |
| False alarm rate | 0.000204 |
| Channel event F0.5 | 0.525210 |
| Average precision | 0.728035 |
| ROC-AUC | 0.901711 |

中文：该结果证明排行榜已经能够优先复用框架模型，但不能据此宣称 DCDetector 是 ESA-ADB 最优模型。当前验证范围只有两个异常事件，单次时间切分对事件级排名过于敏感；下一阶段应优先增加多时间折验证或延长验证期，而不是继续增加模型。

English: This run proves that the leaderboard can prioritize framework-owned models, but it does not establish DCDetector as the best model on ESA-ADB. The validation range contains only two anomaly events, making event-wise ranking highly sensitive to one chronological split. The next methodological priority is multi-fold temporal validation or a longer validation period, not a larger model list.

## CBAM-TCN ablation / CBAM-TCN 消融实验

中文：ESA Mission 1 的“84 个月”描述指完整开发历史，不表示验证集覆盖 84 个月。当前资源受限配置只读取 2004-01 至 2006-09 作为训练，并以 2006-Q4 作为验证。六个选定通道在 Q4 中有三个原始事件 ID：两个 `Anomaly` 和一个 `Rare Event`。窗口化和连续区间合并后，事件指标只有两个有效正事件，因此排名方差很大。

English: ESA Mission 1's “84 months” describes the full development history, not an 84-month validation set. The resource-bounded configuration trains on 2004-01 through 2006-09 and validates on 2006-Q4. The six selected channels contain three raw event IDs in Q4: two `Anomaly` and one `Rare Event`. Protocol v3 preserves short events through overlap-aware resampling and interval aggregation, yielding four effective events because the rare event contains two separated segments.

`CBAMTCNDetector` uses the first `L-1` observations to predict the final observation, preventing direct access to its prediction target. A matched 10-epoch ablation compared the same TCN with and without `CBAM1d`.

| Variant | Parameters | Event F0.5 | False alarm events | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TCN + CBAM1d | 11,458 | 0.004242 | 283 | 0.007467 | 0.540464 |
| TCN without CBAM1d | 10,918 | 0.004227 | 284 | 0.006936 | 0.515726 |

中文：CBAM 对连续异常分数的排序有轻微改善，但没有改变阈值后的事件检测结果，且整体性能显著落后于 `DCDetector`、`AnomalyTransformer` 和 `TranAD`。因此保留 `CBAM1d` 作为可复用组件，并将 `CBAMTCNDetector` 标记为框架适配器；当前证据不支持把它设为 ESA 默认候选或高价值异常检测模型。

English: CBAM slightly improves continuous-score ranking and removes one false-alarm event, but the combined detector remains far behind `DCDetector`, `AnomalyTransformer`, and `TranAD`. `CBAM1d` is therefore retained as a reusable component and `CBAMTCNDetector` as an explicitly selected framework adapter; current evidence does not support promoting it as an ESA default or high-value anomaly detector.

## Current v3 result / 当前 v3 结果

中文：v3 将与原始事件有重叠的每个 30 秒时间桶标为正，并在窗口降采样时保留相邻输出点之间出现的短事件。Q4 的两个短异常不再丢失，验证指标包含四个有效事件。本文前面的 v1/v2 数值仅作为机制演进记录，不再用于当前模型结论。

English: Protocol v3 labels every 30-second bucket that overlaps an event and preserves positives occurring between consecutive stride outputs. The two short Q4 anomalies are no longer lost, producing four effective validation events. Earlier v1/v2 values are historical and must not be used for current model conclusions.

| Rank | Candidate | Validation event F0.5 |
|---:|---|---:|
| 1 | DCDetector | 0.624840 |
| 2 | TranAD | 0.499909 |
| 3 | IsolationForest | 0.499818 |
| 4 | MTADGAT | 0.277581 |
| 5 | AnomalyTransformer | 0.249932 |

Selected `DCDetector` sealed-test corrected event F0.5: `0.714231` (7 TP, 1 FP, 10 FN over 17 effective events). The standalone CBAM-TCN sealed-test event F0.5 is `0.015327`, with 859 false-alarm events.
