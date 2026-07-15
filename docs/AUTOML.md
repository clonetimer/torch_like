# Model Leaderboard Automation / 模型排行榜自动化

## Positioning / 定位

中文：`torch_like.automl` 是模型框架之上的可审计编排层。它负责读取数据、生成防泄漏划分、组合预处理与候选模型、隔离执行 trial，并输出验证集排行榜和可复现产物。它不是 `torch_like` 核心模型 API，也不保证复现论文最优分数。

English: `torch_like.automl` is an auditable orchestration layer over the model framework. It loads data, creates leakage-aware splits, composes preprocessing and candidate models, isolates trials, and exports validation-ranked reproducible artifacts. It is not the core model API and does not promise paper-best reproduction.

## Executable Scope / 可执行范围

- `tabular_classification`
- `tabular_regression`
- `time_series_classification`
- `time_series_forecasting`
- `rul_regression`
- `anomaly_detection`
- `multilabel_classification`
- `multi_target_regression`

中文：`anomaly_detection` 已枚举但尚未接入该编排后端。分类和回归执行样本级标签；forecasting 执行多步未来标签。

English: Classification and regression use sample-level labels; forecasting uses multi-step future targets. `anomaly_detection` supports native ESA-ADB archives with predefined chronological splits, validation-only threshold selection, and a sealed final test.

## ESA-ADB Contract / ESA-ADB 协议

中文：`format: esa_adb` 可以直接读取 ESA 原始 Mission ZIP。配置必须显式指定 target channels、重采样间隔、窗口长度、步长以及互不重叠的 train/val/test 时间范围。候选 trial 只接收 train 和 validation；阈值仅在 validation 标签上选择；最终选择完成后才执行一次 sealed test。

English: `format: esa_adb` reads a raw ESA mission ZIP directly. The configuration explicitly selects target channels, resampling interval, window length, stride, and non-overlapping train/validation/test ranges. Candidate trials receive train and validation only. Thresholds are selected on validation labels, and the sealed test runs once after selection.

Candidate discovery is registry-driven. The leaderboard adds the `GlobalSTD` and `IsolationForest` anchors, then discovers every default-eligible model in the `detection` registry. The current default framework pool contains `AnomalyTransformer`, `DCDetector`, `GDN`, `MEMTO`, `MTADGAT`, `OmniAnomaly`, `RTDetector`, `TranAD`, and `USAD`. Experimental adapters such as `CBAMTCNDetector` remain explicitly selectable through `search.algorithms` but do not enter unrelated searches automatically. One trial is allocated to every requested algorithm before preprocessing variants consume the remaining budget. See [`examples/esa_mission1_anomaly_leaderboard.yaml`](../examples/esa_mission1_anomaly_leaderboard.yaml) and [`ESA_ADB_MODEL_EVALUATION.md`](ESA_ADB_MODEL_EVALUATION.md).

中文：异常检测排行榜优先发现并测试 `torch_like` 的 detection registry，而不是默认寻找外部数据集专用模型。基础统计方法只作为性能锚点。新增 detection 模型如果没有统一 `DetectionModel` 契约或没有经过审计的 AutoML 默认配置，候选发现会立即失败，避免模型被静默遗漏。

PyTorch candidates / PyTorch 候选:

- Time-series classification / 时序分类: `InceptionTime`, `ROCKET`
- RUL regression / 寿命回归: `THGNN`, `InformerRUL`, `AutoformerRUL`, `LogTransRUL`, `TCNRUL`, `PatchTSTRUL`, `TransformerRUL`
- Forecasting / 时序预测: `DLinear`, `PatchTST`, `TSMixer`, plus the zero-parameter `LastValue` baseline / 以及零参数 `LastValue` 基线

Scikit-learn families / Scikit-learn 算法族:

- Classification / 分类: logistic regression, random forest, extra trees, histogram gradient boosting
- Regression / 回归: ridge, random forest, extra trees, histogram gradient boosting
- Optional gradient boosting / 可选梯度提升: LightGBM classification and regression

Install the optional backend / 安装可选后端:

```powershell
pip install "torch_like[lightgbm]"
```

中文：时序 sklearn 管线支持 `statistics`、`vibration_statistics` 和等长序列的 `flatten`。`vibration_statistics` 提取通用时域与频域特征，不属于模型类本身。

English: Sequential sklearn pipelines support `statistics`, `vibration_statistics`, and `flatten` for equal-length sequences. `vibration_statistics` extracts generic time- and frequency-domain features and is not part of a model class.

## Data Contracts / 数据协议

CSV row mode / CSV 行模式:

```csv
sample_id,feature_a,feature_b,label
s001,0.1,1.2,0
s002,0.4,0.8,1
```

CSV long time-series mode / CSV 长表时序模式:

```csv
entity,timestamp,sensor_a,sensor_b,rul
bearing-01,1,0.12,0.31,100
bearing-01,2,0.15,0.35,99
```

中文：`sample_mode` 支持 `row`、`full_sequence` 和 `sliding_window`。序列模式需要 `entity_column`，滑窗还需要 `sequence_length` 与 `stride`。重叠窗口禁止随机或分层拆分，避免同一原始行进入多个集合。

English: `sample_mode` accepts `row`, `full_sequence`, and `sliding_window`. Sequence modes require `entity_column`; windows also require `sequence_length` and `stride`. Overlapping windows cannot use random or stratified splits because raw rows would leak across sets.

External labels / 外部标签:

```yaml
dataset:
  path: data.csv
  labels_path: labels.csv
  label_join_key: sample_id
  label_columns: [health_level]
```

中文：外部标签按唯一连接键执行 `many_to_one` 校验，不按行号静默拼接。

English: External labels are validated with a unique many-to-one join key and are never silently joined by row position.

## Forecasting Contract / 时序预测协议

Forecasting windows contain a historical context and a non-overlapping multi-step target:

forecasting 窗口由历史上下文和不重叠的多步未来目标组成：

```yaml
dataset:
  path: D:/data/ETTh1.csv
  label_columns: [OT]
  time_column: date
  sample_mode: sliding_window
  sequence_length: 96
  prediction_length: 24
  stride: 24
split:
  type: rolling_origin
  rolling_folds: 3
```

中文：目标列会自动保留在历史输入中，也可以通过 `feature_columns` 加入外生变量。v1 要求 `stride >= prediction_length`，避免两个集合共享未来标签。`rolling_origin` 使用 expanding-window folds 排名，最新时间段在候选选择完成前保持封存。当前版本只支持单条时间序列；多实体 panel forecasting 需要后续单独协议。

English: Target columns remain available in the historical input, and `feature_columns` may add exogenous variables. v1 requires `stride >= prediction_length` so future targets cannot overlap across sets. `rolling_origin` ranks expanding-window folds while the latest period stays sealed until selection finishes. The current version supports one series; multi-entity panel forecasting requires a separate protocol.

Overall MAE, RMSE, R², SMAPE, and per-horizon MAE/RMSE are exported. See
[`examples/forecasting_rolling_leaderboard.yaml`](../examples/forecasting_rolling_leaderboard.yaml).

输出整体 MAE、RMSE、R²、SMAPE 及逐 horizon MAE/RMSE。配置示例见
[`examples/forecasting_rolling_leaderboard.yaml`](../examples/forecasting_rolling_leaderboard.yaml)。

## XJTU-SY Contract / XJTU-SY 协议

`format: xjtu_sy` directly reads `condition/BearingX_Y/N.csv`. Each acquisition contains horizontal and vertical vibration signals. The normalized target is generated as:

`format: xjtu_sy` 直接读取 `工况/BearingX_Y/N.csv`。每次采样包含水平和垂直振动信号，归一化标签按下式生成：

```text
rul = (lifetime_files - file_index) / lifetime_files
```

```yaml
dataset:
  path: E:/Data/XJTU-SY_Bearing_Datasets4/Data/XJTU-SY_Bearing_Datasets
  format: xjtu_sy
  label_columns: [rul]
  sample_mode: sliding_window
  sequence_length: 1024
  stride: 1024
  condition: 35Hz12kN
  max_files_per_entity: 40
  max_windows_per_file: 2
split:
  type: group_cv
  cv_folds: 4
  # Optional: use a reproducible sealed group instead of seeded selection.
  # test_groups: [35Hz12kN/Bearing1_5]
```

中文：文件和窗口限制采用全生命周期内的确定性均匀抽样，不改变 RUL 标签分母。XJTU-SY 强制按 bearing 使用 `group` 或 `group_cv`。推荐 `group_cv`：先封存测试 bearing，再在其余 bearing 上轮换验证；候选进程不会收到封存测试数据。`test_groups` 使用完整组名，例如 `35Hz12kN/Bearing1_5`。

English: File and window limits use deterministic even sampling over the full lifetime and do not change the RUL denominator. XJTU-SY requires bearing-level `group` or `group_cv` splitting. `group_cv` is recommended: test bearings are sealed first and validation rotates over the remaining bearings. Candidate processes never receive sealed-test data. `test_groups` uses complete group names such as `35Hz12kN/Bearing1_5`.

## Smoke And Regular Runs / 冒烟与常规实验

中文：1 epoch 冒烟实验只检查接口、显存、NaN、超时和兼容性。冒烟通过后必须创建新的不可变任务，恢复常规 epoch、batch size、学习率和数据规模；不得把冒烟排名当作模型质量结论。

English: A one-epoch smoke run checks API, memory, NaN, timeout, and compatibility only. After it passes, create a new immutable task with regular epochs, batch size, learning rate, and data scale. Never treat smoke ranking as model-quality evidence.

推荐流程 / Recommended flow:

1. Set `torch_epochs: 1` and a small bounded dataset for smoke validation.
2. Change `task_id`, restore the regular hyperparameters, and execute `run` again.
3. Rank only by validation-fold mean metrics and inspect fold standard deviations.
4. Expose the sealed test set once, after the best pipeline is selected and retrained.

PyTorch training controls / PyTorch 训练控制：

- `torch_optimizer`: `adam` or `adamw`
- `torch_weight_decay`, `torch_grad_clip_norm`
- `torch_loss`: `mse`, `mae`, or `huber`
- `torch_lr_scheduler`: `none`, `cosine`, or `plateau`
- `torch_early_stopping_patience`, `torch_min_delta`, `torch_restore_best`
- `torch_target_constraint: unit_interval` for normalized RUL / 用于归一化 RUL

中文：每个 PyTorch fold 保存 `training_history.json`、`training_summary.json`、最后 checkpoint 和最佳 checkpoint。group-CV 的早停验证组来自 outer-train bearings；rolling-origin 的早停区间来自 outer-train 末端。outer validation 始终只用于排行榜评分。

English: Every PyTorch fold stores training history, a summary, and last/best checkpoints. Early stopping uses an inner group from outer-train bearings for group CV, or the tail of outer-train for rolling origin. Outer validation is used only for leaderboard scoring.

## Configuration / 配置

See [`examples/xjtu_sy_rul_leaderboard.yaml`](../examples/xjtu_sy_rul_leaderboard.yaml) for a regular XJTU-SY task.

常规 XJTU-SY 任务见 [`examples/xjtu_sy_rul_leaderboard.yaml`](../examples/xjtu_sy_rul_leaderboard.yaml)。

The optional LightGBM comparison is configured separately in
[`examples/xjtu_sy_rul_lightgbm.yaml`](../examples/xjtu_sy_rul_lightgbm.yaml).

可选 LightGBM 对比任务单独配置在
[`examples/xjtu_sy_rul_lightgbm.yaml`](../examples/xjtu_sy_rul_lightgbm.yaml)。

The mixed LightGBM/TCN/LogTrans group-CV benchmark is configured in
[`examples/xjtu_sy_rul_group_cv_benchmark.yaml`](../examples/xjtu_sy_rul_group_cv_benchmark.yaml).

LightGBM、TCN 与 LogTrans 共同参与的 group-CV 基准配置见
[`examples/xjtu_sy_rul_group_cv_benchmark.yaml`](../examples/xjtu_sy_rul_group_cv_benchmark.yaml)。

Resource devices / 计算设备:

- `cpu`: force CPU / 强制 CPU
- `cuda` or `cuda:N`: require the selected CUDA device / 要求指定 CUDA 设备
- `auto`: use CUDA when available, otherwise CPU / CUDA 可用时使用 CUDA，否则使用 CPU

中文：所有候选共享任务级随机种子，trial 配置、checkpoint、manifest 和环境文件都会记录种子。当前调度器一次运行一个独立 trial。

English: All candidates share the task-level random seed. Trial configs, checkpoints, manifests, and environment files record it. The current scheduler runs one isolated trial at a time.

## CLI / 命令行

```powershell
torch-like validate --config task.yaml
torch-like plan --config task.yaml
torch-like run --config task.yaml
torch-like show runs/bearing-rul-001
```

## Python API / Python 接口

```python
from torch_like.automl import LeaderboardTask, load_config

task = LeaderboardTask(load_config("task.yaml"))
task.validate()
print(task.profile_dataset())
print(task.build_plan())

result = task.run()
print(result.leaderboard)
print(result.best_trial)
```

## Artifacts / 产物

```text
runs/{task_id}/
|-- config.json
|-- dataset_profile.json
|-- data_split.json
|-- environment.json
|-- plan.json
|-- leaderboard.csv
|-- trials.json
|-- sealed_test_metrics.json  # group_cv only
|-- final_report.md
|-- best_model/
`-- trials/{trial_id}/
    |-- config.json
    |-- fold_metrics.json     # group_cv only
    `-- fold-{index}/
        |-- validation_predictions.csv
        |-- artifact_manifest.json
        |-- pipeline.joblib
        `-- last_checkpoint.pt
```

中文：数据指纹覆盖数据内容与抽样配置。PyTorch checkpoint 保存 `state_dict`、优化器、epoch 和 RNG 状态；sklearn/joblib 产物只允许作为可信本地产物加载，因为 pickle/joblib 不是安全交换格式。

English: The data fingerprint covers content and sampling configuration. PyTorch checkpoints contain state dictionaries, optimizer, epoch, and RNG state. Sklearn/joblib artifacts are trusted-local only because pickle/joblib is not a safe interchange format.

## Failure Isolation / 失败隔离

OOM、NaN、超时和一般异常会标记当前 trial，而不会终止整个候选搜索。运行目录非空时任务拒绝覆盖，以保护审计链。

OOM, NaN, timeout, and general exceptions mark the current trial without terminating the whole search. A task refuses to overwrite a non-empty run directory, preserving the audit trail.

## Current Limits / 当前限制

- One trial process at a time / 一次执行一个 trial 进程
- No whole-task pause or resume / 不支持整个任务暂停恢复
- Checkpoints do not yet resume a whole leaderboard automatically / checkpoint 尚不能自动恢复整个排行榜
- Pretraining and parameter-count pruning are declared but not executable / 预训练和参数量剪枝已声明但尚不可执行
- No arbitrary source-code mutation / 不允许任意修改模型源码
- Legacy `group` evaluates every candidate on its test split; use `group_cv` for a physically sealed final test / 兼容模式 `group` 会评估每个候选的测试集；需要物理封存时使用 `group_cv`

## Mechanism Maturity / 机制成熟度

Completed foundation / 已完成基础闭环:

- Native CSV and XJTU-SY ingestion with content fingerprints / CSV 与 XJTU-SY 原生读取和内容指纹
- Multi-step forecasting windows and rolling-origin sealed tests / 多步预测窗口和 rolling-origin 封存测试
- Deterministic sampling, group splitting, and group-CV aggregation / 确定性抽样、分组划分和 group-CV 聚合
- Competing preprocessing, sequence features, sklearn, LightGBM, and PyTorch candidates / 预处理、序列特征、sklearn、LightGBM 和 PyTorch 候选共同竞争
- Fixed task seed, isolated failures, CPU/CUDA selection, and dependency manifests / 固定任务种子、失败隔离、CPU/CUDA 选择和依赖清单
- Validation mean/std ranking plus one-shot sealed-test evaluation / 验证均值与标准差排名，以及一次性封存测试评估
- Auditable PyTorch histories, inner validation, early stopping, and best checkpoints / 可审计的 PyTorch 历史、内层验证、早停和最佳 checkpoint
- Explicit smoke-to-regular workflow / 明确的冒烟转常规实验流程

Required before publication-grade ranking / 论文级排行榜前必须补齐:

1. Predefined complete fold assignments for reproducing published protocols.
2. Nested cross-validation when hyperparameter searches require unbiased uncertainty estimates.
3. Separate raw-window, health-indicator sequence, cross-condition adaptation, and survival task contracts.
4. Cross-seed aggregation and statistical model-comparison tests.
5. Multi-series panel forecasting and probabilistic forecast ranking.

1. 支持完整的预定义 fold 分配，以复现论文协议。
2. 超参数搜索需要无偏不确定性估计时，增加嵌套交叉验证。
3. 分离原始窗口、健康指标序列、跨工况适应和生存分析任务协议。
4. 增加多随机种子汇总和统计模型比较检验。
5. 增加多序列 panel forecasting 和概率预测排行榜。

XJTU-SY literature and local protocol analysis are recorded in [`XJTU_SY_MODEL_EVALUATION.md`](XJTU_SY_MODEL_EVALUATION.md).

XJTU-SY 文献与本地协议评估见 [`XJTU_SY_MODEL_EVALUATION.md`](XJTU_SY_MODEL_EVALUATION.md)。

ETTh1 rolling-origin mechanism validation is recorded in
[`FORECASTING_MODEL_EVALUATION.md`](FORECASTING_MODEL_EVALUATION.md).

ETTh1 rolling-origin 机制验证见
[`FORECASTING_MODEL_EVALUATION.md`](FORECASTING_MODEL_EVALUATION.md)。
