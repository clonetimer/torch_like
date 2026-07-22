# Model Leaderboard Automation / 模型排行榜自动化

## Positioning / 定位

中文：`torch_like.automl` 是模型框架之上的可审计编排层。它负责读取数据、生成防泄漏划分、组合预处理与候选模型、隔离执行 trial，并输出验证集排行榜和可复现产物。它不是 `torch_like` 核心模型 API，也不保证复现论文最优分数。

English: `torch_like.automl` is an auditable orchestration layer over the model framework. It loads data, creates leakage-aware splits, composes preprocessing and candidate models, isolates trials, and exports validation-ranked reproducible artifacts. It is not the core model API and does not promise paper-best reproduction.

## Protocol Trust Audit / 协议可信审计

中文：每次 `build_plan()` 和 `run()` 都会在训练前调用 `audit_split_protocol(...)`。审计检查索引范围与重复、集合完整性、train/validation/test 重叠、group 隔离、rolling-origin 时间顺序、fold 边界和 sealed-test 隔离。失败会直接终止任务；通过结果同时进入 `plan.json` 和独立的 `protocol_audit.json`。

English: Every `build_plan()` and `run()` calls `audit_split_protocol(...)` before training. It checks index bounds and duplicates, partition completeness, train/validation/test overlap, group isolation, rolling-origin chronology, fold boundaries, and sealed-test isolation. Failures stop the task; passing results are stored in both `plan.json` and `protocol_audit.json`.

```python
from torch_like.automl import audit_split_protocol

audit = audit_split_protocol(data, split, split_spec, task_type)
assert audit.status == "passed"
```

## Evaluation Trust Levels / 评估信任等级

中文：`trust.level` 声明排行榜结果的可信等级。默认 `development` 面向快速 smoke、接口验证和模型接入，允许单种子运行，但会在 `evaluation_trust_audit.json` 中记录 `single_seed_development_run` 警告。`validated` 面向可比较实验，要求至少 3 个 `search.evaluation_seeds`，并审计每个候选管线是否获得完整种子覆盖、唯一 trial id、统一单 trial 超时和统一 PyTorch 训练预算。

English: `trust.level` declares the trust level of a leaderboard result. The default `development` level is for smoke tests, API checks, and model integration; it allows a single seed and records a `single_seed_development_run` warning in `evaluation_trust_audit.json`. The `validated` level is for comparable experiments and requires at least three `search.evaluation_seeds`, complete seed coverage for every candidate pipeline, unique trial ids, a uniform timeout per trial, and a uniform PyTorch training budget.

```yaml
trust:
  level: validated
search:
  evaluation_seeds: [11, 29, 47]
```

中文：真实任务结果的 smoke/development/validated 分层和晋升规则见 [`AUTOML_BENCHMARK_TRUST.md`](AUTOML_BENCHMARK_TRUST.md)。验证级 rolling-origin 预测配置见 [`examples/forecasting_rolling_validated.yaml`](../examples/forecasting_rolling_validated.yaml)。

English: Real-task smoke/development/validated tiers and promotion rules are documented in [`AUTOML_BENCHMARK_TRUST.md`](AUTOML_BENCHMARK_TRUST.md). A validated rolling-origin forecasting configuration is available at [`examples/forecasting_rolling_validated.yaml`](../examples/forecasting_rolling_validated.yaml).

## Executable Scope / 可执行范围

- `tabular_classification`
- `tabular_regression`
- `time_series_classification`
- `time_series_forecasting`
- `rul_regression`
- `anomaly_detection`
- `multilabel_classification`
- `multi_target_regression`

中文：分类和回归执行样本级标签，forecasting 执行多步未来标签。`anomaly_detection` 已支持 ESA-ADB 原始归档、预定义时间划分、仅验证集选阈值和一次性封存测试。

English: Classification and regression use sample-level labels; forecasting uses multi-step future targets. `anomaly_detection` supports native ESA-ADB archives with predefined chronological splits, validation-only threshold selection, and a sealed final test.

## ESA-ADB Contract / ESA-ADB 协议

中文：`format: esa_adb` 可以直接读取 ESA 原始 Mission ZIP。配置必须显式指定 target channels、重采样间隔、窗口长度、步长以及互不重叠的 train/val/test 时间范围。候选 trial 只接收 train 和 validation；阈值仅在 validation 标签上选择；最终选择完成后才执行一次 sealed test。

English: `format: esa_adb` reads a raw ESA mission ZIP directly. The configuration explicitly selects target channels, resampling interval, window length, stride, and non-overlapping train/validation/test ranges. Candidate trials receive train and validation only. Thresholds are selected on validation labels, and the sealed test runs once after selection.

Candidate discovery is registry-driven. The leaderboard adds the `GlobalSTD` and `IsolationForest` anchors, then discovers every default-eligible model in the `detection` registry. The current default framework pool contains `AnomalyTransformer`, `DCDetector`, `GDN`, `MEMTO`, `MTADGAT`, `OmniAnomaly`, `RTDetector`, `TranAD`, and `USAD`. Experimental adapters such as `CBAMTCNDetector` remain explicitly selectable through `search.algorithms` but do not enter unrelated searches automatically. One trial is allocated to every requested algorithm before preprocessing variants consume the remaining budget. See [`examples/esa_mission1_anomaly_leaderboard.yaml`](../examples/esa_mission1_anomaly_leaderboard.yaml) and [`ESA_ADB_MODEL_EVALUATION.md`](ESA_ADB_MODEL_EVALUATION.md).

中文：异常检测排行榜优先发现并测试 `torch_like` 的 detection registry，而不是默认寻找外部数据集专用模型。基础统计方法只作为性能锚点。新增 detection 模型如果没有统一 `DetectionModel` 契约或没有经过审计的 AutoML 默认配置，候选发现会立即失败，避免模型被静默遗漏。

PyTorch candidates / PyTorch 候选:

- Tabular classification and multilabel classification / 表格分类与多标签分类: `FTTransformer`, `TabNet`, `NODE`, `SAINT`
- Time-series classification / 时序分类: `InceptionTime`, `ROCKET`
- RUL regression / 寿命回归: `THGNN`, `InformerRUL`, `AutoformerRUL`, `LogTransRUL`, `TCNRUL`, `PatchTSTRUL`, `TransformerRUL`
- Forecasting / 时序预测: `DLinear`, `PatchTST`, `TSMixer`, plus the zero-parameter `LastValue` baseline / 以及零参数 `LastValue` 基线

Scikit-learn families / Scikit-learn 算法族:

- Classification / 分类: logistic regression, random forest, extra trees, histogram gradient boosting
- Regression / 回归: ridge, random forest, extra trees, histogram gradient boosting
- Optional gradient boosting / 可选梯度提升: LightGBM classification and regression

中文：多标签分类使用每个标签一列的 0/1 目标，例如 `label_a,label_b,label_c`。sklearn 多输出包装和 PyTorch 表格候选会共同进入候选池；PyTorch 候选使用 sigmoid + binary cross entropy，`load_best()` 输出同形状的 0/1 标签矩阵。为避免 Windows 隔离 trial 中出现嵌套 joblib/loky 进程池，多输出包装器内部固定为单线程，外层并发仍由排行榜调度器控制。

English: Multilabel classification uses one 0/1 target column per label, for example `label_a,label_b,label_c`. Sklearn multi-output wrappers and PyTorch tabular candidates can compete in the same pool. PyTorch candidates use sigmoid plus binary cross entropy, and `load_best()` returns a same-shaped 0/1 label matrix. To avoid nested joblib/loky process pools inside isolated Windows trials, multi-output wrappers run single-threaded internally while outer scheduling remains controlled by the leaderboard runner.

## Non-graph Classification / 非图分类

中文：`tabular_classification` 和 `time_series_classification` 都是普通监督分类，不是图分类。表格分类接收形如 `[samples, features]` 的二维特征；时序分类接收由 `full_sequence` 或 `sliding_window` 生成的 `[samples, time, features]` 序列。两者都不需要 adjacency、edge index、节点标签或图标签。可复用配置见 [`examples/tabular_classification_leaderboard.yaml`](../examples/tabular_classification_leaderboard.yaml) 和 [`examples/time_series_classification_leaderboard.yaml`](../examples/time_series_classification_leaderboard.yaml)。

English: `tabular_classification` and `time_series_classification` are ordinary supervised-learning tasks, not graph classification. Tabular classification consumes `[samples, features]`; time-series classification consumes `[samples, time, features]` sequences produced by `full_sequence` or `sliding_window`. Neither task requires adjacency matrices, edge indices, node labels, or graph labels. See [`examples/tabular_classification_leaderboard.yaml`](../examples/tabular_classification_leaderboard.yaml) and [`examples/time_series_classification_leaderboard.yaml`](../examples/time_series_classification_leaderboard.yaml).

中文：FordA 示例使用 UCR/TSML 的官方 train/test `.ts` 文件转换成长表 CSV。转换产物建议放在 `runs/data_cache/forda/forda_long.csv`，并通过 `split_column: split` 保留 train/val/test 的预定义边界。

English: The FordA example expects the official UCR/TSML train/test `.ts` files converted into long-form CSV. Keep the converted artifact under `runs/data_cache/forda/forda_long.csv` and preserve train/validation/test boundaries through `split_column: split`.

中文：当前表格神经网络适配支持连续数值特征。`TabTransformer` 要求显式类别特征及每列基数，因此在 CSV 协议增加连续列/类别列声明前不会进入 AutoML 候选池；这不影响直接通过 `torch_like.models.tabular.TabTransformer` 调用模型。

English: The current tabular neural adapters support continuous numeric features. `TabTransformer` requires explicit categorical columns and per-column cardinalities, so it remains outside the AutoML pool until the CSV contract distinguishes continuous and categorical columns. It remains directly available through `torch_like.models.tabular.TabTransformer`.

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

For formal local comparisons, use the validated template
[`examples/forecasting_rolling_validated.yaml`](../examples/forecasting_rolling_validated.yaml),
which adds `trust.level: validated` and three evaluation seeds under the same rolling-origin protocol.

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

Multi-seed evaluation / 多随机种子评估:

```yaml
search:
  max_trials: 8
  evaluation_seeds: [11, 29, 47]
```

中文：`max_trials` 表示候选管线配置数，不是进程总数。配置多个 `evaluation_seeds` 后，每条管线为每个种子创建独立 trial，因此计划执行数为“候选管线数 × 种子数”。只有全部要求种子成功的管线才参与排名。排行榜输出指标均值、`*_seed_std`、逐种子名次、平均名次、名次标准差和最好/最差名次。评估种子只改变估计器或训练过程的随机性，任务数据划分保持固定，因此这些统计量不表示不同数据划分之间的不确定性。最终封存测试仍只执行一次，并使用列表中的第一个种子重训。

English: `max_trials` counts candidate pipeline configurations, not total processes. With multiple `evaluation_seeds`, every pipeline receives one isolated trial per seed, so planned executions equal pipelines times seeds. A pipeline is ranked only after every required seed succeeds. The leaderboard exports metric means, `*_seed_std`, per-seed ranks, mean/std rank, and best/worst rank. Evaluation seeds change estimator/training randomness while the task split remains fixed; they do not estimate uncertainty across alternative data splits. The sealed test still runs once, retraining with the first listed seed.

中文：若同时声明 `trust.level: validated`，多种子不再只是可选统计项，而是硬性协议要求；种子数不足或候选种子覆盖不完整会在训练前失败。

English: With `trust.level: validated`, multi-seed evaluation becomes a hard protocol requirement instead of an optional statistic. Too few seeds or incomplete candidate seed coverage fails before training.

Candidate policy / 候选策略:

```yaml
search:
  candidate_policy: production
```

| Value / 取值 | Purpose / 用途 | Budget behavior / 预算行为 |
|---|---|---|
| `production` | Find the strongest deployable pipeline / 寻找实际最优管线 | All compatible families compete / 所有兼容模型族共同竞争 |
| `torch_like_only` | Validate framework models / 验证框架模型 | Only `torch_like` PyTorch adapters / 只保留 `torch_like` PyTorch 适配器 |
| `baseline_only` | Build a fast reference / 快速建立基线 | Only sklearn, LightGBM, and deterministic baselines / 只保留 sklearn、LightGBM 和确定性基线 |
| `prefer_torch_like` | Spend a constrained budget mainly on framework models / 在有限预算下优先测试框架模型 | Torch-like candidates and variants are selected first; at least one baseline anchor is kept when the budget permits / 优先选择框架模型及其变体，预算允许时至少保留一个基线锚点 |

中文：策略作用于候选规划，不改变指标、数据划分或密封测试协议。显式 `algorithms` 与 `torch_like_only`/`baseline_only` 冲突时任务直接失败，不会静默丢弃算法。最终密封测试始终由总榜冠军触发；族内榜只用于解释框架模型与传统基线各自的相对表现。

English: The policy changes candidate planning, not metrics, splits, or the sealed-test protocol. Explicit algorithms that conflict with `torch_like_only` or `baseline_only` fail validation instead of being silently discarded. The overall winner alone triggers final sealed testing; family leaderboards are diagnostic views.

## CLI / 命令行

```powershell
torch-like validate --config task.yaml
torch-like plan --config task.yaml
torch-like run --config task.yaml
torch-like run --config task.yaml --resume
torch-like show runs/bearing-rul-001
torch-like show runs/bearing-rul-001 --family torch_like
torch-like show runs/bearing-rul-001 --family baseline
```

## Python API / Python 接口

```python
from torch_like.automl import LeaderboardTask, load_best, load_config

task = LeaderboardTask(load_config("task.yaml"))
task.validate()
print(task.profile_dataset())
print(task.build_plan())

result = task.run()
print(result.leaderboard)
print(result.best_trial)

# Continue an interrupted immutable task.
result = task.run(resume=True)

# Trusted-local inference; x follows the same prepared-array contract as training.
bundle = load_best(result.run_dir, device="cpu")
predictions = bundle.predict(x)
# Anomaly bundles additionally expose continuous scores with bundle.score(x).
```

中文：`load_best()` 校验产物哈希并恢复入选管线、预处理器和 PyTorch `state_dict`，要求配置 `artifacts.save_checkpoints: true`。当前推理输入是与训练后端一致的已样本化 NumPy 数组；CSV、XJTU-SY 和 ESA-ADB 原始数据仍应先通过 `prepare_dataset()`，避免把数据读取职责混入模型产物。joblib 和 PyTorch checkpoint 都只能加载可信本地产物。

English: `load_best()` verifies artifact hashes and restores the selected pipeline, preprocessors, and PyTorch state dictionary. It requires `artifacts.save_checkpoints: true`. Inference currently accepts the same prepared NumPy-array contract used by training. Raw CSV, XJTU-SY, and ESA-ADB inputs should still pass through `prepare_dataset()`. Joblib and PyTorch checkpoints are trusted-local artifacts only.

## Candidate Registry / 候选注册表

中文：非异常任务候选的任务兼容性、后端和审计搜索默认值通过 `CandidateSpec` 注册；异常模型继续由 detection model registry 与审计默认配置联合发现。新增适配器不需要修改排行榜编排代码，但仍必须提供对应 sklearn 或 PyTorch 执行适配。随机和网格搜索会先为每个显式指定算法保留一个 trial，再把剩余预算分配给预处理与超参数变体。截断网格会优先覆盖尚未出现的预处理方案和覆盖较少的“算法 + 预处理”组合，避免直接继承声明顺序偏置；预算不足以覆盖全部显式算法时会直接拒绝任务。

English: Non-anomaly task compatibility, backend metadata, and audited search defaults are registered through `CandidateSpec`; anomaly candidates remain jointly discovered from the detection model registry and audited defaults. Adding an adapter no longer changes leaderboard orchestration, but it still requires a corresponding sklearn or PyTorch execution adapter. Grid and random search reserve one trial per explicitly requested algorithm before allocating the remaining budget to preprocessing and hyperparameter variants. Truncated grid search greedily covers unseen preprocessing choices and underrepresented algorithm/preprocessing pairs instead of inheriting declaration-order bias. Insufficient explicit coverage fails validation instead of silently omitting a model.

## Artifacts / 产物

```text
runs/{task_id}/
|-- config.json
|-- dataset_profile.json
|-- data_split.json
|-- environment.json
|-- plan.json
|-- leaderboard.csv
|-- leaderboard_torch_like.csv
|-- leaderboard_baselines.csv
|-- trials.json
|-- sealed_test_metrics.json  # every executable split protocol
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

中文：单种子任务保持 `trial-0000` 命名；多种子任务使用 `trial-0000-seed-0000000011`。`trials.json` 保留每个种子的独立结果，`leaderboard.csv` 每条候选管线只保留一行聚合结果。

English: Single-seed tasks retain names such as `trial-0000`; multi-seed tasks use names such as `trial-0000-seed-0000000011`. `trials.json` preserves independent seed-level outcomes while `leaderboard.csv` contains one aggregate row per candidate pipeline. The two family CSV files preserve overall rank and add an independently computed family rank.

中文：`leaderboard_torch_like.csv` 与 `leaderboard_baselines.csv` 保留总榜名次，并分别增加独立计算的族内名次。它们不会产生额外训练，也不会各自解封测试集。

中文：数据指纹覆盖数据内容与抽样配置。PyTorch checkpoint 保存 `state_dict`、优化器、epoch 和 RNG 状态；sklearn/joblib 产物只允许作为可信本地产物加载，因为 pickle/joblib 不是安全交换格式。

English: The data fingerprint covers content and sampling configuration. PyTorch checkpoints contain state dictionaries, optimizer, epoch, and RNG state. Sklearn/joblib artifacts are trusted-local only because pickle/joblib is not a safe interchange format.

## Failure Isolation / 失败隔离

OOM、NaN、超时和一般异常会标记当前 trial，而不会终止整个候选搜索。运行目录非空时任务拒绝覆盖，以保护审计链。

OOM, NaN, timeout, and general exceptions mark the current trial without terminating the whole search. A task refuses to overwrite a non-empty run directory, preserving the audit trail.

## Trial Resume / Trial 断点续跑

中文：`run(resume=True)` 或 CLI `--resume` 会核验保存的配置、数据画像与指纹、数据划分和候选计划。只有这些不可变状态完全一致时才允许续跑。已有 `metrics.json` 或 `error.json` 的终态 trial 会被复用，未完成的 trial 按原 ID 和种子继续执行；已完成的 `sealed_test_metrics.json` 会直接复用，不会再次训练最佳模型或再次打开测试集。续跑用于中断恢复，不会自动重试已记录的失败 trial。

English: `run(resume=True)` or CLI `--resume` verifies the stored configuration, dataset profile and fingerprint, split, and candidate plan. Resume is allowed only when all immutable state matches exactly. Terminal trials with `metrics.json` or `error.json` are reused; missing trials continue with their original IDs and seeds. A completed `sealed_test_metrics.json` is reused without retraining the winner or reopening test data. Resume recovers interruptions and does not automatically retry recorded failures.

## Current Limits / 当前限制

- One trial process at a time / 一次执行一个 trial 进程
- No cooperative pause command; resume recovers only terminal trials, not a partially completed epoch / 没有协作式暂停命令；续跑复用终态 trial，但不能从 trial 内某个未完成 epoch 自动恢复
- Pretraining and parameter-count pruning are declared but not executable / 预训练和参数量剪枝已声明但尚不可执行
- No arbitrary source-code mutation / 不允许任意修改模型源码
- ModelBundle inference currently consumes prepared arrays rather than raw dataset paths / ModelBundle 推理当前接收已样本化数组，不直接接收原始数据路径

## Mechanism Maturity / 机制成熟度

Completed foundation / 已完成基础闭环:

- Native CSV and XJTU-SY ingestion with content fingerprints / CSV 与 XJTU-SY 原生读取和内容指纹
- Multi-step forecasting windows and rolling-origin sealed tests / 多步预测窗口和 rolling-origin 封存测试
- Deterministic sampling, group splitting, and group-CV aggregation / 确定性抽样、分组划分和 group-CV 聚合
- Competing preprocessing, sequence features, sklearn, LightGBM, and PyTorch candidates / 预处理、序列特征、sklearn、LightGBM 和 PyTorch 候选共同竞争
- Fixed task seed, isolated failures, CPU/CUDA selection, and dependency manifests / 固定任务种子、失败隔离、CPU/CUDA 选择和依赖清单
- Validation mean/std ranking plus one-shot sealed-test evaluation / 验证均值与标准差排名，以及一次性封存测试评估
- Registry-driven task candidates and reusable best-model bundles / registry 驱动的任务候选与可复用最佳模型包
- Cross-seed metric/rank stability with independent failure isolation / 多种子指标与排名稳定性及独立失败隔离
- Auditable PyTorch histories, inner validation, early stopping, and best checkpoints / 可审计的 PyTorch 历史、内层验证、早停和最佳 checkpoint
- Explicit smoke-to-regular workflow / 明确的冒烟转常规实验流程

Required before publication-grade ranking / 论文级排行榜前必须补齐:

1. Predefined complete fold assignments for reproducing published protocols.
2. Nested cross-validation when hyperparameter searches require unbiased uncertainty estimates.
3. Separate raw-window, health-indicator sequence, cross-condition adaptation, and survival task contracts.
4. Statistical model-comparison tests beyond the implemented cross-seed aggregation.
5. Multi-series panel forecasting and probabilistic forecast ranking.

1. 支持完整的预定义 fold 分配，以复现论文协议。
2. 超参数搜索需要无偏不确定性估计时，增加嵌套交叉验证。
3. 分离原始窗口、健康指标序列、跨工况适应和生存分析任务协议。
4. 在已实现的多随机种子汇总之上增加统计模型比较检验。
5. 增加多序列 panel forecasting 和概率预测排行榜。

XJTU-SY literature and local protocol analysis are recorded in [`XJTU_SY_MODEL_EVALUATION.md`](XJTU_SY_MODEL_EVALUATION.md).

XJTU-SY 文献与本地协议评估见 [`XJTU_SY_MODEL_EVALUATION.md`](XJTU_SY_MODEL_EVALUATION.md)。

ETTh1 rolling-origin mechanism validation is recorded in
[`FORECASTING_MODEL_EVALUATION.md`](FORECASTING_MODEL_EVALUATION.md).

ETTh1 rolling-origin 机制验证见
[`FORECASTING_MODEL_EVALUATION.md`](FORECASTING_MODEL_EVALUATION.md)。
