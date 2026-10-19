# torch_like / 本地 PyTorch 风格模型库

中文：`torch_like` 是一个本地 PyTorch-native 模型框架，目标是让高价值模型像 `torch.nn` 模块一样被导入、实例化、迁移设备、保存参数和组合使用。

English: `torch_like` is a local PyTorch-native model framework. Its goal is to make high-value models importable, instantiable, device-movable, serializable, and composable like regular `torch.nn` modules.

长期里程碑、退出条件和当前 M1 路径见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。完整 registry 能力与成熟度审计见 [`docs/MODEL_CATALOG_AUDIT.md`](docs/MODEL_CATALOG_AUDIT.md)。

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for long-term milestones and exit criteria, and [`docs/MODEL_CATALOG_AUDIT.md`](docs/MODEL_CATALOG_AUDIT.md) for the complete registry capability and maturity audit.

R1 诊断库基础：新增 `torch_like.diagnostics` 命名空间，提供明确采样率的信号数据协议、诊断结果语义与独立算法注册接口。当前仅完成接口基础，具体统计诊断算法和流程属于后续阶段。详见 [`docs/DIAGNOSTICS_R1.md`](docs/DIAGNOSTICS_R1.md)。

R2 首批统计诊断基线：ZScoreDetector、CUSUMDetector、EWMADetector 使用健康训练基线拟合后进行无标签推理评分，支持逐窗口时间序列评分与固定阈值告警；详见 [R2 说明](docs/DIAGNOSTICS_R2.md)。

R2 多变量与信号特征增强：新增 PCA T²/SPE 监测器（健康基线建模和独立开发集阈值校准），以及基于采样率 Hz 的 FFT 幅值谱、Welch PSD、时域特征计算。详见 [`docs/DIAGNOSTICS_R2B.md`](docs/DIAGNOSTICS_R2B.md)。

R2 窗口特征与传统机器学习整合：新增 `WindowFeatureTransformer` 和 `SklearnWindowAnomalyDetector`，复用原生 `IsolationForest`，将训练集窗口特征拟合与测试集无标签评分严格分离；详见 [`docs/DIAGNOSTICS_R2C.md`](docs/DIAGNOSTICS_R2C.md)。

R3a 诊断任务执行流程：新增按原始记录分组隔离的 `AnomalySplit`、训练/独立验证阈值选择/一次性封存测试的 `AnomalyDiagnosticWorkflow`，并输出带样本溯源及混淆矩阵指标的 `AnomalyEvaluation`。暂限窗口级异常检测，不能当作故障定位或根因推断。详见 [`docs/DIAGNOSTICS_R3A.md`](docs/DIAGNOSTICS_R3A.md)。

R3b 事件级输出：新增 `extract_anomaly_events`，按预测分数的真实分辨率产生事件（窗口级不编造起止时间；时间级允许显式去抖和间隙合并），并支持将 R3a 的一次性窗口评估结果转换为事件提取输入。详见 [`docs/DIAGNOSTICS_R3.md`](docs/DIAGNOSTICS_R3.md)。

R4a 故障类型分类：基于原生 sklearn RandomForest/SVC 和现有时域特征，新增有监督单标签故障分类、固定故障标签字典、按原始记录隔离的训练/验证/封存测试协议，以及宏平均 F1、混淆矩阵与样本溯源。此阶段不宣称实现故障源定位或根因推断。详见 [`docs/DIAGNOSTICS_R4A.md`](docs/DIAGNOSTICS_R4A.md)。

R4b 故障源候选证据映射：新增版本化专家故障类型/部件关联规则 `FaultSourceRulebook` 和无需训练的 `FaultSourceEvidenceMapper`；只输出候选部件支持分数，不将它们解释为真实故障根因的概率。详见 [`docs/DIAGNOSTICS_R4B.md`](docs/DIAGNOSTICS_R4B.md)。

公共 API 受确定性基线和正式弃用周期保护，详见 [`docs/API_STABILITY.md`](docs/API_STABILITY.md)。

Public APIs are protected by a deterministic baseline and formal deprecation lifecycle; see [`docs/API_STABILITY.md`](docs/API_STABILITY.md).

从 `0.5.0` 开始，模型和 sklearn estimator 可通过统一、机器可读的 `ModelSpec / TaskSpec / InputSpec / OutputSpec` 合同进行编排，并可使用可选 `ModelRuntime` 与统一持久化接口；详见 [`docs/API_CONTRACTS.md`](docs/API_CONTRACTS.md)。

Starting with `0.5.0`, torch_like models and curated sklearn estimators expose machine-readable `ModelSpec / TaskSpec / InputSpec / OutputSpec` contracts plus an optional runtime adapter and unified persistence API; see [`docs/API_CONTRACTS.md`](docs/API_CONTRACTS.md).

从 `0.6.0` 开始，原生 PyTorch 模型可使用共享 `Trainer / TrainerConfig` 训练运行时，统一设备、优化器、验证、早停、梯度裁剪和 checkpoint/resume；AutoML 的普通 torch backend 也复用同一 Trainer。详见 [`docs/TRAINING_RUNTIME.md`](docs/TRAINING_RUNTIME.md)。

Starting with `0.6.0`, native PyTorch models can use the shared `Trainer / TrainerConfig` runtime for device placement, optimization, validation, early stopping, gradient clipping, and checkpoint/resume. The normal torch AutoML backend reuses the same Trainer; see [`docs/TRAINING_RUNTIME.md`](docs/TRAINING_RUNTIME.md).

从 `0.7.0` 开始，`DataAdapter / BatchAdapter / AdaptedBatch / MetricEvaluator` 统一数据加载、domain batch 转换和数据集级指标；F1、R² 等非线性指标不再通过 batch metric 简单平均，AutoML 也复用同一 DataAdapter 与指标公式。详见 [`docs/DATA_EVALUATION_RUNTIME.md`](docs/DATA_EVALUATION_RUNTIME.md)。

Starting with `0.7.0`, `DataAdapter / BatchAdapter / AdaptedBatch / MetricEvaluator` unify loader construction, domain-batch adaptation, and dataset-level metrics. Nonlinear metrics such as F1 and R² are no longer approximated by averaging batch metrics, and AutoML reuses the same data/metric primitives; see [`docs/DATA_EVALUATION_RUNTIME.md`](docs/DATA_EVALUATION_RUNTIME.md).

从 `0.8.0` 开始，常用分类、回归和预测指标提供可合并的 streaming state，不再要求缓存完整验证集预测；checkpoint schema v2 明确记录 rank/world-size，并提供基于 PyTorch `DistributedSampler` 的分片合同。0.8.0 当时尚未提供 DDP 执行；0.9.0 的新增能力见下文。

Starting with `0.8.0`, common classification, regression, and forecasting metrics expose mergeable streaming state instead of retaining complete validation predictions. Checkpoint schema v2 records rank/world-size metadata and the training namespace provides a PyTorch `DistributedSampler` contract. DDP execution was not yet implemented in 0.8.0; see the 0.9.0 update below.

从 **0.9.0** 开始，共享 `Trainer` 提供最小 DDP 执行闭环：显式 `DistributedSession`、精确验证分片、流式指标合并、rank-0 写盘与包含每个 rank 状态的 schema-v3 checkpoint。实测范围为 Linux 双进程 CPU/Gloo；CUDA/NCCL、多机与 Windows 未实测。详见 [`docs/DDP_RUNTIME.md`](docs/DDP_RUNTIME.md)，可运行示例见 [`examples/train_ddp_tabular.py`](examples/train_ddp_tabular.py)。

Starting with **0.9.0**, Trainer supports a bounded DDP execution contract with exact validation sharding and coordinated rank-local resume state. CPU/Gloo with two local ranks is tested; CUDA/NCCL is implemented but unverified in this release environment.

模型成熟度只由可审计的本地证据登记驱动，详见 [`docs/MODEL_EVIDENCE.md`](docs/MODEL_EVIDENCE.md)。

Model maturity is driven only by auditable local evidence records; see [`docs/MODEL_EVIDENCE.md`](docs/MODEL_EVIDENCE.md).

## 定位边界 / Scope Boundary

中文：`torch_like` 是模型框架优先，并提供轻量训练运行时；它不是端到端生产训练平台，也不是 `torch.nn` 的逐层复刻。

English: `torch_like` is a model framework first with a lightweight training runtime. It is not an end-to-end production training platform and does not try to clone every layer in `torch.nn`.

负责内容 / Responsibilities:

- 中文：统一模型定义、任务接口、batch 协议、顶层导入和 registry 构造。
- English: Unified model definitions, task interfaces, batch contracts, top-level imports, and registry-based construction.
- 中文：提供面向训练和推理的稳定模型 API、单进程及受限 DDP 训练生命周期，以及可复用的数据/评估适配层。
- English: Stable training-facing/inference-facing model APIs plus a single-process and bounded-DDP PyTorch training lifecycle and reusable data/evaluation adapters.
- 中文：沉淀新增模型的接入规范。
- English: Integration rules for adding new models.

不负责内容 / Non-goals:

- 中文：不负责数据集专属预处理、分布式生产训练编排、通用超参搜索或复现每篇论文最优分数。
- English: It does not own dataset-specific preprocessing, distributed production-training orchestration, generic hyperparameter search, or best-score reproduction for every paper.
- 中文：`runner/` 只是辅助实验工具，不是核心框架 API。
- English: `runner/` is an auxiliary experiment layer, not the core framework API.

## 与 `torch.nn` 的关系 / Relation to `torch.nn`

中文：兼容目标是 `nn.Module` 对象模型，而不是每个 PyTorch 内置层的特殊调用签名。

English: The compatibility target is the `nn.Module` object model, not every specialized calling convention of PyTorch built-in layers.

支持的通用行为 / Supported common behavior:

- `model(...)` / `forward(...)`
- `state_dict()` / `load_state_dict(...)`
- `parameters()` / `named_parameters()`
- `train()` / `eval()`
- `to()` / `cpu()` / `cuda()`
- `config()` / `from_config(...)`
- `param_count()`

框架约定 / Framework conventions:

- 中文：forecasting 模型使用 `make_forecast_batch(...)` 和 `predict(batch)`。
- English: Forecasting models use `make_forecast_batch(...)` and `predict(batch)`.
- 中文：detection 模型使用 `make_detection_batch(...)`、`fit_step(batch)` 和 `score_batch(batch)`。
- English: Detection models use `make_detection_batch(...)`, `fit_step(batch)`, and `score_batch(batch)`.
- 中文：registry 支持 `create_model(name, domain=..., **kwargs)`。
- English: The registry supports `create_model(name, domain=..., **kwargs)`.
- 中文：`describe(name)` / `get_model_spec(name)` 返回机器可读的任务、输入、输出和 capability 合同。
- English: `describe(name)` / `get_model_spec(name)` returns machine-readable task, input, output, and capability contracts.
- 中文：`as_runtime(model)` 提供可选的跨 backend 编排层；原生 `nn.Module` / sklearn estimator API 保持不变。
- English: `as_runtime(model)` provides an optional cross-backend orchestration layer while preserving native `nn.Module` / sklearn estimator APIs.
- 中文：`save_model(...)` / `load_model(...)` 提供带 manifest 的统一模型持久化。
- English: `save_model(...)` / `load_model(...)` provides manifest-backed model persistence.
- 中文：`Trainer(...)` / `TrainerConfig(...)` 提供统一的单设备训练、验证、早停和断点恢复；`TensorDataAdapter` / `BatchAdapter` / `MetricEvaluator` 统一数据与评估胶水层。
- English: `Trainer(...)` / `TrainerConfig(...)` provides shared single-device training, validation, early stopping, and resume semantics; `TensorDataAdapter` / `BatchAdapter` / `MetricEvaluator` unify data/evaluation glue.
- 中文：registry 支持 `list_domains()`、`list_models(...)` 和 `get_model_info(...)`，返回的元数据副本可安全修改。
- English: The registry exposes `list_domains()`, `list_models(...)`, and `get_model_info(...)`; returned metadata snapshots are safe to modify.
- 中文：registry 以兼容方式同时返回旧 `integration_level` 和 schema v2 的一致性、范围、执行、成熟度与证据字段，并支持 `list_models_by_fidelity(...)`、`list_models_by_maturity(...)` 查询。
- English: Registry snapshots retain legacy `integration_level` while adding schema-v2 fidelity, scope, execution, maturity, and evidence fields, queryable through `list_models_by_fidelity(...)` and `list_models_by_maturity(...)`.

## 当前模型覆盖 / Current Model Coverage

### Forecasting / 时间序列预测

Transformer / attention:

- `AST`
- `Informer`
- `Autoformer`
- `FEDformer`
- `LogTrans`
- `PatchTST`
- `ITransformer`
- `Crossformer`
- `TimeXer`
- `DAG`

Exogenous forecasting:

- `TFT`
- `TiDE`

MLP / decomposition / convolutional mixing:

- `MLCNN`
- `NBEATS`
- `NHITS`
- `DLinear`
- `SCINet`
- `TimesNet`
- `MICN`
- `TimeMixer`
- `TSMixer`
- `ConvTimeNet`
- `ModernTCN`
- `SOFTS`

Graph spatiotemporal forecasting:

- `ASTGCN`
- `AGCRN`
- `D2STGNN`
- `DMSTGCN`
- `MTGNN`
- `StemGNN`
- `GraphWaveNet`
- `DCRNN`
- `FCGAGA`
- `GTS`
- `STFGNN`
- `STGODE`

Traffic-specialized dynamic similarity:

- `STDN`

Probabilistic forecasting:

- `DeepAR`
- `TimeGrad`
- `ProTran`
- `PPM`

Dynamical systems forecasting:

- `Koopa`

Frequency-domain forecasting:

- `FreTS`

### RUL / 剩余寿命预测

- `InformerRUL`
- `AutoformerRUL`
- `LogTransRUL`
- `THGNN`
- `TCNRUL`
- `PatchTSTRUL`
- `TransformerRUL`

### Detection / 异常检测

- `RTDetector`
- `AnomalyTransformer`
- `MEMTO`
- `GDN`
- `MTADGAT`
- `DCDetector`
- `OmniAnomaly`
- `TranAD`
- `USAD`
- `CBAMTCNDetector` (experimental task adapter; explicit AutoML selection only)

### Graph / 通用图学习

- `GCN`
- `EGNN`
- `DGI`
- `GraphCL`
- `GraphMAE`
- `GAT`
- `GATv2`
- `GNAN`
- `GraphSAGE`
- `GIN`
- `GraphTransformer`
- `Graphormer`
- `GPS`
- `PNA`
- `GCNII`
- `SpikeGraphormer`

### Sequence / Long-Context Modeling / 序列与长上下文建模

- `TransformerDecoder`
- `TransformerXL`
- `Performer`
- `Reformer`
- `RetNet`
- `KimiLinear`
- `xLSTM`
- `GPT`
- `LLaMA`
- `Mixtral`
- `LRU`
- `Mamba`
- `Mamba2`
- `MambaLike`
- `Caracal`
- `S4`
- `Hyena`
- `BERT`
- `T5`

`KimiLinear` is a lightweight, randomly initialized Kimi Delta Attention reference model. The package exposes
`KimiDeltaAttention` with `parallel`, `chunkwise`, and recurrent `forward_step(...)` paths, but does not bundle
the official 48B checkpoint, FlashKDA kernels, or claim their throughput.

`xLSTM` provides the matrix-memory mLSTM path with stabilized exponential gates and explicit recurrent state.
It is a portable reference implementation and does not bundle the official xLSTM CUDA kernels or pretrained weights.

### Diagnosis / 故障诊断

- `InceptionTime`
- `ROCKET`

### Vision / 视觉模型

- `ResNet`
- `ViT`
- `MAE`
- `SimCLR`
- `BYOL`
- `DINO`
- `MoCo`
- `SAM`
- `DeiT`
- `UNet`
- `DeepLabV3`
- `DETR`
- `RTDETR`
- `GroundingDINO`
- `Mask2Former`
- `FCN`
- `DenseNet`
- `MobileNetV2`
- `EfficientNet`
- `SwinTransformer`
- `ConvNeXt`
- `GoogLeNet`
- `InceptionV3`
- `VGG`
- `AlexNet`
- `MLPMixer`
- `ResNeXt`
- `SENet`

### Multimodal / 多模态模型

- `CLIP`
- `SigLIP`
- `BLIP`
- `LLaVA`
- `BLIP2`

### Generative / 生成模型

- `DDPM`
- `VAE`
- `DCGAN`
- `DiT`
- `ControlNet`
- `LatentDiffusion`
- `PixelCNN`
- `RectifiedFlow`
- `RealNVP`
- `VQVAE`

### Tabular / 表格模型

- `TabNet`
- `NODE`
- `FTTransformer`

### Recommendation / 推荐系统

- `NCF`
- `DeepFM`
- `DCN`
- `BERT4Rec`

### Audio / 音频与语音

- `Wav2Vec2`
- `EnCodec`
- `Conformer`
- `HuBERT`
- `Whisper`
- `AudioSpectrogramTransformer`

## 安装 / Installation

```powershell
& 'D:\Project\base\base\Scripts\python.exe' -m pip install -e .
```

中文：上面的命令用于开发环境。若要在其他 Python 环境中安装固定版本，应先构建 wheel，再安装 `dist/` 中的文件。

English: The command above is for development. To install an immutable build in another Python environment, build a wheel and install the file under `dist/`.

```powershell
& .\tools\build_package.ps1 -Python 'D:\Project\base\base\Scripts\python.exe'
& 'D:\Path\To\OtherPython\python.exe' -m pip install .\dist\torch_like-0.8.0-py3-none-any.whl
```

中文：完整构建、隔离验收、升级和卸载说明见 [Packaging Guide / 打包安装指南](docs/PACKAGING.md)。

English: See the [Packaging Guide / 打包安装指南](docs/PACKAGING.md) for build, isolated verification, upgrade, and uninstall instructions.

## 离线部署 / Offline Deployment

中文：离线部署不能只复制 `torch_like` wheel，除非目标环境已经安装兼容版本的 PyTorch、NumPy、pandas 和 scikit-learn。推荐在一台与目标机器操作系统、CPU 架构和 Python 版本一致的联网机器上准备完整 wheelhouse。

English: Do not copy only the `torch_like` wheel unless the target already has compatible PyTorch, NumPy, pandas, and scikit-learn installations. Prepare a complete wheelhouse on an online machine matching the target operating system, CPU architecture, and Python version.

### 1. 联网机器准备离线包 / Prepare on an Online Machine

```powershell
New-Item -ItemType Directory -Force .\offline_bundle\wheelhouse

# Build torch_like first.
& .\tools\build_package.ps1 -Python 'D:\Project\base\base\Scripts\python.exe'

# Download torch_like and all transitive dependencies into one directory.
& 'D:\Project\base\base\Scripts\python.exe' -m pip download `
  --dest .\offline_bundle\wheelhouse `
  .\dist\torch_like-0.8.0-py3-none-any.whl

Compress-Archive -Path .\offline_bundle\wheelhouse -DestinationPath .\torch_like-offline.zip -Force
```

中文：如果目标机器需要特定 CUDA 版本，应先按照 PyTorch 官方安装选择器确定对应 wheel 源，并把匹配的 PyTorch wheel 下载到同一个 `wheelhouse`。不要在 CPU、CUDA 版本或 Python ABI 不一致的机器上直接复用 wheelhouse。

English: For a target requiring a specific CUDA build, use the official PyTorch installation selector to identify the matching wheel source and download that PyTorch wheel into the same `wheelhouse`. Do not reuse a wheelhouse across incompatible CPU architectures, CUDA builds, or Python ABIs.

### 2. 离线机器安装 / Install on the Offline Machine

```powershell
Expand-Archive .\torch_like-offline.zip -DestinationPath .\torch_like-offline -Force

& 'D:\Path\To\OfflinePython\python.exe' -m pip install `
  --no-index `
  --find-links .\torch_like-offline\wheelhouse `
  torch_like
```

### 3. 验证安装 / Verify the Installation

```powershell
& 'D:\Path\To\OfflinePython\python.exe' -m pip check
& 'D:\Path\To\OfflinePython\python.exe' -c `
  "import torch, torch_like as tl; print(torch.__version__); print(tl.__version__); print(tl.list_domains())"
```

中文：若目标环境已经预装全部依赖，可以只复制 `torch_like-*.whl` 并使用 `pip install --no-deps`；此模式不会检查或安装缺失依赖，应由部署方负责兼容性验证。

English: If every dependency is already installed on the target, copy only `torch_like-*.whl` and use `pip install --no-deps`. This mode neither checks nor installs missing dependencies, so the deployer owns compatibility verification.

## 名称与许可提示 / Naming and Licensing Notice

中文：`torch_like` 是独立项目，不隶属于、不受 PyTorch Foundation 或 Linux Foundation 赞助或认可。PyTorch 及其相关标识是其权利人的商标。本项目当前适合内部研究和本地使用；在公开发布、注册包名或商业分发前，应完成商标名称审查、项目许可证选择、第三方代码来源清单和专利审查。

English: `torch_like` is an independent project and is not affiliated with, sponsored by, or endorsed by the PyTorch Foundation or Linux Foundation. PyTorch and related marks are trademarks of their respective owners. Project-owned code is provided under Apache-2.0. Trademark clearance, third-party provenance records, and patent review remain necessary before public or commercial distribution.

中文：项目贡献者有权授权的代码采用 [Apache License 2.0](LICENSE)。该许可证不授予 PyTorch 商标使用权，也不会覆盖第三方代码、模型权重或数据集各自的许可义务。

English: Code that project contributors are entitled to license is provided under the [Apache License 2.0](LICENSE). This license grants no rights to PyTorch trademarks and does not replace the separate license obligations of third-party code, model weights, or datasets.

## 0.4 工程化入口 / 0.4 Engineering Entry Points

中文：`torch_like 0.4` 将用户入口分成 PyTorch-native 与 sklearn-native 两条清晰路径。模型 domain 的 `__init__.py` 已改为按需加载；直接访问一个模型不会再加载整个模型库。完整架构见 [`docs/ENGINEERING_ARCHITECTURE.md`](docs/ENGINEERING_ARCHITECTURE.md)。

English: `torch_like 0.4` separates PyTorch-native and sklearn-native entry points. Model-domain `__init__.py` files now load lazily, so accessing one model no longer imports the whole model library. See [`docs/ENGINEERING_ARCHITECTURE.md`](docs/ENGINEERING_ARCHITECTURE.md).

Recommended / 推荐：

```python
import torch_like as tl

# PyTorch-native model; domain can be inferred by the new backend-aware factory.
model = tl.create("DLinear", enc_in=4, dec_in=4, c_out=4, seq_len=24, label_len=12, pred_len=6)

# Native sklearn estimator; returned object is the original sklearn type.
baseline = tl.create("RandomForestRegressor", backend="sklearn", n_estimators=200)
```

轻量发现 / Lightweight discovery:

```python
print(tl.list_domains())
print(tl.list_model_names("forecasting"))
print(tl.list_estimators("classification"))
```

中文：`list_model_names()` 与 `list_estimators()` 只读取静态索引，不导入模型实现或 sklearn。需要完整治理元数据时再使用 `list_models()` / `get_model_info()`。

English: `list_model_names()` and `list_estimators()` read static indexes only; they do not import model implementations or sklearn. Use `list_models()` / `get_model_info()` when full governance metadata is required.

### sklearn 原生兼容 / Native sklearn Compatibility

```python
from torch_like.classical import RandomForestClassifier, Ridge, create_estimator

clf = RandomForestClassifier(n_estimators=100, random_state=0)
reg = create_estimator("Ridge", alpha=1.0)
```

中文：这些类与 `sklearn.*` 中的原类是同一对象；本项目不把 sklearn estimator 伪装成 `torch.nn.Module`。

English: These classes are the same class objects exported by `sklearn.*`; the project does not disguise sklearn estimators as `torch.nn.Module` objects.

## 快速开始 / Quick Start

### 直接导入 / Direct Import

```python
import torch
import torch_like as tl

model = tl.PatchTST(
    enc_in=3,
    dec_in=3,
    c_out=3,
    seq_len=16,
    label_len=8,
    pred_len=4,
    patch_len=4,
)

x_enc = torch.randn(2, 16, 3)
batch = model.make_forecast_batch(x_enc)
out = model.predict(batch)
print(out.shape)  # (2, 4, 3)
```

### Registry 构造 / Registry Construction

```python
from torch_like.models import create_model, get_model_info, list_domains, list_models

print(list_domains())
print(list_models("forecasting").keys())
print(get_model_info("ITransformer", domain="forecasting"))

model = create_model(
    "ITransformer",
    domain="forecasting",
    enc_in=4,
    dec_in=4,
    c_out=4,
    seq_len=16,
    label_len=8,
    pred_len=4,
)
```

### 图预测：学习图 + 外部图先验 / Graph Forecasting: Learned Graph + External Prior

```python
import numpy as np
import torch
import torch_like as tl

adj = np.eye(4, dtype=np.float32)

model = tl.AGCRN(
    enc_in=4,
    dec_in=4,
    c_out=4,
    seq_len=16,
    label_len=8,
    pred_len=4,
    graph_mode="hybrid",
    adjacency=adj,
    prior_weight=0.6,
)

x = torch.randn(2, 16, 4)
out = model.predict(model.make_forecast_batch(x))
```

Graph mode semantics / 图模式含义:

- `learned`: 中文：只学习图；English: learned graph only.
- `prior`: 中文：只使用外部邻接矩阵；English: external adjacency only.
- `hybrid`: 中文：学习图和外部图先验加权融合；English: weighted fusion of learned graph and prior adjacency.

### 异常检测 / Anomaly Detection

```python
import torch
import torch_like as tl

model = tl.GDN(input_dim=8, seq_len=64, d_model=32)
x = torch.randn(4, 64, 8)
batch = model.make_detection_batch(x)

loss = model.fit_step(batch)
scores = model.score_batch(batch)
```

### RUL 预测 / RUL Prediction

```python
import torch
import torch_like as tl

model = tl.InformerRUL(enc_in=8, seq_len=64, d_model=32)
x = torch.randn(4, 64, 8)
out = model.predict(model.make_rul_batch(x))
print(out.shape)  # (4,)
```

All RUL models accept `RULBatch` and expose `predict(...)` and `loss_step(...)`. `TCNRUL`,
`PatchTSTRUL`, and `TransformerRUL` are framework-oriented task adapters, not claims of
paper-exact RUL architectures or SOTA performance.

所有 RUL 模型都接受 `RULBatch`，并提供 `predict(...)` 与 `loss_step(...)`。`TCNRUL`、
`PatchTSTRUL` 和 `TransformerRUL` 是面向框架的任务适配器，不代表论文原样复现或 SOTA 性能声明。

## 辅助 Runner / Auxiliary Runners

中文：`runner/` 用于快速验证模型能否实例化、训练一步、评估和比较；它不是框架核心。

English: `runner/` helps validate instantiation, one-step training, evaluation, and comparison. It is not the core framework.

Common entry points / 常用入口:

- `runner/forecast_runner.py`
- `runner/forecast_comparison_runner.py`
- `runner/detection_runner.py`
- `runner/rul_runner.py`
- `runner/benchmark_compile.py`

## 性能路径 / Performance Paths

中文：`torch_like` 优先复用 PyTorch 原生性能能力，而不是维护自定义 CUDA 内核。`FullAttention`、`GPT`、`LLaMA` 以及复用 LLaMA 注意力的 `Mixtral` 已接入 `scaled_dot_product_attention`；当调用者要求返回注意力权重时，`FullAttention` 会保留可观测的显式计算路径。

English: `torch_like` reuses native PyTorch performance primitives instead of maintaining custom CUDA kernels. `FullAttention`, `GPT`, `LLaMA`, and `Mixtral` through its shared LLaMA attention use `scaled_dot_product_attention`; `FullAttention` retains the explicit path when attention weights are requested.

中文：辅助 runner 通过 `ExecutionConfig` 统一控制 AMP、异步 CUDA 搬运和 `zero_grad(set_to_none=True)`，通过 `CompileConfig` 可选启用 `Module.compile()`。这些选项默认不改变现有 FP32 行为。

English: Auxiliary runners use `ExecutionConfig` for AMP, asynchronous CUDA transfers, and `zero_grad(set_to_none=True)`, with optional `Module.compile()` through `CompileConfig`. Defaults preserve existing FP32 behavior.

```powershell
& 'D:\Project\base\base\Scripts\python.exe' runner/forecast_runner.py `
  --dataset-type ett_hour --root-path .\data --data-path ETTh1.csv `
  --amp --amp-dtype bfloat16 --compile --compile-mode reduce-overhead
```

中文：自回归生成的 KV cache 尚未纳入本轮优化；当前 `generate(...)` 仍会重新计算完整前缀。

English: Autoregressive KV caching is not part of this optimization batch; current `generate(...)` methods still recompute the full prefix.

## 文档 / Documentation

- [User Guide / 用户指南](docs/USER_GUIDE.md)
- [Development Guide / 开发指南](docs/DEVELOPMENT.md)
- [Project Structure / 项目结构](docs/PROJECT_STRUCTURE.md)
- [Sequence Domain Draft / 序列域草案](docs/SEQUENCE_DOMAIN_DRAFT.md)

## 设计重点 / Design Highlights

- 中文：模型优先，runner 辅助。
- English: Models first, runners as auxiliary tools.
- 中文：任务域分层，不把所有序列模型强行塞进 forecasting。
- English: Task domains are separated; not every sequence model is forced into forecasting.
- 中文：新增模型必须能以 `nn.Module` 方式直接调用。
- English: New models must be directly usable as `nn.Module` components.
- 中文：graph forecasting 统一支持 `learned/prior/hybrid` 图模式。
- English: Graph forecasting uses a unified `learned/prior/hybrid` graph mode protocol.

## 测试 / Testing

```powershell
& 'D:\Project\base\base\Scripts\python.exe' -m unittest discover -s tests
& 'D:\Project\base\base\Scripts\python.exe' -m ruff check torch_like runner tests examples tools
& 'D:\Project\base\base\Scripts\python.exe' -m ruff format --check torch_like runner tests examples tools
& 'D:\Project\base\base\Scripts\python.exe' tools/validate_candidate_pool.py
& 'D:\Project\base\base\Scripts\python.exe' -m pytest --cov=torch_like --cov-report=term-missing
```

中文：测试覆盖直接导入、registry 构造、任务 batch、预测输出形状、检测分数接口和 runner 基础行为。

English: Tests cover direct imports, registry construction, task batches, prediction shapes, detection scoring APIs, and basic runner behavior.

## 当前状态 / Current Status

中文：框架已具备多任务模型库骨架，覆盖 forecasting、diagnosis、detection、graph、RUL、sequence、vision、multimodal、generative、tabular、recommendation、audio 和 reinforcement。后续新增模型应优先保持 API 一致性，而不是追求实验脚本数量。

English: The framework now provides a multi-task model-library skeleton covering forecasting, diagnosis, detection, graph, RUL, sequence, vision, multimodal, generative, tabular, recommendation, audio, and reinforcement domains. Future additions should prioritize API consistency over accumulating experiment scripts.

高价值模型候选、证据等级和正式收录决策见 [`docs/MODEL_CANDIDATE_POOL.md`](docs/MODEL_CANDIDATE_POOL.md)。

See [`docs/MODEL_CANDIDATE_POOL.md`](docs/MODEL_CANDIDATE_POOL.md) for the evidence-graded candidate pool and formal admission decisions.

中文：准入阈值、证据要求和复审期限由 [`governance/model_admission_policy.json`](governance/model_admission_policy.json) 统一定义。候选模型不得绕过候选记录、硬门槛、证据评分、复审台账和框架合同测试直接进入公共 API。

English: Admission thresholds, evidence requirements, and review deadlines are defined centrally in [`governance/model_admission_policy.json`](governance/model_admission_policy.json). A model may not enter the public API without a candidate record, hard-gate review, evidence score, review registration, and framework-contract tests.

## AutoML Leaderboard / 自动模型排行榜

The optional `torch_like.automl` layer supports CSV datasets, the native folder layout of
XJTU-SY, and raw ESA-ADB mission ZIP archives. XJTU-SY tasks derive normalized RUL from acquisition-file position, deterministically
bound file/window sampling, and support bearing-level `group_cv` with validation mean/std ranking
and one-shot sealed-test evaluation. See
[`docs/AUTOML.md`](docs/AUTOML.md) and [`examples/xjtu_sy_rul_leaderboard.yaml`](examples/xjtu_sy_rul_leaderboard.yaml).

XJTU-SY literature screening, protocol differences, local regular-run results, and rejected candidates are recorded in
[`docs/XJTU_SY_MODEL_EVALUATION.md`](docs/XJTU_SY_MODEL_EVALUATION.md).

XJTU-SY 文献筛选、协议差异、本地常规实验结果和候选否决记录见
[`docs/XJTU_SY_MODEL_EVALUATION.md`](docs/XJTU_SY_MODEL_EVALUATION.md)。

可选的 `torch_like.automl` 层支持 CSV 数据和 XJTU-SY 原生文件夹结构。XJTU-SY 任务根据采样文件在
生命周期中的位置生成归一化 RUL，可确定性限制文件数和窗口数，并支持轴承级 `group_cv`：按验证 fold
均值和标准差排名，只对最终选中的管线执行一次封存测试。

中文：`torch_like.automl` 提供 CSV 数据画像、`row/full_sequence/sliding_window` 样本化、防泄漏划分、scikit-learn 与部分 `torch_like` 模型候选、预处理竞争、隔离 trial、验证集排名和可复现产物。详细配置、CLI 和 Python API 见 [`docs/AUTOML.md`](docs/AUTOML.md)。

English: `torch_like.automl` provides CSV profiling, `row/full_sequence/sliding_window` sampling, registry-driven candidates, sealed final tests for every split protocol, preprocessing competition, isolated trials, reproducible artifacts, and `load_best()` inference bundles. See [`docs/AUTOML.md`](docs/AUTOML.md) for configuration, CLI, and Python APIs.

中文：排行榜候选策略支持 `production`、`torch_like_only`、`baseline_only` 和 `prefer_torch_like`，用于区分“寻找实际最优管线”和“验证框架模型”两类目标。每次运行同时输出总榜、`torch_like` 模型榜和传统基线榜，但只由总榜冠军执行一次最终封存测试。

English: Candidate policies distinguish practical model selection from framework validation: `production`, `torch_like_only`, `baseline_only`, and `prefer_torch_like`. Every run exports overall, torch-like, and baseline leaderboards, while only the overall winner receives the one-shot sealed test.

中文：非图时序分类可以通过 `sample_mode: full_sequence` 调用 `torch_like` 的 `InceptionTime` 与 `ROCKET`，并与 sklearn 序列特征基线共同进入排行榜。FordA 示例见 [`examples/time_series_classification_leaderboard.yaml`](examples/time_series_classification_leaderboard.yaml)。

English: Non-graph time-series classification can use `sample_mode: full_sequence` to rank `torch_like` `InceptionTime` and `ROCKET` alongside sklearn sequence-feature baselines. The FordA example is in [`examples/time_series_classification_leaderboard.yaml`](examples/time_series_classification_leaderboard.yaml).

中文：多标签分类支持每个标签一列的 0/1 CSV 目标，排行榜可以同时比较 sklearn 多输出包装器和 `torch_like` 表格神经网络候选。

English: Multilabel classification supports one 0/1 CSV column per label, and the leaderboard can compare sklearn multi-output wrappers with `torch_like` tabular neural candidates.

Tabular classification leaderboards can compare sklearn/LightGBM baselines with `torch_like` neural candidates including `FTTransformer`, `TabNet`, `NODE`, and `SAINT` under the same split and ranking protocol.

表格分类排行榜可以在相同数据划分和排名协议下，让 sklearn/LightGBM 基线与 `torch_like` 的 `FTTransformer`、`TabNet`、`NODE`、`SAINT` 神经网络候选共同竞争。

中文：所有划分协议现在都只按验证指标排列候选，测试数据仅在最佳管线选定并重训后解封一次。`load_best()` 可恢复可信本地的入选管线、预处理器和 PyTorch 状态并执行推理。

Interrupted immutable tasks can continue with `torch-like run --config task.yaml --resume`; completed and failed trial records are reused, and a completed sealed test is never reopened automatically.

不可变排行榜任务中断后可通过 `torch-like run --config task.yaml --resume` 继续；已完成和已失败的 trial 记录都会复用，已经完成的封存测试不会被自动再次打开。

Multi-seed ranking is available through `search.evaluation_seeds`. Each pipeline/seed pair runs in isolation; only complete seed coverage is ranked, with metric and rank stability exported before one-shot sealed testing.

可通过 `search.evaluation_seeds` 启用多种子排名。每个管线与种子组合独立运行，只有完成全部种子的管线才参与排名，并在一次性封存测试前输出指标与名次稳定性。

Validated leaderboard trust can be declared with `trust.level: validated`; this requires at least three evaluation seeds and writes `evaluation_trust_audit.json` with seed-coverage and budget checks. The default `development` level remains suitable for smoke tests and model-adapter work.

可通过 `trust.level: validated` 声明验证级排行榜；该模式要求至少 3 个评估种子，并写出包含种子覆盖和预算检查的 `evaluation_trust_audit.json`。默认 `development` 等级仍适合 smoke 测试和模型适配开发。

Benchmark trust tiers and promotion rules are maintained in [`docs/AUTOML_BENCHMARK_TRUST.md`](docs/AUTOML_BENCHMARK_TRUST.md). A validated rolling-origin forecasting template is available at [`examples/forecasting_rolling_validated.yaml`](examples/forecasting_rolling_validated.yaml).

排行榜可信分层和晋升规则见 [`docs/AUTOML_BENCHMARK_TRUST.md`](docs/AUTOML_BENCHMARK_TRUST.md)。验证级 rolling-origin 预测模板见 [`examples/forecasting_rolling_validated.yaml`](examples/forecasting_rolling_validated.yaml)。

AutoML YAML examples are statically checked by `python tools/validate_automl_examples.py` and by CI, so example drift is treated as a governance failure.

AutoML YAML 示例由 `python tools/validate_automl_examples.py` 和 CI 做静态检查；示例漂移会被视为治理失败。

Completed leaderboard runs can be audited with `python tools/validate_automl_runs.py runs/<task_id>`. To scan a mixed local `runs/` folder, use `python tools/validate_automl_runs.py --skip-legacy runs`. For `validated` runs, the checker requires passed protocol audits, complete seed coverage, one completed sealed-test record, and no test metrics leaked into `leaderboard.csv`.

已完成的排行榜运行可用 `python tools/validate_automl_runs.py runs/<task_id>` 审计。扫描混合了旧协议产物的本地 `runs/` 文件夹时，使用 `python tools/validate_automl_runs.py --skip-legacy runs`。对于 `validated` 运行，检查器要求协议审计通过、种子覆盖完整、封存测试完成且 `leaderboard.csv` 不泄露测试指标。

ESA-ADB anomaly leaderboards load selected channels directly from the raw mission ZIP, use validation-only threshold selection, rank corrected event-wise F0.5, and open the test period only for the selected detector. See [`docs/ESA_ADB_MODEL_EVALUATION.md`](docs/ESA_ADB_MODEL_EVALUATION.md) and [`examples/esa_mission1_anomaly_leaderboard.yaml`](examples/esa_mission1_anomaly_leaderboard.yaml).

ESA-ADB 异常检测排行榜可直接读取原始任务 ZIP，只缓存选定通道；阈值仅在验证集选择，按 corrected event-wise F0.5 排名，并只对最终入选检测器解封测试时间段。

Forecasting leaderboard tasks support multi-step targets, expanding-window validation, per-horizon metrics, and `LastValue`/DLinear/PatchTST/TSMixer comparisons. See [`examples/forecasting_rolling_leaderboard.yaml`](examples/forecasting_rolling_leaderboard.yaml).

预测排行榜支持多步目标、扩展窗口验证、逐 horizon 指标，以及 `LastValue`、DLinear、PatchTST、TSMixer 对比。配置见 [`examples/forecasting_rolling_leaderboard.yaml`](examples/forecasting_rolling_leaderboard.yaml)。

ETTh1 上的真实机制验证、目标缩放修复和模型准入结论见
[`docs/FORECASTING_MODEL_EVALUATION.md`](docs/FORECASTING_MODEL_EVALUATION.md)。

LightGBM is an optional AutoML backend rather than an `nn.Module` model. Install it with
`pip install "torch_like[lightgbm]"`; sequence pipelines can compare basic statistics, vibration statistics, and raw flattening.

LightGBM 是可选的 AutoML 后端，不属于 `nn.Module` 模型。可通过 `pip install "torch_like[lightgbm]"` 安装；
序列管线可以比较基础统计、振动统计和原始展平特征。
XJTU-SY 配置示例见 [`examples/xjtu_sy_rul_lightgbm.yaml`](examples/xjtu_sy_rul_lightgbm.yaml)。

## Recently Added Domains / 新增领域

Reinforcement Learning / 强化学习:

- `DQN`
- `ActorCritic`
- `PPOActorCritic`
- `SACActorCritic`
- `TD3ActorCritic`
- `DecisionTransformer`
- `DreamerV3`

English: reinforcement models expose reusable `nn.Module` cores such as Q values, policy logits, value functions, action evaluation, TD loss, PPO clipped loss, SAC twin-Q stochastic continuous-control losses, TD3 deterministic actor-critic losses, and Decision Transformer return-conditioned trajectory prediction. Environment stepping, offline dataset loading, replay buffers, rollout collection, exploration schedules, target-network updates, entropy-temperature tuning, and benchmark protocols remain outside the core framework.

Recommendation / 推荐系统:

- `LightGCN`
- `SASRec`
- `BERT4Rec`

English: `LightGCN` adds graph collaborative filtering with BPR-style ranking loss and full-sort scoring; `SASRec` adds self-attentive sequential recommendation for next-item prediction; `BERT4Rec` adds bidirectional masked-item prediction for item sequences. Negative sampling, candidate generation, ranking metrics, and online serving remain outside model classes.

Tabular / 表格学习:

- `TabTransformer`
- `SAINT`
- `NODE`

English: `TabTransformer` adds contextual categorical embeddings for mixed tabular prediction; `SAINT` adds column and optional row attention over tabular tokens; `NODE` adds differentiable oblivious decision-tree ensembles for tabular classification and regression. Data cleaning, categorical encoding policy, missing-value handling, augmentation, contrastive pre-training loops, and benchmark protocol remain outside model classes.

Vision / 视觉检测:

- `YOLO`
- `GroundingDINO`

English: `YOLO` adds a single-stage real-time detection skeleton with anchor-grid predictions, objectness scores, class logits, normalized boxes, and `post_process(...)`. `GroundingDINO` adds language-grounded open-vocabulary detection with text-token alignment logits, grounded objectness scores, normalized boxes, and `post_process(...)`. Tokenizers, phrase parsing, NMS policy, COCO/LVIS evaluation, Grounded-SAM orchestration, and training recipes remain outside model classes.

English: `SAM` adds promptable segmentation through `encode_image(...)`, `encode_prompts(...)`, `predict_masks(...)`, and `forward(..., points=..., boxes=..., masks=...)`. Official checkpoint loading, interactive annotation tools, and SAM/SAM2 training recipes remain outside model classes.
中文：`SAM` 提供点、框、mask 提示驱动的分割模型接口；官方权重下载、交互标注工具和 SAM/SAM2 训练流程不属于核心模型类。

English: `Mask2Former` adds universal segmentation through query masks, class logits, semantic-logit composition, and `post_process(...)`. Hungarian matching, panoptic fusion, dataset-specific target formatting, and benchmark evaluation remain outside model classes.

Multimodal / 多模态:

- `BLIP`
- `BLIP2`

English: `BLIP` adds image-text fusion, image-text matching logits, and caption language-model logits on top of dual-encoder retrieval features. `BLIP2` adds a Q-Former bridge from image tokens to language-prefix features. Tokenizers, large language model weights, synthetic captions, filtering pipelines, and BLIP/BLIP-2 pre-training schedules remain outside model classes.
