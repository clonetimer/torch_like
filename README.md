# torch_like / 本地 PyTorch 风格模型库

中文：`torch_like` 是一个本地 PyTorch-native 模型框架，目标是让高价值模型像 `torch.nn` 模块一样被导入、实例化、迁移设备、保存参数和组合使用。

English: `torch_like` is a local PyTorch-native model framework. Its goal is to make high-value models importable, instantiable, device-movable, serializable, and composable like regular `torch.nn` modules.

## 定位边界 / Scope Boundary

中文：`torch_like` 是模型框架优先，不是完整训练平台，也不是 `torch.nn` 的逐层复刻。

English: `torch_like` is a model framework first. It is not a full training platform and does not try to clone every layer in `torch.nn`.

负责内容 / Responsibilities:

- 中文：统一模型定义、任务接口、batch 协议、顶层导入和 registry 构造。
- English: Unified model definitions, task interfaces, batch contracts, top-level imports, and registry-based construction.
- 中文：提供面向训练和推理的稳定模型 API。
- English: Stable training-facing and inference-facing model APIs.
- 中文：沉淀新增模型的接入规范。
- English: Integration rules for adding new models.

不负责内容 / Non-goals:

- 中文：不负责数据集专属预处理、生产级训练流水线、超参搜索、复现每篇论文最优分数。
- English: It does not own dataset-specific preprocessing, production training pipelines, hyperparameter search, or best-score reproduction for every paper.
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
- 中文：registry 支持 `list_domains()`、`list_models(...)` 和 `get_model_info(...)`，返回的元数据副本可安全修改。
- English: The registry exposes `list_domains()`, `list_models(...)`, and `get_model_info(...)`; returned metadata snapshots are safe to modify.

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
& 'D:\Path\To\OtherPython\python.exe' -m pip install .\dist\torch_like-0.2.0-py3-none-any.whl
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
  .\dist\torch_like-0.2.0-py3-none-any.whl

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

English: `torch_like.automl` provides CSV profiling, `row/full_sequence/sliding_window` sampling, group CV, rolling-origin forecasting, sealed final tests, scikit-learn and selected `torch_like` candidates, preprocessing competition, isolated trials, and reproducible artifacts. See [`docs/AUTOML.md`](docs/AUTOML.md) for configuration, CLI, and Python APIs.

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
