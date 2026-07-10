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

### Detection / 异常检测

- `RTDetector`
- `AnomalyTransformer`
- `MEMTO`
- `GDN`
- `MTADGAT`
- `DCDetector`

### Graph / 通用图学习

- `GCN`
- `GAT`
- `GNAN`
- `GraphSAGE`
- `GIN`
- `GraphTransformer`
- `SpikeGraphormer`

### Sequence / Long-Context Modeling / 序列与长上下文建模

- `TransformerDecoder`
- `GPT`
- `LRU`
- `Mamba`
- `MambaLike`
- `Caracal`
- `BERT`
- `T5`

### Vision / 视觉模型

- `ResNet`
- `ViT`
- `DeiT`
- `UNet`
- `DeepLabV3`
- `DETR`
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

### Generative / 生成模型

- `DDPM`
- `VAE`
- `DCGAN`

## 安装 / Installation

```powershell
& 'D:\Project\base\base\Scripts\python.exe' -m pip install -e .
```

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
from torch_like.models import create_model

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

## 辅助 Runner / Auxiliary Runners

中文：`runner/` 用于快速验证模型能否实例化、训练一步、评估和比较；它不是框架核心。

English: `runner/` helps validate instantiation, one-step training, evaluation, and comparison. It is not the core framework.

Common entry points / 常用入口:

- `runner/forecast_runner.py`
- `runner/forecast_comparison_runner.py`
- `runner/detection_runner.py`
- `runner/rul_runner.py`
- `runner/benchmark_compile.py`

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
& 'D:\Project\base\base\Scripts\python.exe' -m pytest --cov=torch_like --cov-report=term-missing
```

中文：测试覆盖直接导入、registry 构造、任务 batch、预测输出形状、检测分数接口和 runner 基础行为。

English: Tests cover direct imports, registry construction, task batches, prediction shapes, detection scoring APIs, and basic runner behavior.

## 当前状态 / Current Status

中文：框架已具备多任务模型库骨架，重点覆盖 forecasting、probabilistic forecasting、detection、RUL、graph、sequence、vision、multimodal 和 generative。后续新增模型应优先保持 API 一致性，而不是追求实验脚本数量。

English: The framework now provides a multi-task model-library skeleton covering forecasting, probabilistic forecasting, detection, RUL, graph, sequence, vision, multimodal, and generative domains. Future additions should prioritize API consistency over accumulating experiment scripts.
