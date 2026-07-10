# torch_like Development Guide / 开发指南

中文：本文面向开发者，说明如何维护 `torch_like` 的框架边界、任务接口、模型注册和测试规则。

English: This document is for developers. It explains how to maintain framework boundaries, task interfaces, model registration, and testing rules in `torch_like`.

## Scope / 范围

中文：`torch_like` 是模型框架，不是训练平台。

English: `torch_like` is a model framework, not a training platform.

Core responsibilities / 核心职责:

- 中文：定义可复用模型。
- English: Define reusable models.
- 中文：提供任务级 batch 和 base model contract。
- English: Provide task-level batch and base-model contracts.
- 中文：提供顶层导入和 registry 构造。
- English: Provide top-level imports and registry-based construction.
- 中文：保持模型 API 一致。
- English: Keep model APIs consistent.

Non-goals / 非目标:

- 中文：不把每个论文训练脚本并入核心库。
- English: Do not merge every paper training script into the core library.
- 中文：不负责每个数据集的专属预处理。
- English: Do not own dataset-specific preprocessing for every dataset.
- 中文：不追求所有论文最佳指标复现。
- English: Do not aim to reproduce every paper's best score.

## Framework Boundary / 框架边界

中文：判断一个文件是否应该进入 `torch_like/models/`，看它的主要职责是否是“定义可复用模型结构”。

English: A file belongs under `torch_like/models/` only if its main responsibility is defining reusable model architecture.

Decision rule / 判断规则:

- 中文：模型结构进 `torch_like/models/`。
- English: Model architectures go under `torch_like/models/`.
- 中文：训练循环进 `runner/`。
- English: Training loops go under `runner/`.
- 中文：通用数据协议进 `torch_like/data/`。
- English: Generic data protocols go under `torch_like/data/`.
- 中文：数据集专属解析留在 runner 或外部项目。
- English: Dataset-specific parsing stays in runners or external projects.
- 中文：说明性内容进 `docs/`。
- English: Explanatory content goes under `docs/`.

## Relation to `torch.nn` / 与 `torch.nn` 的关系

中文：`torch_like` 继承 `nn.Module` 的对象模型，但不复刻所有 PyTorch 层。

English: `torch_like` inherits the `nn.Module` object model, but does not clone every PyTorch layer.

Inherited behavior / 继承行为:

- `forward(...)`
- `state_dict()` / `load_state_dict(...)`
- `parameters()`
- `train()` / `eval()`
- `to()` / `cpu()` / `cuda()`

Framework-specific behavior / 框架特有行为:

- `predict(batch)`
- `fit_step(batch)`
- `score_batch(batch)`
- `next_token_logits(batch)`
- `make_forecast_batch(...)`
- `make_detection_batch(...)`
- `make_rul_batch(...)`
- `make_sequence_batch(...)`

## Project Layout / 项目布局

```text
torch_like/
  models/
    forecasting/
    detection/
    graph/
    sequence/
    vision/
    multimodal/
    generative/
    diagnosis/
    rul.py
    registry.py
  data/
  nn/
  utils/
runner/
tests/
docs/
examples/
```

中文：`models/` 是核心；`runner/` 是辅助；`tests/` 是合同验证；`docs/` 是双语说明。

English: `models/` is core; `runner/` is auxiliary; `tests/` validates contracts; `docs/` contains bilingual documentation.

## Model Registration / 模型注册

中文：新增模型必须接入对应 domain 的 `model_zoo`，并通过 lazy export 暴露。

English: New models must be added to the corresponding domain `model_zoo` and exposed through lazy exports.

Required locations / 必改位置:

- `torch_like/models/<domain>/__init__.py`
- `torch_like/models/__init__.py`
- `torch_like/__init__.py`
- `tests/test_registry.py`

Example / 示例:

```python
model_zoo = {
    "AGCRN": {
        "cls": AGCRN,
        "family": "graph_spatiotemporal",
        "layer": "Graph spatiotemporal forecasting",
        "integration_level": "framework_oriented",
        "supports": ["graph_mode", "graph_structure_learning"],
        "paper": "...",
        "description": "...",
        "complexity": "...",
    },
}
```

Metadata rules / 元数据规则:

- 中文：`family` 表示模型家族。
- English: `family` identifies the model family.
- 中文：`layer` 表示文档分层。
- English: `layer` controls documentation grouping.
- 中文：`supports` 表示能力标签。
- English: `supports` lists capability tags.
- 中文：`integration_level` 当前统一使用 `framework_oriented`。
- English: `integration_level` currently uses `framework_oriented`.

## Task Interfaces / 任务接口

### Forecasting / 预测

Base types / 基础类型:

- `ForecastingConfig`
- `ForecastingBatch`
- `ForecastingModel`

Contract / 合同:

- 中文：输入通常是 `(B, seq_len, enc_in)`。
- English: Input is usually `(B, seq_len, enc_in)`.
- 中文：输出通常是 `(B, pred_len, c_out)`。
- English: Output is usually `(B, pred_len, c_out)`.
- 中文：`predict(batch)` 调用模型并返回预测。
- English: `predict(batch)` calls the model and returns predictions.

Graph protocol / 图协议:

- `graph_mode="learned"`
- `graph_mode="prior"`
- `graph_mode="hybrid"`

中文：支持图的 forecasting 模型应尽量复用该协议。

English: Graph forecasting models should reuse this protocol whenever possible.

### Probabilistic Forecasting / 概率预测

Base types / 基础类型:

- `ProbabilisticForecastingConfig`
- `ProbabilisticForecastingBatch`
- `ProbabilisticForecastingModel`

Required methods / 必需方法:

- `predict_distribution(batch)`
- `sample(batch, num_samples=...)`
- `fit_step(batch)`

### Detection / 异常检测

Base types / 基础类型:

- `DetectionConfig`
- `DetectionBatch`
- `DetectionModel`

Contract / 合同:

- 中文：`fit_step(batch)` 返回标量训练损失。
- English: `fit_step(batch)` returns a scalar training loss.
- 中文：`score_batch(batch, reduction=...)` 返回异常分数。
- English: `score_batch(batch, reduction=...)` returns anomaly scores.
- 中文：默认重构/预测最后一个时间步。
- English: The default convention reconstructs or predicts the last timestamp.

### RUL / 剩余寿命预测

Base types / 基础类型:

- `RULConfig`
- `RULBatch`
- `RULModel`

Contract / 合同:

- 中文：输入 `(B, seq_len, enc_in)`。
- English: Input shape is `(B, seq_len, enc_in)`.
- 中文：输出 `(B,)`。
- English: Output shape is `(B,)`.

### Graph / 通用图学习

Base types / 基础类型:

- `GraphConfig`
- `GraphBatch`
- `GraphModel`

中文：该域用于节点任务、图任务、通用 GNN，不等同于 graph forecasting。

English: This domain is for node tasks, graph tasks, and general GNNs. It is not the same as graph forecasting.

### Sequence / 序列建模

Base types / 基础类型:

- `SequenceConfig`
- `SequenceBatch`
- `SequenceModel`
- `Seq2SeqBatch`
- `Seq2SeqModel`

中文：该域用于通用序列、长上下文、SSM、decoder-only 和 encoder-only 模型。

English: This domain is for general sequence, long-context, SSM, decoder-only, and encoder-only models.

Contract / 合同:

- 中文：`next_token_logits(batch)` 返回 next-token logits。
- English: `next_token_logits(batch)` returns next-token logits.
- 中文：因果模型的 `loss_step(batch)` 使用第 t 位 logits 预测第 t+1 位标签；BERT 式掩码语言模型不使用该右移约定。
- English: Causal `loss_step(batch)` uses logits at position t to predict labels at t+1; BERT-style masked language models do not use this shift.
- 中文：支持自回归生成的模型可以额外实现 `generate(...)`。
- English: Models that support autoregressive generation may additionally implement `generate(...)`.
- 中文：`GPT` 表示 GPT/GPT-2 风格 decoder-only LM；tokenizer、RLHF、采样策略库和预训练权重不属于模型类职责。
- English: `GPT` represents a GPT/GPT-2-style decoder-only LM; tokenizers, RLHF, sampling-policy libraries, and pretrained weights are outside the model class boundary.
- 中文：T5 这类 encoder-decoder 模型应使用 `Seq2SeqModel` 和 `make_seq2seq_batch(...)`，不要伪装成 decoder-only next-token 模型。
- English: Encoder-decoder models such as T5 should use `Seq2SeqModel` and `make_seq2seq_batch(...)`, not masquerade as decoder-only next-token models.

### Vision / 视觉模型

Base types / 基础类型:

- `VisionConfig`
- `VisionBatch`
- `VisionModel`

中文：该域用于图像分类、视觉 backbone 和后续视觉任务模型。

English: This domain is for image classification, vision backbones, and future vision task models.

## Current Model Families / 当前模型家族

Forecasting / 时间序列预测:

- Transformer / attention: `AST`, `Informer`, `Autoformer`, `FEDformer`, `LogTrans`, `PatchTST`, `ITransformer`, `Crossformer`, `TimeXer`, `DAG`
- Exogenous forecasting: `TFT`, `TiDE`
- MLP / decomposition / convolutional mixing: `MLCNN`, `NBEATS`, `NHITS`, `DLinear`, `SCINet`, `TimesNet`, `MICN`, `TimeMixer`, `TSMixer`, `ConvTimeNet`, `SOFTS`
- Dynamical systems: `Koopa`
- Frequency-domain forecasting: `FreTS`
- Graph spatiotemporal: `ASTGCN`, `AGCRN`, `D2STGNN`, `DMSTGCN`, `MTGNN`, `StemGNN`, `GraphWaveNet`, `DCRNN`, `FCGAGA`, `GTS`, `STFGNN`, `STGODE`
- Probabilistic: `DeepAR`, `TimeGrad`, `ProTran`, `PPM`

Detection / 异常检测:

- `RTDetector`
- `AnomalyTransformer`
- `MEMTO`
- `GDN`
- `MTADGAT`
- `DCDetector`

RUL / 剩余寿命:

- `InformerRUL`
- `AutoformerRUL`
- `LogTransRUL`
- `THGNN`

Graph / 通用图:

- `GCN`
- `GAT`
- `GNAN`
- `GraphSAGE`
- `GIN`
- `GraphTransformer`
- `SpikeGraphormer`

Sequence / 序列:

- `TransformerDecoder`
- `GPT`
- `LRU`
- `Mamba`
- `MambaLike`
- `Caracal`
- `BERT`
- `T5`

Vision / 视觉:

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

Multimodal / 多模态:

- `CLIP`

Multimodal output protocol / 多模态输出协议:

- Image-text dual encoders should expose `encode_image(...)`, `encode_text(...)`, and `predict(batch)`.
- `predict(batch)` may return a dict with embeddings and similarity logits.
- Tokenization, dataset pairing, large-scale pretraining, and retrieval evaluation belong outside the model class.

Generative / 生成:

- `DDPM`
- `VAE`
- `DCGAN`

Generative output protocol / 生成模型输出协议:

- Diffusion models should expose forward noising, noise prediction, loss, and sampling helpers.
- `DDPM.predict(batch)` returns predicted noise for `batch.x_start`/`batch.timesteps`-compatible tensors.
- `DDPM.sample(..., num_steps=num_timesteps)` uses ancestral DDPM sampling; reduced `num_steps` uses deterministic DDIM updates.
- Latent-variable models should expose encode/decode/reparameterization helpers and keep ELBO-style losses inside `loss_step(batch)`.
- Adversarial models should expose generator/discriminator modules and adversarial loss helpers, but not own optimizer stepping order.
- Dataset transforms, FID/IS metrics, checkpoint sampling grids, and long-running training loops belong outside the model class.

Vision output protocol / 视觉输出协议:

- Classification / 分类: return logits `(batch, num_classes)`.
- Segmentation / 分割: return dense logits `(batch, num_classes, height, width)`.
- Object detection / 目标检测: return a dict with `pred_logits` and `pred_boxes`; do not move matcher/loss/COCO evaluation into the model class.

## Adding a New Model / 新增模型流程

1. 中文：确认模型属于哪个 domain。
   English: Identify the correct domain.
2. 中文：新增模型文件并继承对应 base class。
   English: Add a model file and inherit the proper base class.
3. 中文：保证模型能直接 `model(...)` 调用。
   English: Ensure the model can be called directly via `model(...)`.
4. 中文：实现对应任务的 helper 方法或复用 base class 默认实现。
   English: Implement task helper methods or reuse base defaults.
5. 中文：接入 domain `__init__.py` 的 `model_zoo`。
   English: Register it in the domain `model_zoo`.
6. 中文：接入 `torch_like.models` 和 `torch_like` 顶层 lazy exports。
   English: Add lazy exports in `torch_like.models` and top-level `torch_like`.
7. 中文：补测试。
   English: Add tests.
8. 中文：同步双语文档。
   English: Update bilingual documentation.

## Forecast Data Protocol / 预测数据协议

中文：`ForecastingBatch` 支持传统单输入和外生变量双输入。

English: `ForecastingBatch` supports both legacy single-input forecasting and target-plus-exogenous inputs.

Fields / 字段:

- `x_enc`
- `x_mark_enc`
- `x_dec`
- `x_mark_dec`
- `x_target_enc`
- `x_exog_enc`
- `x_target_dec`
- `x_exog_dec`

中文：只有真正需要目标变量/外生变量分离的模型才应读取 `x_target_*` 和 `x_exog_*`。

English: Only models that truly need target/exogenous separation should consume `x_target_*` and `x_exog_*`.

Validation rules / 校验规则:

- 中文：`seq_len`、`pred_len`、batch size 和 stride 必须为正数，`label_len` 必须位于 `[0, seq_len]`。
- English: `seq_len`, `pred_len`, batch size, and stride must be positive; `label_len` must be in `[0, seq_len]`.
- 中文：NumPy 和图预测数据必须足以产生非空 train/val/test 窗口，不能包含 NaN 或无穷值。
- English: NumPy and graph forecasting data must produce non-empty train/val/test windows and contain no NaN or infinite values.
- 中文：目标变量和外生变量索引必须唯一、范围有效且互不重叠；三维图数据的索引作用于最后一个 feature 轴。
- English: Target and exogenous indices must be unique, in range, and disjoint; for 3D graph data they address the final feature axis.
- 中文：三维图数据完成特征切分后，将 `(nodes, features)` 展平为 forecasting 通道；节点邻接矩阵同步扩展为通道级邻接矩阵。
- English: After feature splitting, 3D graph data flattens `(nodes, features)` into forecasting channels and expands node adjacency to channel-level adjacency.
- 中文：验证和测试 loader 不随机打乱、不丢弃尾批；训练集不足一个 batch 时也保留该 batch。
- English: Validation/test loaders are sequential and keep tail batches; a training split smaller than one batch is retained.

## Runner Contract / Runner 合同

中文：runner 是辅助层，不定义框架能力边界。

English: Runners are auxiliary and do not define the framework boundary.

中文：runner checkpoint 使用显式 `weights_only=True` 安全加载，并在恢复前校验 payload 和 `model_state_dict`。本地数据路径通过 CLI 参数或环境变量传入，不写死开发机目录。

English: Runner checkpoints load explicitly with `weights_only=True` and validate the payload and `model_state_dict`. Local dataset paths come from CLI arguments or environment variables, not machine-specific constants.

中文：所有公共 runner 配置都提供可选 `results_dir`；测试和嵌入式调用应传入隔离目录，CLI 可使用 `--results-dir`。

English: All public runner configs expose an optional `results_dir`; tests and embedded callers should use an isolated directory, while CLIs accept `--results-dir`.

Runner may / runner 可以:

- 中文：加载数据。
- English: Load data.
- 中文：训练一个或多个 epoch。
- English: Train for one or more epochs.
- 中文：输出简单指标。
- English: Report simple metrics.
- 中文：保存比较结果。
- English: Save comparison results.

Runner must not / runner 不应:

- 中文：定义模型结构。
- English: Define model architecture.
- 中文：绕过模型任务接口。
- English: Bypass model task interfaces.
- 中文：把数据集假设写回核心模型。
- English: Push dataset assumptions back into core models.

## Testing / 测试

Run / 运行:

```powershell
& 'D:\Project\base\base\Scripts\python.exe' -m unittest discover -s tests
& 'D:\Project\base\base\Scripts\python.exe' -m ruff check torch_like runner tests examples tools
& 'D:\Project\base\base\Scripts\python.exe' -m pytest --cov=torch_like --cov-report=term-missing
```

Minimum tests for a new model / 新模型最低测试:

- 中文：直接实例化。
- English: Direct instantiation.
- 中文：`from_config` 或 registry 构造。
- English: `from_config` or registry construction.
- 中文：`model.config()` 必须保留完整构造参数，使 `type(model).from_config(model.config())` 能复现相同结构。
- English: `model.config()` must preserve complete constructor arguments so `type(model).from_config(model.config())` reproduces the same architecture.
- 中文：输出 shape。
- English: Output shape.
- 中文：顶层导出。
- English: Top-level export.
- 中文：如果是 detection，测试 `fit_step` 和 `score_batch`。
- English: For detection models, test `fit_step` and `score_batch`.

## Documentation Rule / 文档规则

中文：项目说明文档必须保持中英文双语。

English: Project documentation must remain bilingual.

Required style / 要求:

- 中文：标题使用 `中文 / English` 或 `English / 中文`。
- English: Headings should use `Chinese / English` or `English / Chinese`.
- 中文：核心段落应包含中英文说明。
- English: Core paragraphs should include both Chinese and English explanations.
- 中文：代码示例不需要重复翻译。
- English: Code examples do not need duplicated translations.

## Notes on Fidelity / 论文一致性说明

中文：`torch_like` 的模型实现以框架可用性和接口一致性为目标，通常是 paper-aligned 或 framework-oriented，不承诺逐行复刻官方代码。

English: Implementations in `torch_like` prioritize framework usability and API consistency. They are usually paper-aligned or framework-oriented, not line-by-line official reproductions.

## Boundary Notes / 边界备注

中文：如果某个新模型需要大量训练策略、数据集假设或外部依赖才能成立，应优先作为候选记录，而不是立即进入核心库。

English: If a new model depends heavily on training strategies, dataset assumptions, or external dependencies, it should first be recorded as a candidate instead of entering the core library immediately.
