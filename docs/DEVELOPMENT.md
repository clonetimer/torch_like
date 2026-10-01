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
    <domain>/
      __init__.py      # lazy facade / 按需公共入口
      _catalog.py      # private registry metadata / 私有注册元数据
      common.py        # task contracts / 任务合同
      <model>.py       # implementation / 模型实现
    rul.py
    registry.py
  classical/           # native sklearn compatibility / sklearn 原生兼容层
  data/
  nn/
  automl/
  utils/
runner/
tests/
docs/
examples/
```

中文：`models/` 是核心；`runner/` 是辅助；`tests/` 是合同验证；`docs/` 是双语说明。

English: `models/` is core; `runner/` is auxiliary; `tests/` validates contracts; `docs/` contains bilingual documentation.

## Model Registration / 模型注册

中文：新增模型必须同时接入实现文件、私有 domain catalog 和 lazy export 映射。`__init__.py` 不再保存完整 catalog，也不得强导入 domain 内全部模型。

English: New models must be wired into the implementation file, private domain catalog, and lazy export map. `__init__.py` no longer owns the full catalog and must not eagerly import every model in the domain.

Required locations / 必改位置:

- `torch_like/models/<domain>/<model>.py`
- `torch_like/models/<domain>/_catalog.py`
- corresponding contract and registry tests / 对应合同与 registry 测试

然后运行 / Then run:

```bash
python tools/generate_lazy_exports.py
python tools/generate_lazy_exports.py --check
```

该工具会统一生成以下三个轻量索引，避免人工维护多份导入表：

- `torch_like/models/_domain_exports.py`
- `torch_like/models/_public_exports.py`
- `torch_like/models/_builtin_index.py`

The generator owns the three lightweight indexes above so contributors do not manually synchronize multiple import maps. These generated files contain only string paths and static metadata and must never import model implementations during normal package import. Legacy root-level model shortcuts are derived from `torch_like.models.MODEL_EXPORTS`.

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

- 中文：内置 catalog 的八个字段 `cls`、`family`、`layer`、`integration_level`、`supports`、`paper`、`description`、`complexity` 全部必填。
- English: Built-in catalogs must provide all eight fields: `cls`, `family`, `layer`, `integration_level`, `supports`, `paper`, `description`, and `complexity`.
- 中文：`integration_level` 是 schema v1 兼容字段。registry 会根据 `torch_like/models/catalog_schema_v2.json` 派生 `implementation_fidelity`、`implementation_scope`、`execution_profile`、`maturity`、`benchmark_evidence` 和 `evidence_record`；不得自行创造未映射复合值。任何 maturity 或 benchmark 晋升都必须提供 evidence record。
- English: `integration_level` is a schema-v1 compatibility field. The registry derives `implementation_fidelity`, `implementation_scope`, `execution_profile`, `maturity`, `benchmark_evidence`, and `evidence_record` from `torch_like/models/catalog_schema_v2.json`. Do not invent unmapped compound values. Any maturity or benchmark promotion requires an evidence record.
- 中文：`family` 表示模型家族。
- English: `family` identifies the model family.
- 中文：`layer` 表示文档分层。
- English: `layer` controls documentation grouping.
- 中文：`supports` 表示能力标签。
- English: `supports` lists capability tags.
- 中文：`integration_level` 当前统一使用 `framework_oriented`。
- English: `integration_level` currently uses `framework_oriented`.
- 中文：`register_model_catalog(...)` 会先验证完整 catalog，再原子写入；不要直接修改内部 `MODEL_ZOO`。
- English: `register_model_catalog(...)` validates the complete catalog before an atomic write; do not mutate internal `MODEL_ZOO` directly.

## Classical sklearn Compatibility / sklearn 兼容层

中文：`torch_like.classical` 是 sklearn 的发现/构造入口，不是 sklearn 的重新实现。新增兼容 estimator 时，只登记其真实 sklearn import path、任务类型与 family；不要复制 sklearn 源码，也不要创建 `nn.Module` 包装器。

English: `torch_like.classical` is a discovery/construction surface for sklearn, not a reimplementation. Add compatibility entries by registering the real sklearn import path, task kind, and family; do not copy sklearn source or wrap estimators as `nn.Module`.

```python
from torch_like.classical import create_estimator

model = create_estimator("RandomForestClassifier", n_estimators=200)
```

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
- 中文：`LLaMA` 表示现代 decoder-only LLM 结构；RoPE、RMSNorm、SwiGLU 和 GQA 属于模型结构，tokenizer、chat template、权重转换和推理服务不属于模型类职责。
- English: `LLaMA` represents a modern decoder-only LLM architecture; RoPE, RMSNorm, SwiGLU, and GQA belong in the model, while tokenizers, chat templates, checkpoint conversion, and serving policy remain outside model classes.
- 中文：`Mixtral` 表示稀疏 MoE decoder-only LLM；top-k router、experts 和 router logits 属于模型结构，专家并行、负载均衡训练策略、量化和服务调度不属于模型类职责。
- English: `Mixtral` represents a sparse MoE decoder-only LLM; the top-k router, experts, and router logits belong in the model, while expert parallelism, load-balancing training policy, quantization, and serving schedulers remain outside model classes.
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
- MLP / decomposition / convolutional mixing: `MLCNN`, `NBEATS`, `NHITS`, `DLinear`, `SCINet`, `TimesNet`, `MICN`, `TimeMixer`, `TSMixer`, `ConvTimeNet`, `ModernTCN`, `SOFTS`
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
- `OmniAnomaly`
- `TranAD`
- `USAD`

RUL / 剩余寿命:

- `InformerRUL`
- `AutoformerRUL`
- `LogTransRUL`
- `THGNN`

Graph / 通用图:

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

Sequence / 序列:

- `S4`
- `Hyena`

- `TransformerDecoder`
- `TransformerXL`
- `Performer`
- `Reformer`
- `RetNet`
- `KimiLinear` (`paper_aligned_reference`; portable KDA scan, no bundled FlashKDA kernel or 48B checkpoint)
- `xLSTM` (`paper_aligned_mlstm_reference`; matrix-memory path only, no sLSTM or official CUDA kernels)
- `GPT`
- `LLaMA`
- `Mixtral`
- `LRU`
- `Mamba`
- `Mamba2`
- `MambaLike`
- `Caracal`
- `BERT`
- `T5`

Vision / 视觉:

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
- `RTDETR` (`paper_aligned_portable_subset`; RT-DETR AIFI hybrid encoder and confidence-based query selection)
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

Multimodal / 多模态:

- `CLIP`
- `SigLIP`
- `BLIP`
- `BLIP2`
- `LLaVA`

Diagnosis / 故障诊断:

- `InceptionTime`
- `ROCKET`

Multimodal output protocol / 多模态输出协议:

- Image-text dual encoders should expose `encode_image(...)`, `encode_text(...)`, and `predict(batch)`.
- Self-supervised vision models may expose `fit_step(batch)` and reconstruction outputs while retaining `VisionBatch`.
- Paper checkpoints and external tokenizers remain optional integrations, not implicit model-construction side effects.
- `predict(batch)` may return a dict with embeddings and similarity logits.
- Tokenization, dataset pairing, large-scale pretraining, and retrieval evaluation belong outside the model class.
- Visual instruction models such as `LLaVA` may expose `encode_image_tokens(...)`, projector helpers, causal `lm_logits`, and `generate(...)`; tokenizer, chat template, external LLM checkpoint mapping, and serving policy remain outside model classes.
- Q-Former bridge models such as `BLIP2` may expose query features, language-prefix projections, image-text matching logits, and caption logits while keeping external LLM checkpoints, tokenizers, prompt templates, and pretraining schedules outside model classes.

Generative / 生成:

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

Tabular / 表格:

- `TabNet`
- `NODE`
- `FTTransformer`

Recommendation / 推荐系统:

- `NCF`
- `DeepFM`
- `DCN`
- `BERT4Rec`

Audio / 音频与语音:

- `Wav2Vec2`
- `EnCodec` (`paper_aligned_portable_subset`; bandwidth-scalable RVQ without pretrained weights or entropy model)
- `Conformer`
- `HuBERT`
- `Whisper`
- `AudioSpectrogramTransformer`

Generative output protocol / 生成模型输出协议:

- Diffusion models should expose forward noising, noise prediction, loss, and sampling helpers.
- `DDPM.predict(batch)` returns predicted noise for `batch.x_start`/`batch.timesteps`-compatible tensors.
- Diffusion Transformer models such as `DiT` should keep the same diffusion helper surface while changing only the denoiser architecture.
- Latent diffusion models such as `LatentDiffusion` should expose image encode/decode helpers, latent-space diffusion helpers, optional conditioning inputs, and image sampling while keeping tokenizers, external checkpoint loading, guidance policy, and scheduler libraries outside the model class.
- Controlled diffusion models such as `ControlNet` should expose control encoding and control residual helpers while keeping edge/depth/pose annotators, ControlNet checkpoint conversion, and prompt policy outside the model class.
- Flow-matching models such as `RectifiedFlow` should expose interpolation, target velocity, velocity prediction, flow-matching loss, and sampling helpers while keeping external ODE solvers, guidance policy, and pretrained checkpoint loading outside the model class.
- Flow models such as `RealNVP` should expose exact `log_prob(...)`, `inverse(...)`, `sample(...)`, and negative-likelihood `loss_step(batch)`.
- Autoregressive image models such as `PixelCNN` should expose discrete `log_prob(...)`, sequential `sample(...)`, and cross-entropy `loss_step(batch)`.
- Discrete latent models such as `VQVAE` should expose `encode(...)`, `quantize(...)`, `decode(...)`, `sample(...)`, and reconstruction plus codebook `loss_step(batch)`.
- `DDPM.sample(..., num_steps=num_timesteps)` uses ancestral DDPM sampling; reduced `num_steps` uses deterministic DDIM updates.
- Latent-variable models should expose encode/decode/reparameterization helpers and keep ELBO-style losses inside `loss_step(batch)`.
- Adversarial models should expose generator/discriminator modules and adversarial loss helpers, but not own optimizer stepping order.
- Dataset transforms, FID/IS metrics, checkpoint sampling grids, and long-running training loops belong outside the model class.

Tabular protocol / 表格协议:

- Tabular models should use `TabularBatch(x_cont, x_cat=None, y=None)`, `predict(batch)`, and `loss_step(batch)`.
- Classification uses cross-entropy when `output_dim > 1` and integer labels are provided; regression uses MSE.
- Feature preprocessing, missing-value imputation, categorical vocabulary fitting, and benchmark splits belong outside model classes.

Recommendation protocol / 推荐协议:

- Recommendation models should use `RecommendationBatch(user_ids=None, item_ids=None, x_cat=None, x_cont=None, y=None)`.
- Collaborative-filtering models such as `NCF` consume `user_ids` and `item_ids`.
- CTR models such as `DeepFM` and `DCN` consume categorical fields and optional continuous fields.
- Sequential models such as `SASRec` and `BERT4Rec` consume item-history sequences through `RecommendationBatch.x_cat`.
- Negative sampling, candidate generation, feature store lookup, online serving, and ranking metrics belong outside model classes.

Audio protocol / 音频协议:

- Audio models should use `AudioBatch(x, lengths=None, y=None)`, `predict(batch)`, and `loss_step(batch)`.
- Raw-waveform models such as `Wav2Vec2` consume `(batch, time)` or `(batch, channels, time)`.
- Neural codecs such as `EnCodec` should expose `encode(...)`, `decode(...)`, bandwidth selection, discrete codes, and reconstruction `loss_step(batch)` while keeping entropy coding and adversarial training orchestration outside the model class.
- Masked hidden-unit models such as `HuBERT` may expose `extract_features(...)`, `mask_features(...)`, `predict_units(...)`, and `loss_step(batch)`; offline clustering and hidden-unit label generation remain outside model classes.
- Acoustic-sequence models such as `Conformer` consume `(batch, time, feature_dim)`.
- Encoder-decoder ASR models such as `Whisper` consume log-mel tensors and decoder token ids, and may expose `encode_audio(...)`, `decode_tokens(...)`, `generate(...)`, and teacher-forcing `loss_step(batch)`.
- Spectrogram classifiers such as `AudioSpectrogramTransformer` consume `(batch, channels, freq, time)`.
- Feature extraction pipelines, tokenizers, CTC/beam decoding, augmentation, and ASR metrics belong outside model classes.

Vision output protocol / 视觉输出协议:

- Classification / 分类: return logits `(batch, num_classes)`.
- Segmentation / 分割: return dense logits `(batch, num_classes, height, width)`.
- Universal segmentation / 通用分割: query-mask models such as `Mask2Former` may return `pred_logits`, `pred_masks`, and composed `semantic_logits`; target assignment, matching, and panoptic fusion remain outside model classes.
- Object detection / 目标检测: return a dict with `pred_logits` and `pred_boxes`; do not move matcher/loss/COCO evaluation into the model class.
- Open-vocabulary detection / 开放词表检测: language-grounded detectors such as `GroundingDINO` may return `token_logits`, `objectness`, and normalized `pred_boxes`; tokenizers, phrase parsing, NMS, Grounded-SAM orchestration, and benchmark evaluation remain outside model classes.
- Promptable segmentation / 提示分割: `SAM`-style models should expose `encode_image(...)`, `encode_prompts(...)`, `predict_masks(...)`, and `forward(..., points=..., boxes=..., masks=...)`; checkpoint loading, annotation UI, video memory, and training recipes remain outside the model class.

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
6. 中文：接入 `torch_like.models` 的 lazy export；`torch_like` 顶层入口会自动同步。
   English: Add the lazy export to `torch_like.models`; the top-level `torch_like` shortcut synchronizes automatically.
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

## 性能实现规范 / Performance Implementation Rules

中文：性能优化必须优先组合 PyTorch 的公共原语，并保持模型输出协议稳定。不要仅为可能的速度收益引入未经基准验证的自定义算子。

English: Performance work should compose public PyTorch primitives while preserving model output contracts. Do not add custom operators for speculative speedups without benchmarks.

- 中文：标准点积注意力优先使用 `torch.nn.functional.scaled_dot_product_attention`；只有需要返回权重或算法语义不同的注意力保留显式实现。
- English: Standard dot-product attention should prefer `torch.nn.functional.scaled_dot_product_attention`; explicit implementations remain for returned weights or genuinely different algorithms.
- 中文：顶层模型编译优先调用 `nn.Module.compile()`，旧版 PyTorch 才回退到 `torch.compile(model, ...)`。
- English: Top-level compilation should prefer `nn.Module.compile()`, falling back to `torch.compile(model, ...)` for older PyTorch versions.
- 中文：AMP、GradScaler、异步搬运和梯度清零属于 runner 执行策略，不得写入模型构造参数或 registry 元数据。
- English: AMP, GradScaler, asynchronous transfers, and gradient clearing are runner execution policies, not model constructor arguments or registry metadata.
- 中文：任何性能改造至少需要数值一致性测试、默认配置 smoke test 和目标设备基准；不能用单次首轮延迟评价 compile。
- English: Every performance change needs numerical-consistency tests, a default-config smoke test, and a target-device benchmark; compile must not be judged by first-call latency.
- 中文：KV cache 会改变生成状态协议，应作为独立功能设计，不与无状态注意力优化混合提交。
- English: KV caching changes the generation-state protocol and should be designed separately from stateless attention optimization.

References / 参考:

- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch Scaled Dot Product Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)

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
& 'D:\Project\base\base\Scripts\python.exe' -m ruff format --check torch_like runner tests examples tools
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
- 中文：catalog 元数据字段完整且 capability tag 不重复。
- English: Catalog metadata is complete and capability tags contain no duplicates.
- 中文：如果是 detection，测试 `fit_step` 和 `score_batch`。
- English: For detection models, test `fit_step` and `score_batch`.

## Package Build Gate / 包构建门禁

中文：任何影响顶层导出、包发现、依赖或版本号的改动，都必须重新构建 wheel 并执行隔离安装验收。构建产物不得包含 runner、测试、示例或临时数据。

English: Changes to top-level exports, package discovery, dependencies, or versions must rebuild the wheel and pass isolated installation verification. Distribution artifacts must not include runners, tests, examples, or temporary data.

```powershell
& .\tools\build_package.ps1 -Python 'D:\Project\base\base\Scripts\python.exe'
& .\tools\verify_package_install.ps1 -Python 'D:\Project\base\base\Scripts\python.exe'
```

中文：详细流程见 [Packaging Guide / 打包安装指南](PACKAGING.md)。

English: See the [Packaging Guide / 打包安装指南](PACKAGING.md) for the complete workflow.

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

## Model Governance Workflow / 模型治理流程

中文：模型收录采用“发现、硬门槛、正式评分、决策、实现验收、定期复审”六阶段流程。`governance/model_admission_policy.json` 是阈值、证据要求和复审期限的唯一机器可读依据；修改政策必须同步提升 `policy_version`，并重新验证全部候选。

English: Model admission follows six stages: discovery, hard gates, formal scoring, decision, implementation acceptance, and periodic review. `governance/model_admission_policy.json` is the sole machine-readable source for thresholds, evidence requirements, and review deadlines. Policy changes must bump `policy_version` and revalidate every candidate.

State transitions / 状态转换:

1. 中文：新发现只能先进入 `Candidate Record`，不得直接实现并导出。
   English: A new discovery must first enter `Candidate Record`; it may not jump directly to implementation and export.
2. 中文：六项硬门槛全部通过后才能进入评分表；失败项不得获得形式化分数。
   English: Formal scoring is allowed only after all six hard gates pass; failed candidates receive no formal score.
3. 中文：评分决策只能是 `core`、`observation` 或 `rejected`，由政策阈值自动计算。
   English: Scored decisions are limited to `core`, `observation`, or `rejected` and are computed from policy thresholds.
4. 中文：`core` 才默认进入实现队列；`observation` 需要解决“Missing information”并重新评分，不能仅凭维护者偏好升级。
   English: Only `core` enters the implementation queue by default. An `observation` must resolve its missing information and be rescored; maintainer preference alone cannot promote it.
5. 中文：实现完成必须登记真实 registry 名称和 `integration_level`，并通过导入、构造、前向、梯度、关键数学性质和边界条件测试。
   English: An accepted implementation must record its real registry name and `integration_level` and pass import, construction, forward, gradient, key mathematical-property, and boundary-condition tests.
6. 中文：每个候选都必须在复审台账中登记负责人、截止日期和提前触发条件；过期结论使 CI 失败关闭。
   English: Every candidate must record an owner, deadline, and early-review trigger. Expired decisions fail closed in CI.

Registry bypass prevention / 注册绕过防护:

- 中文：`governance/model_registry_baseline.txt` 是治理启用时的内置模型快照，不是推荐名单。
- English: `governance/model_registry_baseline.txt` is the built-in model snapshot at governance activation, not a recommendation list.
- 中文：不要为了让 CI 通过而直接把新模型追加到基线。新模型必须先获候选池批准；验证器会把候选池中的 `implemented` registry 声明作为基线外新增项的唯一授权。
- English: Never append a new model to the baseline merely to make CI pass. It must first be approved in the candidate pool; the validator treats an `implemented` registry claim there as the only authorization for an addition beyond the baseline.

Validation / 验证:

```powershell
& 'D:\Project\base\base\Scripts\python.exe' tools/validate_candidate_pool.py
& 'D:\Project\base\base\Scripts\python.exe' tools/validate_candidate_pool.py --as-of 2026-10-15
```

中文：第一条按当前日期检查；第二条可在合并前预演未来复审到期情况。网络链接的内容真实性仍需人工或检索工具复核，链接数量通过不等于证据质量通过。

English: The first command checks against today; the second can rehearse upcoming review expiry before merge. Source authenticity still requires human or research-tool review; passing a link count does not establish evidence quality.

## Public API Stability / 公共 API 稳定性

中文：`torch_like`、`torch_like.models`、`torch_like.nn`、`torch_like.data` 和 `torch_like.automl` 的 `__all__` 导出、符号身份和调用签名由 `governance/public_api_baseline.json` 保护。兼容新增后运行 `tools/audit_public_api.py --write`；破坏性变化必须先登记 `governance/api_deprecations.json`，保留旧入口并发出 `FutureWarning`。详细周期见 [`API_STABILITY.md`](API_STABILITY.md)。

English: `__all__` exports, symbol identity, and call signatures in `torch_like`, `torch_like.models`, `torch_like.nn`, `torch_like.data`, and `torch_like.automl` are protected by `governance/public_api_baseline.json`. Run `tools/audit_public_api.py --write` after compatible additions. Breaking changes must first enter `governance/api_deprecations.json`, retain the old entry point, and emit `FutureWarning`. See [`API_STABILITY.md`](API_STABILITY.md) for the lifecycle.

### Model evidence / 模型证据

中文：内置模型不得直接在 domain catalog 中自行声明成熟度。真实基准登记在 `torch_like/models/model_evidence.json`，由注册器合并并由 `tools/audit_model_catalog.py` 校验模型归属、证据等级及配置/报告路径。新增或晋升证据前阅读 [`MODEL_EVIDENCE.md`](MODEL_EVIDENCE.md)。

English: Built-in domain catalogs must not self-assign maturity. Real benchmarks are registered in `torch_like/models/model_evidence.json`, merged by the registry, and validated by `tools/audit_model_catalog.py` for model ownership, evidence levels, and config/report paths. Read [`MODEL_EVIDENCE.md`](MODEL_EVIDENCE.md) before adding or promoting evidence.

### AutoML protocol trust / AutoML 协议可信性

中文：新增 split 类型必须实现并通过 `torch_like.automl.audit_split_protocol` 的硬校验，不能只依赖配置层推断“应该没有泄漏”。运行前审计产物为 `protocol_audit.json`；测试至少覆盖有效协议和一个人为构造的泄漏协议。

English: Every new split type must implement and pass the hard checks in `torch_like.automl.audit_split_protocol`; configuration-level assumptions are insufficient. The pre-training audit artifact is `protocol_audit.json`, and tests must cover both a valid protocol and one deliberately leaked protocol.

### AutoML evaluation trust / AutoML 评估可信性

中文：排行榜信任等级由 `trust.level` 声明。`development` 保持快速单种子兼容；`validated` 必须至少 3 个评估种子，并通过 `torch_like.automl.audit_evaluation_trust`。新增候选规划逻辑必须保持每个候选获得完整种子覆盖、唯一 trial id、统一单 trial 超时和统一 PyTorch 训练预算；运行产物必须包含 `evaluation_trust_audit.json`。

English: Leaderboard trust is declared through `trust.level`. `development` keeps fast single-seed compatibility; `validated` requires at least three evaluation seeds and must pass `torch_like.automl.audit_evaluation_trust`. New candidate-planning logic must preserve full seed coverage for every candidate, unique trial ids, uniform per-trial timeout, and a uniform PyTorch training budget. Runs must include `evaluation_trust_audit.json`.

### AutoML example governance / AutoML 示例治理

中文：新增或修改 `examples/*.yaml` 后必须运行 `python tools/validate_automl_examples.py`。该脚本只做配置层校验，不读取外部数据集；它检查 task_id 唯一性、`validated` 与 `benchmark` 命名语义、三种子要求、显式算法预算覆盖和基础产物目录。CI 会执行同一脚本。

English: After adding or changing `examples/*.yaml`, run `python tools/validate_automl_examples.py`. The script validates configuration structure without reading external datasets. It checks unique task ids, `validated` and `benchmark` naming semantics, the three-seed requirement, explicit-algorithm budget coverage, and a basic artifact directory contract. CI runs the same script.

### AutoML run artifact governance / AutoML 运行产物治理

中文：完成排行榜运行后，使用 `python tools/validate_automl_runs.py runs/<task_id>` 审计实际产物。该脚本检查核心 JSON/CSV/报告文件、协议审计状态、信任等级一致性、trial 数量、封存测试状态、排行榜中是否泄露 `test_*` 指标，以及 `validated` 运行的完整种子覆盖。CI 使用 `--allow-empty --skip-legacy runs` 固定入口并跳过旧协议历史产物；本地完成的当前契约运行应指定具体 run 目录。

English: After a leaderboard run finishes, audit the artifacts with `python tools/validate_automl_runs.py runs/<task_id>`. The script checks core JSON/CSV/report files, protocol audit status, trust-level consistency, trial count, sealed-test status, absence of leaked `test_*` metrics in the leaderboard, and full seed coverage for `validated` runs. CI uses `--allow-empty --skip-legacy runs` to keep the entry point active while ignoring pre-trust-audit historical runs; local completed current-contract runs should pass an explicit run directory.

- 新增可选参数兼容；新增必填参数不兼容。 / Adding optional parameters is compatible; adding required parameters is not.
- 删除参数、修改默认值、改变参数 kind 或替换符号均视为破坏性变化。 / Removing parameters, changing defaults or parameter kind, and replacing symbols are breaking changes.
- 弃用至少跨两个 minor 版本和 90 天，删除时台账状态必须为 `removed`。 / Deprecation spans at least two minor releases and 90 days, and the ledger status must be `removed` at deletion.
- 不要通过手工编辑生成基线绕过 transition validator。 / Never manually edit the generated baseline to bypass transition validation.

## AutoML Boundary / 自动适配边界

### Task Adapter Governance / 任务适配器治理

中文：把已知骨干改造成新任务输出头，不等同于引入一个新的论文模型。此类实现必须登记在
`governance/task_adapter_approvals.json`，记录来源、目标任务、适配价值、风险、测试和复审日期。
候选池校验器会自动拒绝既不在历史 registry 基线、又没有候选池批准或任务适配器批准的注册项。

English: Adapting a known backbone to a new task head is distinct from admitting a new paper model.
Such adapters must be recorded in `governance/task_adapter_approvals.json` with source, target task,
rationale, risks, tests, and review deadline. The candidate-pool validator rejects registry additions
that are neither grandfathered, candidate-approved, nor explicitly adapter-approved.

RUL adapters must inherit `RULModel`, return one scalar per sample, and support `RULBatch`, `predict`,
and `loss_step`. Dataset-specific discovery and label derivation belong in `torch_like.data` or
`torch_like.automl.data`, never in model classes.

中文：`torch_like.automl` 比较的是“预处理器 + 特征策略 + 算法 + 超参数”完整管线。数据划分必须在搜索前固定，输入和目标预处理器只能在 inner training 数据拟合，排行榜只能按 validation 指标排序。所有划分协议的测试数据均不得传入候选进程，只能在最终管线选定并重训后评估一次。

English: `torch_like.automl` compares complete preprocessing, feature, algorithm, and hyperparameter pipelines. Splits must be fixed before search, input and target preprocessors fit training data only, and rankings use validation metrics. Test data under every split protocol must stay outside candidate processes and be evaluated once after final selection and retraining.

Rules / 规则:

- Built-in PyTorch models remain in the model registry; sklearn estimators are backend candidates and do not pretend to be `nn.Module` models.
- Trial failures must be serialized and isolated from the remaining search.
- Resume must verify config, dataset fingerprint/profile, split, and candidate plan before reusing terminal trials.
- A completed sealed-test record is immutable and must not trigger another final retraining during resume.
- Multi-seed runs use one isolated trial per pipeline/seed pair; incomplete seed coverage cannot enter ranking.
- `max_trials` counts pipeline configurations. Resource planning must multiply it by the number of evaluation seeds.
- Overlapping windows may not use random/stratified splitting.
- Forecasting targets may not overlap across windows in v1; use `stride >= prediction_length` and `rolling_origin`.
- PyTorch early stopping must use inner validation, never the outer ranking fold.
- Every forecasting leaderboard must include a simple baseline before a complex model can claim value.
- Anomaly scores are not operational decisions until a threshold is selected on validation data; test labels must never tune that threshold.
- ESA-ADB reports must distinguish the local uniform-grid approximation from the official event-ID/time-domain evaluator.
- joblib artifacts are trusted-local only and require hashes plus dependency versions.
- New AutoML adapters must register a `CandidateSpec` and provide a backend execution adapter; candidate planning must not grow task-specific model-name branches.
- `load_best()` must verify artifact hashes before deserializing trusted-local model files.
- New executable tasks require a task metric contract, split contract, backend adapter, negative tests, and bilingual documentation.

## Reinforcement Learning Domain / 强化学习域

Current models / 当前模型:

- `DQN`
- `ActorCritic`
- `PPOActorCritic`
- `SACActorCritic`
- `TD3ActorCritic`
- `DecisionTransformer`
- `DreamerV3` (`paper_aligned_portable_subset`; discrete RSSM, symlog world-model losses, and latent imagination)

Protocol / 协议:

- Reinforcement models should inherit `ReinforcementModel`.
- World-model agents should expose observation and imagination transitions separately; environment collection and replay remain external.
- Batches should use `ReinforcementBatch(states, actions=None, rewards=None, next_states=None, dones=None, returns=None, returns_to_go=None, timesteps=None, attention_mask=None, advantages=None, old_log_probs=None)`.
- Value-based models should expose `q_values(...)`, action selection helpers, and TD-style losses.
- Policy-gradient models should expose policy logits, value functions, action evaluation, and scalar `loss_step(batch)` helpers.
- Maximum-entropy continuous-control models should expose Gaussian policy sampling, twin-Q evaluation, actor loss, and critic loss.
- Deterministic continuous-control models should expose policy actions, target-policy smoothing, twin-Q evaluation, actor loss, and critic loss.
- Offline trajectory models should expose causal trajectory encoding, return-conditioned action prediction, and masked sequence losses.
- Environment loops, offline dataset adapters, replay buffers, rollout collectors, target-network updates, advantage estimators, exploration schedules, entropy-temperature tuning, and benchmark evaluation belong outside model classes.

## Recommendation Domain Extension / 推荐系统域扩展

Current additional models / 当前新增模型:

- `LightGCN`
- `SASRec`
- `BERT4Rec`

Protocol / 协议:

- Graph collaborative-filtering models may expose `full_sort_scores(...)` and ranking losses such as `bpr_loss(...)`.
- Sequential recommendation models may use `RecommendationBatch.x_cat` as item-history sequences and expose `next_item_logits(...)`.
- Masked sequential recommendation models may expose `sequence_logits(...)`, `masked_item_logits(...)`, and `score_items(...)`.
- Negative sampling, candidate generation, feature stores, ranking metrics, and online serving belong outside model classes.

## Tabular Domain Extension / 表格学习域扩展

Current additional models / 当前新增模型:

- `TabTransformer`
- `SAINT`
- `NODE`

Protocol / 协议:

- Contextual categorical models may expose `contextual_embeddings(...)`.
- Feature-token models should expose `tokenize(...)` and `forward_features(...)` when these helpers are meaningful.
- Neural tree ensemble models may expose `tree_outputs(...)` and `feature_importance()`.
- Data cleaning, categorical encoding policy, missing-value handling, augmentation, contrastive pre-training loops, calibration, and benchmark protocol belong outside model classes.

## Vision Detection Extension / 视觉检测域扩展

Current additional model / 当前新增模型:

- `YOLO`
- `GroundingDINO`
- `Mask2Former`

Protocol / 协议:

- Single-stage detectors may return `pred_logits`, `pred_boxes`, and `objectness`.
- Detection boxes should be normalized `cx, cy, w, h` before post-processing.
- Language-grounded detectors may consume already-tokenized text ids and return token-alignment logits instead of fixed class logits.
- Query-mask segmentation models may return per-query class logits, per-query mask logits, and composed semantic logits.
- Anchor clustering, assignment, NMS policy, detection losses, COCO evaluation, and training recipes belong outside model classes.

## Multimodal Extension / 多模态域扩展

Current additional model / 当前新增模型:

- `BLIP`
- `BLIP2`

Protocol / 协议:

- Vision-language generation models may expose image/text encoders, cross-modal `fuse(...)`, image-text matching logits, and caption language-model logits.
- Q-Former bridge models may expose `qformer_features(...)`, `language_prefix(...)`, and caption logits without owning the external LLM.
- The framework should consume already-tokenized text ids instead of owning tokenizer logic.
- Tokenizers, large language model weights, synthetic captions, filtering pipelines, and BLIP/BLIP-2 pre-training schedules belong outside model classes.
