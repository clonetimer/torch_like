# torch_like User Guide / 用户指南

中文：本文面向使用者，说明如何导入、构造、调用和选择 `torch_like` 中的模型。

English: This guide is for users. It explains how to import, construct, call, and choose models in `torch_like`.

## What `torch_like` Is / `torch_like` 是什么

中文：`torch_like` 是一个本地 PyTorch-native 模型库，主要沉淀高价值模型的统一调用方式。

English: `torch_like` is a local PyTorch-native model library that provides unified access to high-value models.

中文：它不是完整训练平台，也不是论文复现仓库。训练脚本和 runner 只是辅助验证工具。

English: It is not a full training platform or a paper-reproduction repository. Training scripts and runners are auxiliary validation tools.

## Available Model Families / 可用模型家族

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

### Graph / 通用图学习

- `GCN`
- `EGNN`

`EGNN` uses a separate geometric batch so existing graph models do not need to
accept meaningless coordinate arguments:

```python
import torch
import torch_like as tl

model = tl.EGNN(input_dim=8, output_dim=4, num_nodes=16, coordinate_dim=3)
batch = model.make_geometric_batch(
    x=torch.randn(2, 16, 8),
    coordinates=torch.randn(2, 16, 3),
    adjacency=torch.ones(2, 16, 16),
)
prediction = model.predict(batch)
prediction, updated_coordinates = model.forward_geometric(
    batch.x, batch.coordinates, batch.adjacency
)
```

中文：节点/图级预测对 E(n) 变换保持不变，`updated_coordinates` 对旋转、
平移和反射保持等变。`GeometricGraphBatch` 与普通 `GraphBatch` 分离，避免破坏
GCN、GAT 等非几何图模型的接口。
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

### Sequence / 序列建模

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
- `YOLO`
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
- `BLIP2`
- `LLaVA`

中文：`CLIP` 接收图像张量和已经 tokenized 的 `input_ids`，输出归一化 image/text embeddings 以及 image-text similarity logits。框架不绑定 tokenizer，也不自动下载预训练权重。
English: `CLIP` consumes image tensors and already-tokenized `input_ids`, then returns normalized image/text embeddings and image-text similarity logits. `BLIP2` consumes image tensors and already-tokenized text ids, then returns Q-Former query embeddings, language-prefix features, image-text matching logits, and caption logits. `LLaVA` consumes image tensors plus already-tokenized prompt ids and returns visual-prefix causal LM logits for visual instruction modeling. The framework does not bind a tokenizer, chat template, external LLM checkpoint, or pretrained weight downloader automatically.

中文：`ModernTCN` 提供 patch + 大卷积核的纯卷积预测骨架；`S4` 提供 DPLR 状态空间结构和可移植 recurrent scan；`TransformerXL` 提供 segment memory 长上下文自回归；`Performer` 提供随机特征线性注意力；`Reformer` 提供 bucketed LSH 局部注意力；`Graphormer` 提供中心性与最短路空间编码；`GATv2` 提供动态 pairwise 图注意力；`GPS` 组合局部图传播与全局注意力；`PNA` 提供多聚合器与度缩放的消息传递；`GraphCL` 提供两路图增强视图的对比表征学习接口；`GraphMAE` 提供 masked node-feature 图自编码重建接口；`Hyena` 提供隐式 FFT 长卷积；`InceptionTime` 提供多尺度故障诊断分类；`ROCKET` 提供固定随机卷积特征和线性分类头；`TranAD` 提供两阶段自条件异常重建；`USAD` 提供双自编码器无监督异常检测；`MAE` 通过 `fit_step(batch)` 计算 masked-patch 重建损失；`SimCLR`、`BYOL` 和 `DINO` 提供视觉自监督表征学习接口；`SigLIP` 使用独立图文对的 sigmoid loss。所有实现默认随机初始化，不自动下载论文预训练权重。

English: `ModernTCN` provides a patch-based large-kernel convolutional forecaster; `S4` uses a DPLR state-space structure with a portable recurrent scan; `TransformerXL` adds segment-memory long-context autoregression; `Performer` adds random-feature linear attention; `Reformer` adds bucketed LSH local attention; `RetNet` provides equivalent parallel-training and recurrent-state inference paths; `LLaMA` adds RoPE, RMSNorm, SwiGLU, and grouped-query attention for modern decoder-only language modeling; `Mixtral` adds sparse MoE top-k expert routing on top of LLaMA-style decoder blocks; `Graphormer` adds centrality and shortest-path encodings; `GATv2` adds dynamic pairwise graph attention; `GPS` combines local propagation and global attention; `PNA` adds multi-aggregator degree-scaled message passing; `GCNII` adds initial residual and identity mapping for deep GCNs; `DGI` adds self-supervised graph representation learning through corruption and global summary discrimination; `GraphCL` adds graph-level contrastive learning over two augmented graph views; `GraphMAE` adds masked node-feature graph autoencoding; `Hyena` provides implicit FFT long convolution; `InceptionTime` provides multiscale diagnosis classification; `ROCKET` provides fixed random convolutional features and a linear classifier; `TranAD` provides two-phase self-conditioned anomaly reconstruction; `USAD` provides dual-autoencoder unsupervised anomaly detection; `OmniAnomaly` provides variational recurrent anomaly reconstruction; `MAE.fit_step(batch)` computes masked-patch reconstruction loss; `SimCLR`, `BYOL`, `DINO`, and `MoCo` provide visual self-supervised representation interfaces; and `SigLIP` uses an independent pairwise sigmoid objective. All implementations are randomly initialized and do not download paper checkpoints automatically.

### Kimi Linear reference API / Kimi Linear 参考接口

```python
import torch
import torch_like as tl
from torch_like import nn

kda = nn.KimiDeltaAttention(embed_dim=32, num_heads=4, chunk_size=16)
x = torch.randn(2, 64, 32)
y, state = kda(x, mode="chunkwise", return_state=True)
step_y, state = kda.forward_step(torch.randn(2, 32), state)

model = tl.KimiLinear(vocab_size=128, max_seq_len=256, d_model=32, n_heads=4, n_layers=2)
logits = model(torch.randint(0, 128, (2, 32)))

# Insert a causal full-attention layer after every three KDA layers.
hybrid = tl.KimiLinear(
    vocab_size=128, max_seq_len=256, d_model=32, n_heads=4,
    n_layers=4, full_attention_interval=4,
)
```

中文：该实现提供可移植的细粒度门控 delta-rule scan、分块调用和显式递归状态。它不是官方 48B 模型，且不包含 FlashKDA、FLA 或 vLLM 内核，因此不能直接套用官方吞吐结论。

English: This implementation provides a portable fine-grained gated delta-rule scan, chunked calls, and explicit recurrent state. Setting `full_attention_interval > 0` enables a hybrid causal-attention model; fixed-size recurrent state is then unavailable because full attention requires a KV history. It is not the official 48B model and does not bundle FlashKDA, FLA, or vLLM kernels, so official throughput claims do not apply directly.

### xLSTM mLSTM API / xLSTM mLSTM 接口

```python
import torch
import torch_like as tl
from torch_like import nn

cell = nn.mLSTMCell(embed_dim=32, num_heads=4)
y, state = cell(torch.randn(2, 16, 32), return_state=True)
step_y, state = cell.forward_step(torch.randn(2, 32), state)

model = tl.xLSTM(vocab_size=128, max_seq_len=256, d_model=32, n_heads=4, n_layers=2)
tokens = torch.randint(0, 128, (2, 32))
logits, states = model.forward_recurrent(tokens)
```

中文：当前 `xLSTM` 实现聚焦论文中的 mLSTM 矩阵记忆、指数门控和数值稳定化递推，不包含 sLSTM 路径、官方 7B 权重或定制 CUDA kernel。因此 catalog 使用 `paper_aligned_mlstm_reference`，而不是完整官方复刻。

English: The current `xLSTM` implementation focuses on the paper's mLSTM matrix memory, exponential gates, and numerically stabilized recurrence. It excludes the sLSTM path, official 7B weights, and custom CUDA kernels, so the catalog labels it `paper_aligned_mlstm_reference` rather than a complete official reproduction.

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
- `TabTransformer`
- `SAINT`

English: `TabNet` exposes attentive feature masks through `explain(batch)`; `NODE` exposes differentiable oblivious tree ensembles through `tree_outputs(...)` and `feature_importance()`; `FTTransformer` exposes feature tokenization through `tokenize(...)` and `forward_features(...)`; `TabTransformer` exposes categorical contextual embeddings through `contextual_embeddings(...)`; and `SAINT` exposes column/row attention tokenization through `tokenize(...)` and `forward_features(...)`. Data cleaning, categorical encoding policy, missing-value policy, augmentation, contrastive pre-training loops, calibration, and tabular benchmark protocol remain outside model classes.

### Recommendation / 推荐系统

- `NCF`
- `DeepFM`
- `DCN`
- `LightGCN`
- `SASRec`
- `BERT4Rec`

English: recommendation models use `RecommendationBatch` and expose `predict(batch)` plus `loss_step(batch)` where the task allows it. `NCF` handles user-item ids; `DeepFM` and `DCN` handle CTR-style categorical and continuous fields; `LightGCN` handles graph collaborative filtering with `full_sort_scores(...)` and `bpr_loss(...)`; `SASRec` handles item-sequence recommendation with `next_item_logits(...)`; `BERT4Rec` handles bidirectional masked-item prediction with `sequence_logits(...)`, `masked_item_logits(...)`, and `score_items(...)`. Negative sampling, candidate generation, feature stores, online serving, and ranking metrics belong outside model classes.

### Audio / 音频与语音

- `Wav2Vec2`
- `EnCodec`
- `Conformer`
- `HuBERT`
- `Whisper`
- `AudioSpectrogramTransformer`

English: audio models use `AudioBatch` and expose `predict(batch)` plus `loss_step(batch)`. `Wav2Vec2` consumes raw waveforms, `HuBERT` consumes raw waveforms and predicts masked hidden-unit labels, `Conformer` consumes frame-level acoustic features, `Whisper` consumes log-mel features plus token ids for decoder training/generation, and `AudioSpectrogramTransformer` consumes spectrogram tensors. Tokenization, hidden-unit clustering, CTC/beam decoding, mel extraction, augmentation, and ASR evaluation remain outside model classes.

EnCodec / 神经音频编解码:

```python
import torch
import torch_like as tl

codec = tl.EnCodec(d_model=64, target_bandwidth=6.0)
waveform = torch.randn(2, 24000)
encoded = codec.encode(waveform, bandwidth=3.0)
reconstructed = codec.decode(encoded["codes"], length=waveform.size(-1))
codec.set_target_bandwidth(6.0)
```

中文：`EnCodec` 提供随机初始化的因果卷积编码器、带宽可调 RVQ 和波形解码器。它不自动下载官方权重，也不包含熵模型或多尺度判别器训练器。

English: `EnCodec` provides a randomly initialized causal convolutional encoder, bandwidth-scalable RVQ, and waveform decoder. It does not download official weights or bundle the entropy model or multi-scale discriminator trainer.

### Reinforcement Learning / 强化学习

- `DQN`
- `ActorCritic`
- `PPOActorCritic`
- `SACActorCritic`
- `TD3ActorCritic`
- `DecisionTransformer`
- `DreamerV3`

English: reinforcement models use `ReinforcementBatch` and expose reusable model-side functions such as `q_values(...)`, `greedy_action(...)`, `policy_logits(...)`, `value(...)`, `evaluate_actions(...)`, `td_loss(...)`, `ppo_loss(...)`, SAC-style `actor_loss(...)` / `critic_loss(...)`, TD3-style deterministic `policy(...)` / `target_actions(...)`, and Decision Transformer `encode_trajectory(...)` / `predict_action(...)` / `action_loss(...)`. Environment stepping, offline dataset loading, replay buffers, rollout collection, target-network updates, exploration schedules, advantage estimation, entropy-temperature tuning, and benchmark protocols remain outside model classes.

DreamerV3 / 世界模型强化学习:

```python
import torch
import torch_like as tl

model = tl.DreamerV3(state_dim=8, action_dim=3, d_model=64)
states = torch.randn(4, 16, 8)
actions = torch.randn(4, 16, 3).tanh()
outputs = model(states, actions)
imagined = model.imagine(outputs["posterior"], horizon=8)
losses = model.actor_critic_loss(outputs["posterior"], horizon=8)
```

中文：`DreamerV3` 提供 categorical RSSM、observation/reward/continuation heads、latent imagination 和 actor/critic loss。环境交互、replay buffer、优化器更新顺序和 benchmark runner 不属于模型类。

English: `DreamerV3` provides a categorical RSSM, observation/reward/continuation heads, latent imagination, and actor/critic losses. Environment interaction, replay buffers, optimizer update ordering, and benchmark runners remain outside the model class.

```python
import torch
import torch_like as tl

model = tl.PPOActorCritic(state_dim=8, action_dim=4, d_model=32)
states = torch.randn(16, 8)
actions, old_log_probs, _ = model.act(states)
batch = model.make_reinforcement_batch(
    states=states,
    actions=actions,
    old_log_probs=old_log_probs.detach(),
    returns=torch.randn(16),
    advantages=torch.randn(16),
)
loss = model.loss_step(batch)
```

中文：`DDPM` 提供 `q_sample(...)`、`predict_noise(...)`、`loss_step(batch)` 和 `sample(...)`。数据集、FID 评估和长时间训练 runner 不属于模型类职责。
English: `DDPM` exposes `q_sample(...)`, `predict_noise(...)`, `loss_step(batch)`, and `sample(...)`. Datasets, FID evaluation, and long-running training runners are outside the model class boundary.

中文：`DDPM.sample(...)` 在完整步数下使用随机 DDPM 采样；减少 `num_steps` 时使用确定性 DDIM 更新，避免错误地跨时间步套用相邻后验公式。
English: `DDPM.sample(...)` uses stochastic DDPM sampling for the full schedule and deterministic DDIM updates when `num_steps` is reduced.

中文：`VAE` 提供 `encode(...)`、`decode(...)`、`reparameterize(...)`、`loss_step(batch)` 和 `sample(...)`。
English: `VAE` exposes `encode(...)`, `decode(...)`, `reparameterize(...)`, `loss_step(batch)`, and `sample(...)`.

中文：`DCGAN` 提供 `sample(...)`、`discriminate(...)`、`generator_loss(...)` 和 `discriminator_loss(...)`；两个 optimizer 的训练顺序属于 runner 或用户代码。
English: `DCGAN` exposes `sample(...)`, `discriminate(...)`, `generator_loss(...)`, and `discriminator_loss(...)`; the two-optimizer update schedule belongs in runners or user code.

中文：`DiT` 使用 patch token 和 timestep-conditioned Transformer blocks 做扩散噪声预测；`RealNVP` 提供可逆 affine coupling flow、精确 `log_prob(...)` 和 `sample(...)`。
English: `DiT` uses patch tokens and timestep-conditioned Transformer blocks for diffusion noise prediction; `LatentDiffusion` exposes `encode_to_latent(...)`, `decode_latent(...)`, latent-space `q_sample(...)`, `predict_noise(...)`, `loss_step(batch)`, `sample_latents(...)`, and image `sample(...)`; `ControlNet` adds `encode_control(...)`, `control_residuals(...)`, control-conditioned `predict_noise(...)`, `loss_step(batch, control=...)`, and control-conditioned `sample(...)`; `RectifiedFlow` exposes straight-line `interpolate(...)`, `target_velocity(...)`, `predict_velocity(...)`, flow-matching `loss_step(batch)`, and Euler `sample(...)`; `PixelCNN` provides masked-convolution autoregressive `log_prob(...)`, `loss_step(batch)`, and ancestral `sample(...)`; `RealNVP` provides invertible affine coupling flows, exact `log_prob(...)`, and `sample(...)`; `VQVAE` provides a discrete codebook, `quantize(...)`, reconstruction loss, and sampling.

中文：`LatentDiffusion` 只实现潜空间扩散的模型核心；文本编码器、tokenizer、classifier-free guidance、外部 Stable Diffusion 权重和采样 scheduler 库不属于模型类职责。
English: `LatentDiffusion` implements only the model-side latent diffusion core. Text encoders, tokenizers, classifier-free guidance, external Stable Diffusion checkpoints, and sampler scheduler libraries remain outside the model class.

中文：`ControlNet` 只实现控制分支和零初始化残差注入；边缘图、深度图、姿态图等控制信号的外部预处理器，以及 ControlNet/Stable Diffusion 官方权重转换不属于模型类职责。
English: `ControlNet` implements only the control branch and zero-initialized residual injection. External annotators for edges, depth, pose, and other control maps, plus official ControlNet/Stable Diffusion checkpoint conversion, stay outside the model class.

中文：`RectifiedFlow` 实现 Flow Matching / Rectified Flow 的模型侧向量场；高阶 ODE 求解器、guidance 策略、大规模预训练权重和数据分布工程不属于模型类职责。
English: `RectifiedFlow` implements the model-side velocity field for Flow Matching / Rectified Flow. Higher-order ODE solvers, guidance strategies, large pretrained checkpoints, and data-distribution engineering remain outside the model class.

### Vision Output Protocol / 视觉输出协议

中文：vision 域共享 `make_vision_batch(x, y=None)` 和 `predict(batch)`，但不同任务的输出不同。
English: The vision domain shares `make_vision_batch(x, y=None)` and `predict(batch)`, while output shapes depend on the task.

- Classification / 分类: returns logits with shape `(batch, num_classes)`.
- Segmentation / 分割: returns dense logits with shape `(batch, num_classes, height, width)`.
- Universal segmentation / 通用分割: `Mask2Former` returns query `pred_logits`, query `pred_masks`, and composed `semantic_logits`.
- Object detection / 目标检测: `DETR` returns `{"pred_logits": (batch, queries, classes + 1), "pred_boxes": (batch, queries, 4)}`. `pred_boxes` uses normalized `cx, cy, w, h`; call `model.post_process(...)` for normalized `xyxy` detections.
- Open-vocabulary detection / 开放词表检测: `GroundingDINO` returns `{"token_logits": (batch, queries, text_len), "objectness": (batch, queries), "pred_boxes": (batch, queries, 4)}` and consumes already-tokenized text ids.
- Promptable segmentation / 提示分割: `SAM` returns `{"masks": (batch, masks, height, width), "low_res_masks": ..., "iou_predictions": ...}` and accepts point, box, and mask prompts through `forward(...)`.

## Import Patterns / 导入方式

### Top-level import / 顶层导入

中文：适合交互式使用和快速实验。

English: Best for interactive usage and quick experiments.

```python
import torch_like as tl

model = tl.PatchTST(
    enc_in=3,
    dec_in=3,
    c_out=3,
    seq_len=16,
    label_len=8,
    pred_len=4,
)
```

### Domain import / 按域导入

中文：适合明确知道模型所属任务域时使用。

English: Useful when the task domain is already known.

```python
from torch_like.models.forecasting import AGCRN
from torch_like.models.detection import GDN
```

### Registry construction / Registry 构造

中文：适合配置驱动、CLI 或批量实验。

English: Useful for config-driven code, CLI tools, or batch experiments.

```python
from torch_like.models import (
    create_model,
    get_model_info,
    list_domains,
    list_models,
    list_models_by_fidelity,
    list_models_by_maturity,
)

print(list_domains())
print(list_models("forecasting").keys())
print(get_model_info("Informer", domain="forecasting"))
print(list_models_by_fidelity("paper_aligned", domain="forecasting").keys())
print(list_models_by_maturity("unassessed", domain="forecasting").keys())

model = create_model(
    "Informer",
    domain="forecasting",
    enc_in=4,
    dec_in=4,
    c_out=4,
    seq_len=16,
    label_len=8,
    pred_len=4,
)
```

中文：`integration_level` 在兼容期继续保留。新增的 `implementation_fidelity`、`implementation_scope`、`execution_profile`、`maturity`、`benchmark_evidence` 和 `evidence_record` 是可独立查询的 schema v2 字段。`unassessed` 表示尚无足够登记证据，不表示模型无价值或不可运行。

English: `integration_level` remains available during the compatibility period. The additive `implementation_fidelity`, `implementation_scope`, `execution_profile`, `maturity`, `benchmark_evidence`, and `evidence_record` fields form schema v2. `unassessed` means that sufficient registered evidence is not yet available; it does not mean the model is unusable or valueless.

## Forecasting Usage / 预测模型用法

### Standard forecasting / 标准预测

```python
import torch
import torch_like as tl

model = tl.TSMixer(
    enc_in=3,
    dec_in=3,
    c_out=3,
    seq_len=16,
    label_len=8,
    pred_len=4,
    d_model=16,
    e_layers=2,
    d_ff=32,
)

x_enc = torch.randn(2, 16, 3)
batch = model.make_forecast_batch(x_enc)
out = model.predict(batch)
print(out.shape)  # (2, 4, 3)
```

中文：`make_forecast_batch` 会在缺省时自动构造 decoder 输入和时间特征占位。

English: `make_forecast_batch` automatically builds decoder inputs and placeholder time features when omitted.

### Exogenous forecasting / 外生变量预测

中文：`TimeXer`、`DAG` 和 `TFT` 支持目标变量与外生变量的双输入协议。

English: `TimeXer`, `DAG`, and `TFT` support the dual-input protocol for target variables and exogenous variables.

```python
import torch
import torch_like as tl

model = tl.TimeXer(
    enc_in=4,
    dec_in=2,
    c_out=2,
    seq_len=16,
    label_len=8,
    pred_len=4,
    d_model=16,
    n_heads=4,
    e_layers=2,
    d_ff=32,
    patch_len=4,
)

x_enc = torch.randn(2, 16, 4)
batch = model.make_forecast_batch(
    x_enc=x_enc,
    x_target_enc=x_enc[..., :2],
    x_exog_enc=x_enc[..., 2:],
    x_target_dec=torch.randn(2, 12, 2),
    x_exog_dec=torch.randn(2, 12, 2),
)

out = model.predict(batch)
print(out.shape)  # (2, 4, 2)
```

### Probabilistic forecasting / 概率预测

中文：概率预测模型提供 `predict_distribution`、`sample` 和 `fit_step`。

English: Probabilistic forecasting models expose `predict_distribution`, `sample`, and `fit_step`.

```python
import torch
import torch_like as tl

model = tl.DeepAR(
    enc_in=3,
    dec_in=3,
    c_out=3,
    seq_len=12,
    label_len=6,
    pred_len=4,
    d_model=16,
    e_layers=2,
    num_samples=20,
)

x_enc = torch.randn(2, 12, 3)
y_true = torch.randn(2, 4, 3)
batch = model.make_probabilistic_forecast_batch(x_enc=x_enc, y_true=y_true)

mean = model.predict(batch)
dist = model.predict_distribution(batch)
samples = model.sample(batch)
loss = model.fit_step(batch)
```

## Graph-aware Forecasting / 图感知预测

中文：图时序模型统一支持 `graph_mode` 协议。

English: Graph spatiotemporal models share a unified `graph_mode` protocol.

Supported models / 支持模型:

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

Graph modes / 图模式:

- `learned`: 中文：只使用模型学习出的图；English: use only the learned graph.
- `prior`: 中文：只使用外部邻接矩阵；English: use only the external adjacency.
- `hybrid`: 中文：学习图与外部图先验加权融合；English: weighted fusion of learned graph and external prior.

```python
import numpy as np
import torch
import torch_like as tl

adj = np.eye(5, dtype=np.float32)

model = tl.D2STGNN(
    enc_in=5,
    dec_in=5,
    c_out=5,
    seq_len=24,
    label_len=12,
    pred_len=6,
    graph_mode="hybrid",
    adjacency=adj,
    prior_weight=0.7,
)

x = torch.randn(2, 24, 5)
out = model.predict(model.make_forecast_batch(x))
```

## Detection Usage / 异常检测用法

中文：检测模型统一使用 `DetectionBatch`，并约定 `fit_step` 返回训练损失，`score_batch` 返回异常分数。

English: Detection models use `DetectionBatch`; `fit_step` returns a training loss and `score_batch` returns anomaly scores.

### RTDetector

```python
import torch
import torch_like as tl

model = tl.RTDetector(input_dim=8, seq_len=64)
x = torch.randn(4, 64, 8)
batch = model.make_detection_batch(x)
scores = model.score_batch(batch)
```

### MEMTO

```python
import torch
import torch_like as tl

model = tl.MEMTO(input_dim=4, seq_len=10, d_model=16, n_heads=4, e_layers=1, d_ff=32, mem_size=8)
x = torch.rand(3, 10, 4)
batch = model.make_detection_batch(x)
out = model.predict(batch)
score = model.score_batch(batch, reduction="mean")
```

### GDN

中文：`GDN` 是图异常检测代表模型，适合多变量时间序列传感器依赖建模。

English: `GDN` is the graph-anomaly representative for modeling sensor dependencies in multivariate time series.

```python
import numpy as np
import torch
import torch_like as tl

adj = np.eye(8, dtype=np.float32)

model = tl.GDN(
    input_dim=8,
    seq_len=64,
    d_model=32,
    graph_mode="hybrid",
    adjacency=adj,
    prior_weight=0.6,
)

x = torch.randn(4, 64, 8)
batch = model.make_detection_batch(x)
loss = model.fit_step(batch)
scores = model.score_batch(batch, reduction="mean")
```

## RUL Usage / RUL 用法

### Transformer-adapter RUL / Transformer 适配器 RUL

```python
import torch
import torch_like as tl

model = tl.InformerRUL(enc_in=8, seq_len=64, d_model=32)
x = torch.randn(4, 64, 8)
out = model.predict(model.make_rul_batch(x))
print(out.shape)  # (4,)
```

Shared training contract / 统一训练合同:

```python
import torch
import torch_like as tl

model = tl.PatchTSTRUL(enc_in=2, seq_len=512, d_model=32, n_heads=4, e_layers=1)
x = torch.randn(8, 512, 2)
targets = torch.rand(8)
batch = model.make_rul_batch(x, targets=targets)
loss = model.loss_step(batch, loss="huber")
loss.backward()
predictions = model.predict(batch)
```

中文：三个新 RUL 类是统一任务适配器。它们适合在相同数据、预处理和划分协议下比较，但不应被描述为
论文原样复现。XJTU-SY 文件夹排行榜配置见 `examples/xjtu_sy_rul_leaderboard.yaml`；该配置使用
`group_cv` 在 development bearings 上轮换验证，并只对最终选中的管线解封一次测试 bearing。

English: The three new RUL classes are task adapters under one contract. They support controlled
comparison under the same data, preprocessing, and split protocol, but are not paper-exact reproductions.
The recommended XJTU-SY configuration uses `group_cv` and a one-shot sealed test.

AutoML forecasting uses a separate rolling-origin contract. `sequence_length` defines historical context,
`prediction_length` defines the future horizon, and the latest windows remain sealed until model selection finishes.
See `examples/forecasting_rolling_leaderboard.yaml`.

AutoML forecasting 使用独立的 rolling-origin 协议：`sequence_length` 定义历史上下文，
`prediction_length` 定义未来步长，最新时间窗口在模型选择完成前保持封存。配置见
`examples/forecasting_rolling_leaderboard.yaml`。

AutoML anomaly detection uses a predefined chronological protocol. Native `format: esa_adb` ingestion reads selected channels from the raw mission ZIP, applies zero-order-hold resampling, selects the binary threshold on validation data, and evaluates the sealed test only after model selection. See `examples/esa_mission1_anomaly_leaderboard.yaml`.

AutoML 异常检测使用预定义的时间协议。`format: esa_adb` 可直接从原始任务 ZIP 读取指定通道，执行 zero-order hold 重采样，只在验证集选择二值阈值，并在模型选择完成后才评估封存测试集。

All leaderboard split protocols keep test data outside candidate processes. After validation ranking, the selected pipeline is retrained once and exported as a trusted-local inference bundle:

所有排行榜划分协议都不会把测试数据传入候选进程。完成验证排名后，入选管线只重训一次，并导出为可信本地推理包：

```python
from torch_like.automl import load_best, load_config, prepare_dataset

bundle = load_best("runs/my-task", device="cpu")
config = load_config(bundle.run_dir / "config.json")
prepared = prepare_dataset(config.dataset)
predictions = bundle.predict(prepared.x)
```

中文：`bundle.predict()` 接收与训练一致的已样本化 NumPy 输入；异常检测还可用 `bundle.score()` 获取连续异常分数。不要加载来源不可信的 joblib 或 PyTorch checkpoint。

English: `bundle.predict()` consumes the same prepared NumPy input contract as training. Anomaly bundles also expose continuous scores through `bundle.score()`. Never load joblib or PyTorch checkpoints from untrusted sources.

Interrupted tasks can reuse terminal trial artifacts without reopening a completed sealed test:

中断任务可以复用已结束的 trial，并且不会再次打开已经完成的封存测试：

```powershell
torch-like run --config task.yaml --resume
```

```python
result = LeaderboardTask(load_config("task.yaml")).run(resume=True)
```

中文：续跑要求配置、数据指纹、划分和候选计划完全一致；已记录为 OOM、NaN、超时或失败的 trial 保持终态，不会自动重试。

English: Resume requires an exact configuration, data fingerprint, split, and candidate-plan match. Trials already recorded as OOM, NaN, timeout, or failed remain terminal and are not retried automatically.

Use multiple independent seeds when ranking stability matters:

需要评估排名稳定性时，可以配置多个独立随机种子：

```yaml
search:
  max_trials: 6
  evaluation_seeds: [11, 29, 47]
```

中文：每个“管线 + 种子”组合都是独立 trial。只有全部种子完成的管线才会参与排名；`leaderboard.csv` 会给出指标均值、种子标准差和跨种子名次稳定性。封存测试仍只执行一次。注意实际 trial 数约等于 `max_trials × evaluation_seeds 数量`。

English: Every pipeline/seed pair is an independent trial. Only pipelines completing all seeds are ranked. `leaderboard.csv` reports metric means, seed standard deviations, and cross-seed rank stability, while the sealed test remains one-shot. Planned trial count is approximately `max_trials × number of evaluation_seeds`.

Validated leaderboard trust / 验证级排行榜可信性:

```yaml
trust:
  level: validated
search:
  evaluation_seeds: [11, 29, 47]
```

中文：默认 `trust.level: development` 适合快速验证模型是否能接入。需要把结果作为正式对比时，使用 `validated`；框架会要求至少 3 个评估种子，并写出 `evaluation_trust_audit.json`，证明每个候选管线获得同样的种子覆盖和训练预算。

English: The default `trust.level: development` is suitable for quick model-integration checks. For formal comparisons, use `validated`; the framework requires at least three evaluation seeds and writes `evaluation_trust_audit.json` to prove equal seed coverage and training-budget treatment for every candidate pipeline.

Benchmark trust tiers and promotion rules are maintained in [`AUTOML_BENCHMARK_TRUST.md`](AUTOML_BENCHMARK_TRUST.md). For a ready validated forecasting template, start from [`examples/forecasting_rolling_validated.yaml`](../examples/forecasting_rolling_validated.yaml).

排行榜可信分层和晋升规则见 [`AUTOML_BENCHMARK_TRUST.md`](AUTOML_BENCHMARK_TRUST.md)。验证级预测模板见 [`examples/forecasting_rolling_validated.yaml`](../examples/forecasting_rolling_validated.yaml)。

Choose the candidate pool according to the purpose of the task:

根据排行榜任务的目的选择候选策略：

```yaml
search:
  candidate_policy: production  # production | torch_like_only | baseline_only | prefer_torch_like
```

中文：`production` 用于寻找实际最优结果；`torch_like_only` 用于单独验证框架模型；`baseline_only` 用于快速建立传统基线；`prefer_torch_like` 在预算截断时优先分配给框架模型，同时尽可能保留基线锚点。运行后分别查看 `leaderboard.csv`、`leaderboard_torch_like.csv` 和 `leaderboard_baselines.csv`。只有总榜冠军会执行最终密封测试。

English: Use `production` for the strongest practical result, `torch_like_only` for framework validation, `baseline_only` for a fast traditional reference, and `prefer_torch_like` to prioritize framework candidates under a constrained budget while retaining a baseline anchor when possible. Runs export overall, torch-like, and baseline leaderboards. Only the overall winner receives final sealed-test evaluation.

### THGNN

```python
import torch
import torch_like as tl

model = tl.THGNN(enc_in=8, seq_len=64, d_model=32)
x = torch.randn(4, 64, 8)
out = model.predict(model.make_rul_batch(x))
```

## Sequence Usage / 序列域用法

中文：sequence 域用于 decoder-only、SSM、长上下文和通用序列建模，不强行塞进 forecasting。

`Mamba2` additionally exposes equivalent full-sequence, chunkwise, and
fixed-state recurrent execution. The portable SSD path prioritizes semantic
correctness and CPU/CUDA portability over the optional fused kernels:

```python
import torch
import torch_like as tl

model = tl.Mamba2(
    vocab_size=32000,
    max_seq_len=2048,
    d_model=256,
    n_layers=4,
    d_ff=1024,
    d_state=64,
    head_dim=64,
    chunk_size=128,
)
tokens = torch.randint(0, 32000, (2, 256))
logits = model(tokens)
chunk_logits = model.forward_chunkwise(tokens)
step_logits, states = model.forward_recurrent(tokens[:, :1])
```

中文：`forward_chunkwise` 用于保持状态的分块扫描，`forward_recurrent`
返回每层固定大小的 `SSDState`，适合流式推理。当前实现不宣称达到官方融合 CUDA
内核速度；三种路径的语义等价由测试保证。

English: The sequence domain is for decoder-only, SSM, long-context, and general sequence modeling, instead of forcing these models into forecasting.

```python
import torch
import torch_like as tl

model = tl.LRU(vocab_size=128, max_seq_len=32, d_model=32, n_layers=2)
tokens = torch.randint(0, 128, (2, 16))
batch = model.make_sequence_batch(tokens)
logits = model.next_token_logits(batch)
```

Seq2Seq / 编码器-解码器:

```python
model = tl.T5(vocab_size=128, max_seq_len=32, d_model=32, n_heads=4, n_layers=2)
src = torch.randint(1, 128, (2, 12))
tgt = torch.randint(1, 128, (2, 6))
batch = model.make_seq2seq_batch(src, tgt)
logits = model.predict(batch)
```

## Runner Overview / Runner 概览

中文：runner 是辅助实验工具，用于验证模型能否跑通，不是核心框架边界。

English: Runners are auxiliary experiment tools for validating that models can run; they are not the core framework boundary.

Common runners / 常用 runner:

- `runner/forecast_runner.py`
- `runner/forecast_comparison_runner.py`
- `runner/detection_runner.py`
- `runner/rul_runner.py`
- `runner/benchmark_compile.py`

Example / 示例:

```powershell
& 'D:\Project\base\base\Scripts\python.exe' runner/forecast_runner.py --help
```

## 性能配置 / Performance Configuration

中文：模型层无需额外开关即可使用原生 SDPA。`FullAttention(output_attention=False)`、`GPT`、`LLaMA` 和 `Mixtral` 会将兼容的注意力计算交给 PyTorch；实际选择 Flash、内存高效或数学后端由 PyTorch 根据设备、dtype、形状和掩码决定。

English: Native SDPA requires no model-level switch. `FullAttention(output_attention=False)`, `GPT`, `LLaMA`, and `Mixtral` delegate compatible attention operations to PyTorch; PyTorch selects Flash, memory-efficient, or math backends according to device, dtype, shapes, and masks.

Python runner configuration / Python runner 配置:

```python
from runner.compile_utils import CompileConfig
from runner.performance import ExecutionConfig
from runner.forecast_runner import ForecastRunConfig

config = ForecastRunConfig(
    data=data_config,
    execution=ExecutionConfig(
        mixed_precision=True,
        amp_dtype="bfloat16",
        non_blocking=True,
        zero_grad_set_to_none=True,
    ),
    compile=CompileConfig(enabled=True, mode="reduce-overhead"),
)
```

CLI flags / CLI 参数:

- `--amp`: enable automatic mixed precision / 启用自动混合精度
- `--amp-dtype {float16,bfloat16}`: select AMP dtype / 选择 AMP 数据类型
- `--blocking-transfers`: disable non-blocking CUDA copies / 禁用异步 CUDA 搬运
- `--zero-grad-values`: use zero-filled gradients instead of `None` / 使用数值清零而不是将梯度设为 `None`
- `--compile`: enable `Module.compile()` / 启用 `Module.compile()`

中文：CUDA FP16 训练会使用 GradScaler；BF16 通常不需要缩放。`non_blocking=True` 只有在 CUDA 搬运与 pinned-memory DataLoader 配合时才可能带来收益。编译存在首次调用开销，应使用 `runner/benchmark_compile.py` 或任务级基准测量稳态收益。

English: CUDA FP16 training uses GradScaler; BF16 normally does not require scaling. `non_blocking=True` can help only when CUDA transfers are paired with pinned-memory loaders. Compilation has first-call overhead, so measure steady-state gains with `runner/benchmark_compile.py` or a task-level benchmark.

中文：当前生成接口尚未实现 KV cache，因此 GPT/LLaMA 的逐 token 生成仍重新计算完整前缀。

English: Generation APIs do not yet implement a KV cache, so token-by-token GPT/LLaMA generation still recomputes the full prefix.

## Forecast Data Protocol / 预测数据协议

中文：`ForecastingBatch` 支持传统单输入和目标变量 + 外生变量双输入。

English: `ForecastingBatch` supports both legacy single-input forecasting and target-plus-exogenous dual-input forecasting.

Key fields / 关键字段:

- `x_enc`: encoder input / 编码器输入
- `x_dec`: decoder input / 解码器输入
- `x_target_enc`: historical target variables / 历史目标变量
- `x_exog_enc`: historical exogenous variables / 历史外生变量
- `x_target_dec`: decoder target-side context / 解码器目标侧上下文
- `x_exog_dec`: decoder exogenous-side context / 解码器外生侧上下文

中文：数据适配器会校验拆分比例、最小长度、有限数值、目标/外生索引和图邻接形状。非法配置会在创建 loader 前抛出明确的 `ValueError`。

English: Data adapters validate split ratios, minimum lengths, finite values, target/exogenous indices, and graph adjacency shapes. Invalid configurations raise a clear `ValueError` before loaders are created.

中文：三维图数据使用 `(time, nodes, features)`，其中 `target_indices` 和 `exogenous_indices` 先选择最后一个 feature 轴。loader 随后把 `(nodes, features)` 展平为统一的 forecasting 通道，并把节点邻接矩阵扩展为通道级邻接矩阵；原始节点邻接矩阵保存在 bundle 的 `node_adjacency` 元数据中。

English: Three-dimensional graph data uses `(time, nodes, features)`. Target and exogenous indices first select the final feature axis. The loader then flattens `(nodes, features)` into standard forecasting channels and expands node adjacency to channel-level adjacency; the original node adjacency remains available as `node_adjacency` in the bundle metadata.

中文：辅助 runner 默认写入自身的 `results/`，也可以通过配置中的 `results_dir` 或 CLI 的 `--results-dir` 重定向输出。

English: Auxiliary runners write to their local `results/` by default; use config `results_dir` or CLI `--results-dir` to redirect output.

## Development References / 开发参考

- [Packaging Guide / 打包安装指南](PACKAGING.md)
- [Development Guide / 开发指南](DEVELOPMENT.md)
- [Project Structure / 项目结构](PROJECT_STRUCTURE.md)
- [Sequence Domain Draft / 序列域草案](SEQUENCE_DOMAIN_DRAFT.md)

## Vision Extension Notes / 视觉域补充

YOLO / 单阶段检测:

- `YOLO` returns `{"pred_logits": (batch, anchors, classes), "pred_boxes": (batch, anchors, 4), "objectness": (batch, anchors)}`.
- Boxes use normalized `cx, cy, w, h`.
- Anchor clustering, NMS policy, assignment, loss matching, and COCO evaluation remain outside model classes.

GroundingDINO / 开放词表检测:

- `GroundingDINO` consumes image tensors plus already-tokenized `input_ids`.
- It exposes `encode_image_tokens(...)`, `encode_text_tokens(...)`, `forward_features(...)`, token alignment logits, grounded `objectness`, normalized boxes, and `post_process(...)`.
- Tokenization, phrase span extraction, NMS policy, Grounded-SAM orchestration, official checkpoint conversion, and dataset evaluation remain outside model classes.

SAM / 提示分割:

- `SAM` exposes `encode_image(...)`, `encode_prompts(...)`, `predict_masks(...)`, and `forward(..., points=..., boxes=..., masks=...)`.
- Points and boxes can use normalized coordinates or pixel coordinates through `prompt_coordinate_format`.
- Official checkpoints, interactive annotation UI, SAM2 video memory, and training recipes remain outside model classes.

Mask2Former / 通用分割:

- `Mask2Former` exposes `encode_pixels(...)`, `forward_features(...)`, `predict_masks(...)`, `semantic_logits(...)`, and `post_process(...)`.
- It returns class logits for mask queries, per-query mask logits, and composed dense semantic logits.
- Hungarian matching, panoptic fusion, dataset-specific target formatting, and COCO/ADE20K evaluation remain outside model classes.

## Multimodal Extension Notes / 多模态域补充

BLIP / 图文理解与生成:

- `BLIP` exposes `encode_image(...)`, `encode_text(...)`, `fuse(...)`, `itm_logits`, and caption `lm_logits`.
- `BLIP2` exposes `qformer_features(...)`, `language_prefix(...)`, `caption_logits(...)`, image-text matching logits, and retrieval embeddings.
- The framework expects already-tokenized `input_ids`.
- Tokenizers, large language model weights, synthetic captions, filtering pipelines, and BLIP/BLIP-2 pre-training schedules remain outside model classes.

## Testing / 测试

```powershell
& 'D:\Project\base\base\Scripts\python.exe' -m unittest discover -s tests
& 'D:\Project\base\base\Scripts\python.exe' -m ruff check torch_like runner tests examples tools
& 'D:\Project\base\base\Scripts\python.exe' -m ruff format --check torch_like runner tests examples tools
& 'D:\Project\base\base\Scripts\python.exe' -m pytest --cov=torch_like --cov-report=term-missing
```

中文：测试通过不等于论文指标复现，只表示框架接口、导入、registry 和基础 runner 行为有效。

English: Passing tests do not imply paper-score reproduction. They validate framework interfaces, imports, registry behavior, and basic runner behavior.

## CBAM1d and CBAM-TCN / CBAM1d 与 CBAM-TCN

中文：`CBAM1d` 是处理 `(batch, channels, length)` 特征的通道与时间注意力组件，不是独立模型。`CBAMTCNDetector` 将它接入因果一维 TCN，仅使用前 `L-1` 步预测最后一步，并通过预测误差产生异常分数。ESA 消融实验未显示事件指标收益，因此该适配器默认不进入 AutoML 搜索，需要通过 `search.algorithms: [CBAMTCNDetector]` 显式启用。

English: `CBAM1d` is a channel-and-time attention component for `(batch, channels, length)` features, not a standalone model. `CBAMTCNDetector` combines it with a causal 1D TCN, predicts the final sample from the first `L-1` steps, and emits prediction-error anomaly scores. The ESA ablation did not improve event metrics, so this adapter is excluded from default AutoML discovery and must be selected explicitly with `search.algorithms: [CBAMTCNDetector]`.

```python
from torch_like.nn import CBAM1d
import torch_like as tl

attention = CBAM1d(channels=32, reduction=8, kernel_size=7)
detector = tl.CBAMTCNDetector(input_dim=6, seq_len=64, d_model=32)
```

## Practical Caveats / 实用注意事项

- 中文：新增模型应优先保证 API 一致性和可维护性。
- English: New models should prioritize API consistency and maintainability.
- 中文：不要把训练策略误认为模型结构。
- English: Do not confuse training strategies with model architectures.
- 中文：外部数据集解析优先留在 runner 或项目外层。
- English: Dataset-specific parsing should stay in runners or outside the framework.
