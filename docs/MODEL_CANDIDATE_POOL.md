# High-Value Model Candidate Pool / 高价值模型候选池

## Scope / 范围

中文：本轮评估面向 `torch_like` 尚未覆盖的通用深度学习能力，候选数为 12。评估日期为 2026-07-14。正式收录必须能在不下载论文权重的前提下保留明确的架构价值，并兼容项目 Apache-2.0 定位。已有模型以当前 registry/catalog 为准。

English: This round evaluates 12 candidates that could fill capability gaps in `torch_like`, as of 2026-07-14. Formal inclusion requires architectural value that survives without automatically downloaded paper weights and remains compatible with the project's Apache-2.0 positioning. Existing entries are determined from the current registry/catalog.

## Method / 方法

- Value score: relevance 25, capability/performance 20, engineering utility 15, efficiency/cost 10, innovation 10, ecosystem/adoption 10, reproducibility 10.
- Evidence coefficient: A=1.00, B=0.90, C=0.75, D=0.50.
- Final score: `value score * evidence coefficient - risk deduction`.
- Core inclusion: final score >=80 and evidence A/B.
- Observation: final score 65-79, or high potential with insufficient evidence.
- Candidate record: evidence collection is incomplete.
- Rejected: final score <65 or a hard gate fails.
- Independent adoption is not treated as independent performance validation.

## Gate Summary / 硬门槛汇总

| Candidate/version | Identity | Primary source | Relevant | Verifiable evaluation | Non-duplicate | License/availability | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| RWKV-7 Goose | yes | yes | yes | yes | yes | Apache-2.0 | pass |
| Gated DeltaNet-2, May 2026 | yes | yes | yes | yes | **weak**: overlaps KDA | NVIDIA Source Code License-NC | pass with high risk |
| Griffin/RecurrentGemma 1.1 | yes | yes | yes | yes | yes | Gemma terms, gated weights | pass with risk |
| Titans, NeurIPS 2025 | yes | yes | yes | yes | yes | **official code/license unavailable** | fail |
| DINOv2 + registers | yes | yes | yes | yes | **no**: random-init architecture overlaps DINO/ViT | Apache-2.0 | fail |
| SAM 2.1 | yes | yes | yes | yes | yes | Apache-2.0 | pass |
| RT-DETRv2 | yes | yes | yes | yes | yes | Apache-2.0 | pass |
| SegFormer B0-B5 | yes | yes | yes | yes | yes | official code non-commercial | pass with risk |
| TabPFN v2 | yes | yes | yes | yes | **weak**: value depends on pretrained prior | modified Apache + attribution | pass with risk |
| DreamerV3, Nature 2025 code | yes | yes | yes | yes | yes | MIT | pass |
| EnCodec 24/48 kHz | yes | yes | yes | yes | yes | MIT | pass |
| Uni-Mol v1 | yes | yes | yes | yes | yes | MIT | pass |

Titans and DINOv2 do not enter formal scoring because they fail a hard gate. Their indicative scores are retained only to explain the decision and must not be interpreted as formal ranking.

## Dimension Rationale / 逐维评分理由

Abbreviations: R=target relevance, P=capability/performance, U=engineering utility, E=efficiency/cost, I=innovation, A=ecosystem/adoption, X=reproducibility. Every score below is tied to the evidence links in the candidate's detailed entry.

| Candidate | R /25 | P /20 | U /15 | E /10 | I /10 | A /10 | X /10 | Rationale |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| RT-DETRv2 | 24 | 18 | 14 | 9 | 8 | 9 | 9 | Fills real-time DETR gap; strong COCO trade-off; reusable detector; efficient hybrid encoder; end-to-end query design; HF/Ultralytics adoption; official Apache code/logs. |
| DreamerV3 | 24 | 19 | 14 | 7 | 9 | 9 | 9 | Fills world-model RL gap; broad 150-task evidence; reusable RSSM/imagination; high training cost; unified robust algorithm; RLlib/community adoption; paper and MIT reference reproduction. |
| EnCodec | 24 | 16 | 15 | 8 | 8 | 9 | 9 | Fills neural-codec gap; strong but no longer universal SOTA; RVQ is highly reusable; low-bitrate inference; bandwidth-scalable codec; broad audio-token use; MIT code/models and later independent codecs. |
| RWKV-7 | 23 | 18 | 12 | 10 | 9 | 9 | 8 | Relevant recurrent LM; strong author benchmarks; usable but overlaps sequence domain; constant-state inference; expressive state update; active ecosystem; Apache code/checkpoints but limited equal-budget replication. |
| SAM 2.1 | 23 | 18 | 13 | 7 | 9 | 10 | 9 | Adds video segmentation; strong image/video results; prompt/memory APIs useful; model is substantial; streaming memory is distinct; very broad adoption; Apache training/inference release. |
| Uni-Mol v1 | 22 | 17 | 12 | 7 | 9 | 8 | 8 | Opens molecular domain; strong task results; useful geometry/pair states; chemistry preprocessing cost; 3D pretraining is distinct; established scientific ecosystem; MIT code/checkpoints but local invariance tests absent. |
| TabPFN v2 | 23 | 19 | 13 | 6 | 9 | 10 | 8 | Strong tabular relevance/performance; practical on small data; GPU/checkpoint cost; learned synthetic prior is novel; broad adoption; Nature/code evidence but modified license and weight dependence reduce reproducibility. |
| SegFormer | 22 | 16 | 13 | 9 | 7 | 9 | 8 | Relevant segmentation family; established ADE20K/Cityscapes results; simple decoder; efficient B0-B5 range; moderate current novelty; broad adoption; official code exists but non-commercial terms constrain reuse. |
| Griffin/RecurrentGemma | 20 | 16 | 10 | 9 | 7 | 8 | 8 | Relevant but overlaps hybrids; credible official results; gated weights reduce utility; efficient recurrence/local attention; moderate novelty after Mamba/RetNet; Gemma ecosystem; papers/code available under restrictive terms. |
| Gated DeltaNet-2 | 20 | 18 | 11 | 9 | 8 | 4 | 7 | Relevant but near KDA duplicate; strong author-only results; reusable operator; constant-state recurrence; separate erase/write gates; adoption still unknown; code is non-commercial and independent replication is missing. |

Risk deductions are not hidden inside dimension scores: they separately represent license incompatibility, framework duplication, dependence on pretrained weights, hardware-sensitive claims, protocol gaps, and missing independent validation.

## Core Inclusion / 核心收录

### RT-DETRv2

- Category: real-time end-to-end object detection.
- Core value: closes the gap between the existing educational `DETR` and `YOLO` entries with an efficient hybrid encoder and query selection designed for real-time DETR inference.
- Evidence: CVPR paper and official Apache-2.0 PyTorch/Paddle implementation; the official repository reports adoption in Hugging Face and Ultralytics. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf), [official repository](https://github.com/lyuwenyu/RT-DETR)
- Peer comparison: compared with `DETR`, it targets substantially better latency/accuracy; compared with `YOLO`, it removes NMS and provides an end-to-end set-prediction path. YOLOv10 reports a counterexample where YOLOv10-S is faster at similar AP, so RT-DETR is not universally dominant. [YOLOv10 paper](https://arxiv.org/abs/2405.14458)
- Value: relevance 24, performance 18, utility 14, efficiency 9, innovation 8, adoption 9, reproducibility 9 = **91**.
- Evidence: **A**. Risk deduction: **4** (hardware-sensitive speed claims and implementation size).
- Final: `91 * 1.00 - 4 = 87`.
- Decision: **core inclusion**.
- Missing: a framework-local latency benchmark across CPU/CUDA remains unavailable.

### DreamerV3

- Category: model-based reinforcement learning/world model.
- Core value: adds learned dynamics, categorical latent state, imagined rollouts, and actor-critic learning from imagination, none of which exists in the current DQN/PPO/SAC/TD3-focused domain.
- Evidence: evaluated over more than 150 tasks with one configuration in the paper, later published in Nature; the MIT repository states that its reimplementation reproduces results across environments. Independent ecosystem implementation and later research use DreamerV3 as an empirical baseline. [Nature paper](https://www.nature.com/articles/s41586-025-08744-2), [official-author repository](https://github.com/danijar/dreamerv3), [RLlib implementation](https://github.com/ray-project/ray/blob/master/rllib/algorithms/dreamerv3/dreamerv3.py), [independent comparative study](https://openreview.net/pdf?id=ph68z0OGHX)
- Peer comparison: unlike `PPO`, it trains behavior through imagined trajectories; unlike `SAC`, it learns a reusable world model instead of relying only on model-free replay. The cost is much greater implementation and training complexity.
- Value: relevance 24, performance 19, utility 14, efficiency 7, innovation 9, adoption 9, reproducibility 9 = **91**.
- Evidence: **A**. Risk deduction: **6** (JAX reference, complex stop-gradient/KL details, environment-dependent validation).
- Final: `91 * 1.00 - 6 = 85`.
- Decision: **core inclusion**.
- Missing: equivalence tests against the JAX reference and long environment runs are not yet available locally.

### EnCodec 24 kHz reference architecture

- Category: neural audio codec.
- Core value: adds waveform encoding, residual vector quantization, bandwidth control, and waveform decoding. The current audio domain contains representation/speech models but no neural codec.
- Evidence: paper, MIT official code, released 24/48 kHz models, and independent ecosystem use as a codec/tokenizer baseline. [paper](https://arxiv.org/abs/2210.13438), [official repository](https://github.com/facebookresearch/encodec), [independent DAC repository](https://github.com/descriptinc/descript-audio-codec)
- Peer comparison: unlike `Wav2Vec2`, it reconstructs audio rather than only encoding representations; unlike `Whisper`, it produces discrete acoustic tokens and decoded waveforms rather than text. DAC is a strong later codec, so EnCodec is a representative foundation rather than current universal SOTA.
- Value: relevance 24, performance 16, utility 15, efficiency 8, innovation 8, adoption 9, reproducibility 9 = **89**.
- Evidence: **B**. Risk deduction: **0**.
- Final: `89 * 0.90 = 80.1`.
- Decision: **core inclusion**.
- Missing: independent apples-to-apples reconstruction metrics across speech and music are incomplete.

## Observation / 观察收录

### SAM 2.1

- Category: promptable image/video segmentation with streaming memory.
- Core value: extends the existing image-only SAM entry into video object segmentation and memory-conditioned propagation.
- Evidence: paper and Apache-2.0 official inference/training code, plus independent surgical-video robustness studies. [paper](https://arxiv.org/abs/2408.00714), [official repository](https://github.com/facebookresearch/sam2), [independent evaluation 1](https://arxiv.org/abs/2408.04593), [independent evaluation 2](https://arxiv.org/abs/2408.04098)
- Peer comparison: unlike existing `SAM`, it has streaming video memory; unlike `Mask2Former`, it is promptable and propagates object masks through video rather than performing fixed-category segmentation.
- Value: 23+18+13+7+9+10+9 = **89**. Evidence **B**. Risk **5**. Final **75.1**.
- Decision: **observation** until a video batch/state protocol is designed.
- Missing: independent quality/latency evaluation and a framework-level video tensor contract.

### Uni-Mol v1

- Category: 3D molecular representation learning.
- Core value: would establish a scientific/molecular domain with atom and pair representations, 3D-aware pretraining, property prediction, conformation generation, and docking adapters.
- Evidence: ICLR paper, MIT official repository, released checkpoints, and an independent conformation/property benchmark reporting both strengths and task-dependent parity. [paper](https://openreview.net/forum?id=6K2RM6wVqKu), [official repository](https://github.com/deepmodeling/Uni-Mol), [independent benchmark](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c00018)
- Peer comparison: unlike `Graphormer`, it explicitly models 3D molecular geometry and pair states; unlike generic `GPS`, it includes molecule-specific objectives and coordinate-aware heads.
- Value: 22+17+12+7+9+8+8 = **83**. Evidence **B**. Risk **6**. Final **68.7**.
- Decision: **observation**; valuable, but it requires a new molecular data protocol and chemistry-specific invariants before model code.
- Missing: independent benchmark synthesis and a decision between invariant and equivariant geometry APIs.

### TabPFN v2

- Category: pretrained tabular foundation model.
- Core value: strong small-tabular inference through a learned synthetic prior.
- Evidence: Nature paper and official repository; an independent clinical benchmark reports task-dependent results rather than universal superiority. v2 uses a modified Apache license with an attribution requirement, while later default weights are non-commercial. [Nature paper](https://www.nature.com/articles/s41586-024-08328-6), [official repository/license summary](https://github.com/PriorLabs/TabPFN), [independent clinical benchmark](https://www.medrxiv.org/content/10.64898/2026.02.02.26345274v1.full)
- Peer comparison: unlike `FTTransformer` and `TabNet`, its principal value comes from pretrained weights and in-context inference, not merely the architecture.
- Value: 23+19+13+6+9+10+8 = **88**. Evidence **B**. Risk **12**. Final **67.2**.
- Decision: **observation**, but not code inclusion, because a random-init `torch_like` class would not preserve TabPFN's core value and weight distribution is outside current scope.
- Missing: acceptable checkpoint policy, attribution obligations, and a useful no-pretraining architecture contract.

## Candidate Record / 候选记录

No candidate remains in this category after license and evidence verification. New or incomplete candidates should enter here before receiving a formal score.

## Rejected / 淘汰

### Gated DeltaNet-2 (May 2026)

- Category: gated delta-rule linear attention.
- Core value: separates erase and write gates and directly generalizes Gated DeltaNet/KDA.
- Evidence: only a very recent paper and official NVIDIA code are currently available; the repository uses NVIDIA Source Code License-NC. [paper](https://arxiv.org/abs/2605.22791), [official repository](https://github.com/NVlabs/GatedDeltaNet-2), [official license](https://github.com/NVlabs/GatedDeltaNet-2/blob/main/LICENSE)
- Peer comparison: more flexible erase/write control than `KimiDeltaAttention`; stronger claimed retrieval than `Mamba`; extremely close to an existing KDA implementation.
- Value: 20+18+11+9+8+4+7 = **77**. Evidence **C**. Risk **15**. Final **42.75**.
- Decision: **rejected** because the final score is below 65, the code is non-commercial, and independent validation is absent.
- Missing: independent replication, a permissive implementation source, and controlled comparison against KDA.

### RWKV-7 Goose

- Category: linear-time recurrent language model.
- Core value: dynamic state evolution, vector-valued gating, constant recurrent inference state, and an active open ecosystem.
- Evidence: COLM 2025 paper, Apache-2.0 training/inference code and released checkpoints. [paper](https://arxiv.org/abs/2503.14456), [official repository](https://github.com/BlinkDL/RWKV-LM)
- Peer comparison: more mature end-to-end LM ecosystem than `RetNet`; more architecture-specific token/channel mixing than `Mamba`; overlaps the newly added KDA/xLSTM recurrent-state capability.
- Value: 23+18+12+10+9+9+8 = **89**. Evidence **C** because controlled independent performance validation was not found. Risk **5**. Final **61.75**.
- Decision: **rejected for this round** because the final score is below 65; the architecture remains trackable.
- Missing: independent controlled comparisons using equal data, tokenizer, parameter count, and kernels.

### SegFormer B0-B5

- Category: efficient semantic segmentation Transformer.
- Core value: hierarchical MiT encoder and lightweight all-MLP decoder are useful independent of pretrained weights.
- Evidence: NeurIPS paper and official implementation. [paper](https://arxiv.org/abs/2105.15203), [official repository](https://github.com/NVlabs/SegFormer)
- Peer comparison: lighter decoder than `DeepLabV3`; simpler multiscale Transformer segmentation path than `Mask2Former`.
- Value: 22+16+13+9+7+9+8 = **84**. Evidence **B**. Risk **12** for the official non-commercial license and age relative to newer methods. Final **63.6**.
- Decision: **rejected** because the final score is below 65; reconsider only after a clean-room/license review.
- Missing: legal review and evidence that a new implementation does not reuse restricted code/assets.

### Griffin/RecurrentGemma 1.1

- Category: gated linear recurrence plus local attention.
- Core value: efficient hybrid recurrent LM.
- Evidence: Griffin and RecurrentGemma papers, official code and gated Gemma model cards. [Griffin paper](https://arxiv.org/abs/2402.19427), [RecurrentGemma paper](https://arxiv.org/abs/2404.07839), [official repository](https://github.com/google-deepmind/recurrentgemma), [model card](https://huggingface.co/google/recurrentgemma-9b)
- Peer comparison: similar hybrid goal to KimiLinear; recurrence overlaps Mamba/RetNet while local attention overlaps existing decoder models.
- Value: 20+16+10+9+7+8+8 = **78**. Evidence **B**. Risk **10**. Final **60.2**.
- Decision: **rejected** because final score is below 65 and Gemma terms complicate reuse; marginal framework value is modest.
- Missing: permissive architecture-code boundary and independent equal-budget evaluation.

### Titans, NeurIPS 2025

- Category: test-time neural long-term memory.
- Core value: learns a memory model from surprise-driven updates during inference.
- Evidence: peer-reviewed paper, but no official implementation/license. An independent reimplementation reports architectural ambiguities and that Titans does not consistently beat established baselines, although neural memory helps. [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html), [critical replication](https://arxiv.org/abs/2510.09551)
- Peer comparison: more adaptive than fixed-state `Mamba` and `xLSTM`, but much harder to specify and reproduce than both.
- Indicative value: 22+17+9+6+10+7+3 = **74**. Evidence **B**. Risk **15**. Indicative final **51.6**.
- Decision: **rejected at hard gate** because official code/license and an unambiguous reference are unavailable.
- Missing: official code, license, exact update semantics, and independent confirmation of long-context claims.

### DINOv2 with registers

- Category: self-supervised visual foundation model.
- Core value: highly transferable pretrained visual features.
- Evidence: paper and Apache-2.0 official code/weights. [paper](https://arxiv.org/abs/2304.07193), [official repository](https://github.com/facebookresearch/dinov2)
- Peer comparison: improves scale/data curation over existing `DINO`; uses a ViT backbone already represented by `ViT`. Without pretrained weights, most practical advantage disappears.
- Indicative value: 18+19+9+6+7+10+9 = **78**. Evidence **B**. Risk **8**. Indicative final **62.2**.
- Decision: **rejected at hard gate** as a duplicate random-init architecture. A future optional checkpoint adapter is more appropriate than a new core model class.
- Missing: a project policy for optional pretrained backbones.

## Round Summary / 本轮总结

1. Most valuable for inclusion: **RT-DETRv2**, **DreamerV3**, then **EnCodec**.
2. Most likely overestimated: **Titans**; novelty is high, but independent work identifies ambiguities and inconsistent wins.
3. Weakest evidence: **Gated DeltaNet-2**, because it is very recent and currently relies mainly on author results.
4. Track continuously: **SAM 2.1**, **RWKV-7**, **Uni-Mol**, and **Gated DeltaNet-2**.
5. Search bias: English-language papers and open GitHub ecosystems are overrepresented; benchmark numbers are often hardware- or training-budget-dependent; citation counts were not used as a proxy for quality; some independent validations may not have been indexed or found.

## Chinese Decision Summary / 中文决策摘要

| 模型与版本 | 类别 | 核心价值与同类比较 | 价值分 | 证据/风险 | 最终分 | 结论与缺失信息 |
|---|---|---|---:|---:|---:|---|
| RT-DETRv2 | 实时目标检测 | 相比 DETR 更重视实时效率；相比 YOLO 保持端到端集合预测 | 91 | A / 4 | 87 | 核心收录；仍缺本地跨硬件速度基准 |
| DreamerV3 | 世界模型强化学习 | 相比 PPO/SAC 增加 RSSM 与潜空间想象训练 | 91 | A / 6 | 85 | 核心收录；仍缺与 JAX 参考实现的长程等价实验 |
| EnCodec 24 kHz | 神经音频编解码 | 相比 Wav2Vec2/Whisper 可重建波形并生成离散声学 token | 89 | B / 0 | 80.1 | 核心收录；仍缺统一语音/音乐独立指标 |
| SAM 2.1 | 图像/视频提示分割 | 相比 SAM 增加流式视频记忆；相比 Mask2Former 支持提示传播 | 89 | B / 5 | 75.1 | 观察；缺少框架视频 batch/state 协议 |
| Uni-Mol v1 | 3D 分子表征 | 相比 Graphormer/GPS 增加分子 3D 几何与 pair state | 83 | B / 6 | 68.7 | 观察；缺少分子数据协议和不变/等变 API 决策 |
| TabPFN v2 | 表格基础模型 | 相比 FTTransformer/TabNet，主要价值来自预训练 synthetic prior | 88 | B / 12 | 67.2 | 观察但不实现；缺少权重与附加署名许可证策略 |
| RWKV-7 Goose | 线性时间语言模型 | 相比 RetNet/Mamba 生态更完整，但与现有递归序列域重叠 | 89 | C / 5 | 61.75 | 淘汰；缺等预算独立受控复现 |
| SegFormer B0-B5 | 语义分割 | 相比 DeepLabV3 解码器更轻；相比 Mask2Former 更简单 | 84 | B / 12 | 63.6 | 淘汰；官方代码非商业，缺 clean-room 法务确认 |
| Griffin/RecurrentGemma 1.1 | 混合递归语言模型 | 与 KimiLinear、Mamba、RetNet 重叠较高 | 78 | B / 10 | 60.2 | 淘汰；Gemma 条款与复用边界不理想 |
| Gated DeltaNet-2 | 线性注意力 | 相比 KDA 分离 erase/write gate；相比 Mamba 强调关联记忆 | 77 | C / 15 | 42.75 | 淘汰；官方代码非商业且无独立复现 |
| Titans NeurIPS 2025 | 测试时神经记忆 | 比 Mamba/xLSTM 更自适应，但更新语义更复杂 | 74* | B / 15 | 51.6* | 硬门槛淘汰；无官方代码/许可证且描述存在歧义 |
| DINOv2 + registers | 自监督视觉基础模型 | 相比 DINO/ViT 的核心优势依赖预训练数据和权重 | 78* | B / 8 | 62.2* | 硬门槛淘汰；随机初始化骨架重复 |

`*` 表示硬门槛失败后的解释性估分，不属于正式评分。

## Admission Order / 实施顺序

1. RT-DETRv2: **implemented** with multi-scale hybrid encoding, confidence query selection, auxiliary encoder outputs, and NMS-free post-processing.
2. EnCodec: **implemented** with reusable RVQ, bandwidth control, waveform reconstruction, spectral loss, and explicit encode/decode contracts.
3. DreamerV3: **implemented** with a discrete RSSM, world-model losses, latent imagination, and actor/critic losses; runner remains auxiliary.

No observation, candidate-record, or rejected item should be implemented without a new evidence review.
