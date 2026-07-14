# High-Value Model Candidate Pool / 高价值模型候选池

## Scope / 范围

本候选池用于控制 `torch_like` 的正式模型收录质量，而不是追求模型数量。评估范围覆盖当前注册表尚未充分覆盖的通用深度学习能力。当前评估日期为 2026-07-14；模型身份、许可证和可用状态均以对应版本为准。

This pool governs admission quality for `torch_like`; it is not a model-count target. The scope covers general deep-learning capabilities not already represented adequately in the current registry. Status is evaluated as of 2026-07-14 and is version-specific.

- Target domain: general deep-learning architecture gaps across the registered `torch_like` domains.
- Minimum candidate count: 12.
- Evaluation date: 2026-07-14.

## Rules / 规则

- 搜索到模型不等于值得收录；缺少证据的内容标记为 unknown。
- 每个候选必须通过身份、一手来源、领域相关性、可验证评价、非重复、许可证/可用性六项硬门槛，才能正式评分。
- 每个候选至少与两个同类模型比较；采用度不能代替独立性能验证。
- 价值分：相关性 25、能力/性能 20、工程实用性 15、效率/成本 10、创新 10、生态/采用度 10、可复现性 10。
- 最终分：`价值分 × 证据系数 - 风险扣分`；A=1.00，B=0.90，C=0.75，D=0.50。
- 核心收录：最终分 >=80 且证据 A/B；观察收录：65-79 或价值高但证据不足；候选记录：评价未完成；淘汰：低于 65 或硬门槛失败。

## Hard-Gate Summary / 硬门槛汇总

| Model/version | Identity | Primary source | Relevant | Verifiable | Non-duplicate | License/availability | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| RT-DETR (CVPR 2024) | yes | yes | yes | yes | yes | Apache-2.0 | pass |
| DreamerV3 (Nature 2025 code) | yes | yes | yes | yes | yes | MIT | pass |
| EnCodec 24/48 kHz | yes | yes | yes | yes | yes | MIT | pass |
| Mamba-2 / SSD | yes | yes | yes | yes | yes: SSD is distinct from Mamba-1 | Apache-2.0 | pass |
| EGNN (ICML 2021) | yes | yes | yes | yes | yes: coordinate-equivariant API absent | MIT | pass |
| SAM 2.1 | yes | yes | yes | yes | yes | Apache-2.0 | pass |
| Uni-Mol v1 | yes | yes | yes | yes | yes | MIT | pass |
| TabPFN v2 | yes | yes | yes | yes | weak: value depends on pretrained prior | modified Apache/weight restrictions | pass with risk |
| Fourier Neural Operator | yes | yes | yes | yes | yes: operator-learning domain absent | BSD-3-Clause | pass |
| V-JEPA 2 | yes | yes | yes | yes | weak: random-init backbone overlaps ViT/video encoders | CC-BY-NC weights; code availability varies by component | pass with high risk |
| RWKV-7 Goose | yes | yes | yes | yes | yes | Apache-2.0 | pass |
| Gated DeltaNet-2 (2026-05) | yes | yes | yes | yes | weak: overlaps KDA | NVIDIA Source Code License-NC | pass with high risk |
| Griffin/RecurrentGemma 1.1 | yes | yes | yes | yes | yes | Gemma terms/gated weights | pass with risk |
| SegFormer B0-B5 | yes | yes | yes | yes | yes | official code non-commercial | pass with risk |
| Titans (NeurIPS 2025) | yes | yes | yes | yes | yes | official code/license unknown | **fail** |
| DINOv2 with registers | yes | yes | yes | yes | **no**: random-init value duplicates DINO/ViT | Apache-2.0 | **fail** |

Hard-gate failures do not receive a formal score. Indicative values below only explain rejection.

## Scorecard / 评分表

R=相关性，P=能力/性能，U=工程实用性，E=效率/成本，I=创新，A=生态/采用度，X=可复现性。

| Model | R/25 | P/20 | U/15 | E/10 | I/10 | A/10 | X/10 | Value | Evidence | Risk | Final | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RT-DETR | 24 | 18 | 14 | 9 | 8 | 9 | 9 | 91 | A | 4 | 87.0 | core, implemented |
| DreamerV3 | 24 | 19 | 14 | 7 | 9 | 9 | 9 | 91 | A | 6 | 85.0 | core, implemented |
| EnCodec | 24 | 16 | 15 | 8 | 8 | 9 | 9 | 89 | B | 0 | 80.1 | core, implemented |
| Mamba-2 / SSD | 23 | 18 | 14 | 10 | 9 | 9 | 9 | 92 | B | 5 | 77.8 | observation, implemented |
| EGNN | 23 | 17 | 14 | 9 | 9 | 10 | 9 | 91 | A | 4 | 87.0 | core, implemented |
| SAM 2.1 | 23 | 18 | 13 | 7 | 9 | 10 | 9 | 89 | B | 5 | 75.1 | observation |
| Fourier Neural Operator | 23 | 16 | 14 | 9 | 9 | 10 | 9 | 90 | B | 5 | 76.0 | observation |
| Uni-Mol v1 | 22 | 17 | 12 | 7 | 9 | 8 | 8 | 83 | B | 6 | 68.7 | observation |
| TabPFN v2 | 23 | 19 | 13 | 6 | 9 | 10 | 8 | 88 | B | 12 | 67.2 | observation, no code |
| V-JEPA 2 | 20 | 19 | 9 | 3 | 9 | 9 | 7 | 76 | C | 12 | 45.0 | rejected |
| RWKV-7 Goose | 23 | 18 | 12 | 10 | 9 | 9 | 8 | 89 | C | 5 | 61.8 | rejected |
| SegFormer | 22 | 16 | 13 | 9 | 7 | 9 | 8 | 84 | B | 12 | 63.6 | rejected |
| Griffin/RecurrentGemma | 20 | 16 | 10 | 9 | 7 | 8 | 8 | 78 | C | 10 | 48.5 | rejected |
| Gated DeltaNet-2 | 20 | 18 | 11 | 9 | 8 | 4 | 7 | 77 | C | 15 | 42.8 | rejected |

## Dimension Rationale / 逐维评分理由

The compact rationales below map R/P/U/E/I/A/X to the evidence links in each
detailed entry. A score is not valid merely because its arithmetic is valid;
the cited primary and independent evidence remains mandatory.

| Model | R | P | U | E | I | A | X |
|---|---|---|---|---|---|---|---|
| RT-DETR | fills real-time DETR gap | strong COCO trade-off, not universal | reusable detector | efficient hybrid encoder | NMS-free query design | HF/Ultralytics adoption | official code plus independent reviews |
| DreamerV3 | fills world-model RL gap | broad 150-task evidence | reusable RSSM/imagination | expensive training | unified robust algorithm | RLlib and research use | author reference plus multiple comparisons |
| EnCodec | fills neural-codec gap | strong, no longer universal SOTA | reusable RVQ/codec API | low-bitrate inference | bandwidth-scalable codec | broad audio-token use | MIT code/models and later codecs |
| Mamba-2 / SSD | distinct sequence operator | equal-data study supports competitiveness | full/chunk/recurrent API | linear scan and fixed state | SSM-attention duality | author and NVIDIA ecosystems | controlled study is partly author-affiliated; one independent portability study |
| EGNN | geometric graph gap | competitive molecular/dynamics evidence | reusable coordinate API | avoids higher-order irreps | E(n)-equivariant coordinate update | extensive downstream adoption | paper, MIT code, multiple independent extensions |
| SAM 2.1 | video segmentation gap | strong official and domain results | prompt/memory API | substantial model cost | streaming video memory | broad adoption | paper/code plus domain evaluations |
| Fourier Neural Operator | scientific operator gap | established PDE results, task dependent | reusable function-to-function layer | FFT global mixing | resolution-transfer operator view | mature NeuralOperator ecosystem | paper, two codebases, JMLR survey |
| Uni-Mol v1 | molecular domain gap | strong but task-dependent | useful atom/pair states | chemistry preprocessing cost | 3D molecular pretraining | established scientific use | paper/code and independent benchmark |
| TabPFN v2 | small-tabular relevance | strong under bounded regimes | practical small-data inference | checkpoint/GPU cost | synthetic prior learning | very broad adoption | official evidence plus mixed clinical result |
| V-JEPA 2 | physical video is relevant | official results are strong | random-init utility is low | internet-scale training | latent predictive world model | high attention/adoption | official evidence dominates; broad validation incomplete |
| RWKV-7 Goose | recurrent LM relevant | mostly author-controlled results | overlaps sequence domain | constant recurrent state | expressive state evolution | active ecosystem | equal-budget replication missing |
| SegFormer | segmentation remains relevant | established older benchmarks | simple decoder | efficient B0-B5 family | moderate current novelty | broad adoption | official code license constrains reuse |
| Griffin/RecurrentGemma | hybrid recurrence relevant | credible official results | gated weights reduce utility | efficient recurrence | moderate novelty after SSMs | Gemma ecosystem | official evidence dominates and terms limit reuse |
| Gated DeltaNet-2 | near-duplicate of KDA | strong author-only claims | reusable operator | constant-state recurrence | separate erase/write gates | adoption unknown | independent replication absent |

## Core Inclusion / 核心收录

### RT-DETR

- Category: real-time end-to-end object detection.
- Core value: efficient hybrid encoding and NMS-free set prediction fill the gap between educational `DETR` and `YOLO` entries.
- Evidence: [primary: CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf), [official: Apache-2.0 repository](https://github.com/lyuwenyu/RT-DETR), [independent: YOLOv10 counter-comparison](https://arxiv.org/abs/2405.14458), and [independent: DETR review comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC12252279/).
- Peer comparison: faster and more deployment-oriented than `DETR`; unlike `YOLO`, remains NMS-free, while YOLOv10 shows RT-DETR is not universally faster.
- Score: value 91; evidence A; risk 4; final 87.
- Decision: core, implemented.
- Conclusion reason: clears the core threshold with independent support and fills the real-time end-to-end detector gap.
- Missing information: framework-local CPU/CUDA latency benchmark.
- Registry: `RTDETR`; integration level `paper_aligned_portable_subset`.

### DreamerV3

- Category: model-based reinforcement learning/world model.
- Core value: discrete RSSM, imagined rollouts, and actor-critic learning add a capability absent from DQN/PPO/SAC/TD3.
- Evidence: [primary: Nature paper](https://www.nature.com/articles/s41586-025-08744-2), [official: MIT author repository](https://github.com/danijar/dreamerv3), [independent: RLlib implementation](https://github.com/ray-project/ray/tree/master/rllib/algorithms/dreamerv3), [independent: comparative study](https://openreview.net/pdf?id=ph68z0OGHX), and [independent: AAMAS control benchmark](https://aamas.csc.liv.ac.uk/Proceedings/aamas2024/pdfs/p335.pdf).
- Peer comparison: learns imagined trajectories unlike `PPO`; learns reusable dynamics unlike model-free `SAC`, at much higher implementation/training cost.
- Score: value 91; evidence A; risk 6; final 85.
- Decision: core, implemented.
- Conclusion reason: clears the core threshold and adds a non-duplicative world-model RL capability.
- Missing information: long-run equivalence tests against the JAX reference.
- Registry: `DreamerV3`; integration level `paper_aligned_portable_subset`.

### EnCodec 24 kHz

- Category: neural audio codec.
- Core value: waveform encode/decode, residual vector quantization, and bandwidth control add a reusable codec contract.
- Evidence: [primary: paper](https://arxiv.org/abs/2210.13438), [official: MIT repository](https://github.com/facebookresearch/encodec), and [independent: later DAC codec comparison](https://github.com/descriptinc/descript-audio-codec).
- Peer comparison: reconstructs audio unlike `Wav2Vec2`; emits acoustic tokens rather than text unlike `Whisper`; DAC means it is foundational rather than universal current SOTA.
- Score: value 89; evidence B; risk 0; final 80.1.
- Decision: core, implemented.
- Conclusion reason: clears the core threshold and contributes a reusable neural-codec/RVQ contract absent from the audio domain.
- Missing information: uniform independent speech/music reconstruction evaluation.
- Registry: `EnCodec`; integration level `paper_aligned_portable_subset`.

### EGNN

- Category: E(n)-equivariant geometric graph network.
- Core value: updates node features and coordinates while preserving translation, rotation, reflection, and permutation equivariance without higher-order irreducible representations.
- Evidence: [primary: ICML paper](https://proceedings.mlr.press/v139/satorras21a.html), [official: MIT author repository](https://github.com/vgsatorras/egnn), [independent: E(n)-equivariant flows](https://proceedings.neurips.cc/paper/2021/hash/21b5680d80f75a616096f2e791affac6-Abstract.html), [independent: molecular interaction model](https://pmc.ncbi.nlm.nih.gov/articles/PMC10839827/), and [independent: molecular-dynamics evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12344694/).
- Peer comparison: unlike `GCN`, it has an explicit coordinate update and Euclidean equivariance; unlike `Graphormer`, it does not rely on discrete structural biases; compared with SE(3)-Transformer it avoids expensive higher-order representations.
- Score: value 91; evidence A; risk 4; final 87.
- Decision: core, implemented.
- Conclusion reason: clears the core threshold, adds a non-duplicative geometric API, and passes local E(n)/permutation tests.
- Missing information: sparse edge-index backend and framework-local molecular benchmark validation.
- Registry: `EGNN`; integration level `paper_aligned_dense`.

## Observation / 观察收录

### Mamba-2 / Structured State Space Duality

- Category: efficient sequence state-space layer.
- Core value: SSD gives a mathematically explicit bridge between structured SSMs and attention-like matrix forms, with chunk-parallel training and recurrent inference.
- Evidence: [primary: ICML paper](https://arxiv.org/abs/2405.21060), [official: Apache-2.0 author repository](https://github.com/state-spaces/mamba), [controlled comparison: NVIDIA empirical study](https://arxiv.org/abs/2406.07887), and [independent: portable SSD verification](https://arxiv.org/abs/2603.09555).
- Peer comparison: compared with `Mamba`, SSD improves parallel training structure and reported kernel speed; compared with `RetNet`, it derives the operator from semiseparable SSM duality rather than retention. Performance remains hardware/kernel dependent.
- Score: value 92; evidence B; risk 5; final 77.8.
- Decision: observation, implemented.
- Conclusion reason: architectural value is high, but evidence grade B lowers the final score below the core threshold; the downgrade does not silently remove a stable API.
- Missing information: multiple independent controlled language-model comparisons, fused-kernel parity, and local hardware benchmarks.
- Registry: `Mamba2`; integration level `paper_aligned_portable_scan`.

### SAM 2.1

- Category: promptable video segmentation with streaming memory.
- Core value: extends image-only `SAM` with temporal memory and prompted mask propagation.
- Evidence: [primary: paper](https://arxiv.org/abs/2408.00714), [official: Apache-2.0 repository](https://github.com/facebookresearch/sam2), [independent: domain evaluation 1](https://arxiv.org/abs/2408.04593), and [independent: domain evaluation 2](https://arxiv.org/abs/2408.04098).
- Peer comparison: adds temporal memory over `SAM`; unlike `Mask2Former`, propagates prompted objects rather than fixed categories.
- Score: value 89; evidence B; risk 5; final 75.1.
- Decision: observation.
- Conclusion reason: strong video-memory value is offset by model cost and the absence of a framework video-state protocol.
- Missing information: framework video batch/state protocol and independent latency evaluation.

### Fourier Neural Operator

- Category: operator learning for parametric PDEs.
- Core value: would establish a scientific-ML domain rather than adding another forecasting model.
- Evidence: [primary: ICLR paper](https://openreview.net/forum?id=c8P9NQVtmnO), [official: implementation](https://github.com/scaomath/fourier_neural_operator), [independent: maintained NeuralOperator library](https://github.com/neuraloperator/neuraloperator), and [independent: JMLR neural-operator survey](https://www.jmlr.org/papers/v24/21-1524.html).
- Peer comparison: unlike `UNet`, learns a function-to-function map with spectral global mixing; unlike `PINN`, it is data-driven and amortizes inference but does not enforce a PDE residual by default.
- Score: value 90; evidence B; risk 5; final 76.
- Decision: observation.
- Conclusion reason: high scientific-ML value is not implementable responsibly before an operator-learning data contract exists.
- Missing information: boundary-condition API, irregular-grid policy, and local resolution-transfer tests.

### Uni-Mol v1

- Category: 3D molecular representation learning.
- Core value: introduces molecule-specific atom/pair states and geometry-aware pretraining.
- Evidence: [primary: ICLR paper](https://openreview.net/forum?id=6K2RM6wVqKu), [official: MIT repository](https://github.com/deepmodeling/Uni-Mol), and [independent: benchmark](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00018).
- Peer comparison: adds molecular 3D geometry over `Graphormer`; is chemistry-specific compared with generic `GPS`.
- Score: value 83; evidence B; risk 6; final 68.7.
- Decision: observation.
- Conclusion reason: opens a valuable molecular domain, but chemistry-specific data and geometry contracts are unresolved.
- Missing information: molecular data protocol and invariant/equivariant API decision.

### TabPFN v2

- Category: pretrained tabular foundation model for small datasets.
- Core value: provides synthetic-prior in-context inference in the small-data regime.
- Evidence: [primary: Nature paper](https://www.nature.com/articles/s41586-024-08328-6), [official: repository](https://github.com/PriorLabs/TabPFN), and [independent: clinical benchmark](https://www.medrxiv.org/content/10.64898/2026.02.02.26345274v1.full).
- Peer comparison: unlike `FTTransformer` and `TabNet`, value comes mainly from a synthetic-data prior and checkpoints, not the random-init architecture.
- Score: value 88; evidence B; risk 12; final 67.2.
- Decision: observation, no code.
- Conclusion reason: performance is valuable in a bounded regime, but most value comes from restricted pretrained assets rather than a random-init class.
- Missing information: acceptable checkpoint/attribution policy and evidence outside small IID datasets.

## Candidate Record / 候选记录

No unscored candidate currently remains. New discoveries must enter here before formal scoring; they may not jump directly from a paper announcement into implementation.

## Rejected / 淘汰

### V-JEPA 2

- Category: self-supervised video representation and action-conditioned world modeling.
- Core value: combines latent video prediction with a post-trained action-conditioned planning model.
- Evidence: [primary: paper](https://arxiv.org/abs/2506.09985) and [official: repository](https://github.com/facebookresearch/vjepa2); independent broad physical-reasoning validation remains insufficient.
- Peer comparison: richer predictive video objective than `DINO`; broader visual representation than `DreamerV3`, but unlike DreamerV3 its practical value depends on internet-scale pretrained weights and robot data.
- Score: value 76; evidence C; risk 12; final 45.
- Decision: rejected.
- Conclusion reason: final score is below 65 because practical value depends on internet-scale pretrained assets and official evidence dominates.
- Missing information: permissive checkpoint path, independent controlled validation, and a non-duplicative random-init contract.

### RWKV-7 Goose

- Category: linear-time recurrent language model.
- Core value: offers dynamic recurrent state evolution and constant-size inference state.
- Evidence: [primary: COLM paper](https://arxiv.org/abs/2503.14456), [official: Apache-2.0 repository](https://github.com/BlinkDL/RWKV-LM).
- Peer comparison: compared with `RetNet` and `Mamba`, it has a stronger end-to-end ecosystem but substantial recurrent-sequence overlap.
- Score: value 89; evidence C; risk 5; final 61.75.
- Decision: rejected.
- Conclusion reason: evidence coefficient and overlap with existing recurrent sequence models reduce the final score below 65.
- Missing information: independent equal-data/tokenizer/parameter evaluation.

### SegFormer B0-B5

- Category: efficient semantic segmentation Transformer.
- Core value: combines a hierarchical MiT encoder with a lightweight all-MLP decoder.
- Evidence: [primary: NeurIPS paper](https://arxiv.org/abs/2105.15203), [official: non-commercial repository](https://github.com/NVlabs/SegFormer), [independent: urban-scene competition](https://arxiv.org/abs/2206.12571), and [independent: cross-dataset medical benchmark](https://arxiv.org/abs/2607.02555).
- Peer comparison: lighter than `DeepLabV3` and simpler than `Mask2Former`, but licensing limits reuse.
- Score: value 84; evidence B; risk 12; final 63.6.
- Decision: rejected.
- Conclusion reason: non-commercial official code and age-adjusted marginal value reduce the final score below 65.
- Missing information: clean-room implementation and legal review.

### Griffin/RecurrentGemma 1.1

- Category: gated linear recurrence plus local-attention language model.
- Core value: provides a hybrid recurrent/local-attention architecture with bounded recurrent state.
- Evidence: [primary: Griffin paper](https://arxiv.org/abs/2402.19427), [primary: RecurrentGemma report](https://arxiv.org/abs/2404.07839), and [official: repository](https://github.com/google-deepmind/recurrentgemma).
- Peer comparison: compared with `Mamba` and `KimiLinear`, capability overlap is high and Gemma terms reduce utility.
- Score: value 78; evidence C; risk 10; final 48.5.
- Decision: rejected.
- Conclusion reason: official-only evidence, restrictive model terms, and overlap with existing hybrids reduce the final score below 65.
- Missing information: permissive architecture-code boundary and independent equal-budget evidence.

### Gated DeltaNet-2

- Category: gated delta-rule linear attention.
- Core value: separates erase and write gates in an input-dependent associative-memory update.
- Evidence: [primary: paper](https://arxiv.org/abs/2605.22791), [official: non-commercial repository](https://github.com/NVlabs/GatedDeltaNet-2).
- Peer comparison: it separates erase/write gates relative to KDA but remains close to existing `KimiLinear` and `Mamba` capabilities.
- Score: value 77; evidence C; risk 15; final 42.75.
- Decision: rejected.
- Conclusion reason: official-only evidence, non-commercial licensing, and near-duplication of KDA produce a low final score.
- Missing information: independent replication and a permissive implementation source.

### Titans (NeurIPS 2025)

- Category: test-time neural long-term memory.
- Core value: proposes surprise-driven memory updates during inference.
- Evidence: [peer-reviewed paper](https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) and [critical replication](https://arxiv.org/abs/2510.09551).
- Peer comparison: more adaptive test-time memory than `Mamba`/`xLSTM`, but update semantics are less reproducible.
- Decision: rejected at hard gate; no formal score.
- Conclusion reason: official code/license and unambiguous update semantics are unknown, so formal scoring is prohibited.
- Missing information: official code, explicit license, exact update semantics, and independent confirmation.

### DINOv2 with registers

- Category: self-supervised visual foundation model.
- Core value: provides highly transferable pretrained visual representations and register tokens.
- Evidence: [paper](https://arxiv.org/abs/2304.07193), [Apache-2.0 repository](https://github.com/facebookresearch/dinov2).
- Peer comparison: stronger pretrained features than `DINO`/`ViT`, but random-init architecture value is substantially duplicated.
- Decision: rejected at hard gate; no formal score.
- Conclusion reason: its random-init architecture duplicates existing DINO/ViT entries and most value resides in pretrained assets.
- Missing information: optional pretrained-backbone and checkpoint policy.

## Round Summary / 本轮总结

1. 最值得收录：EGNN；RT-DETR、DreamerV3、EnCodec 仍满足核心门槛。Mamba-2 因证据等级复核降为观察，但保留已实现的稳定 API。
2. 最可能被高估：V-JEPA 2 和 Titans；前者受大规模预训练驱动，后者存在实现歧义。
3. 证据最薄弱：Gated DeltaNet-2，当前主要依赖作者结果且许可证受限。
4. 持续跟踪：FNO/神经算子、SAM 2.1、Uni-Mol；应先设计领域协议，再写模型代码。
5. 潜在偏差：英文论文和开源 GitHub 生态被过度代表；硬件相关速度、训练数据重叠和作者自报结果可能抬高评分；未找到的独立复现不等于不存在。

## Admission Order / 实施顺序

1. Mamba-2: **implemented** with full-sequence, chunked, and recurrent-state contracts.
2. EGNN: **implemented** with coordinate-equivariant message passing and E(n) transformation tests.
3. Do not implement observation/rejected entries without a new evidence review and a domain protocol.
