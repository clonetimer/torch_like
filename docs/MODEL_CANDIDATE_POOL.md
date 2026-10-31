# High-Value Model Candidate Pool / 高价值模型候选池

## Scope / 范围

本候选池用于控制 `torch_like` 的正式模型收录质量，而不是追求模型数量。评估范围覆盖当前注册表尚未充分覆盖的通用深度学习能力。最近增量评估日期为 2026-10-31；各模型的实际复核日期以复审台账为准，身份、许可证和可用状态均以对应版本为准。

This pool governs admission quality for `torch_like`; it is not a model-count target. The scope covers general deep-learning capabilities not already represented adequately in the current registry. The latest incremental evaluation is dated 2026-10-31; each candidate's actual review date is recorded below, and its status is version-specific.

- Target domain: general deep-learning architecture gaps across the registered `torch_like` domains.
- Minimum candidate count: 12.
- Evaluation date: 2026-10-31.
- Governance policy: `governance/model_admission_policy.json` version 1.0.

## Rules / 规则

- 搜索到模型不等于值得收录；缺少证据的内容标记为 unknown。
- 每个候选必须通过身份、一手来源、领域相关性、可验证评价、非重复、许可证/可用性六项硬门槛，才能正式评分。
- 每个候选至少与两个同类模型比较；采用度不能代替独立性能验证。
- 价值分：相关性 25、能力/性能 20、工程实用性 15、效率/成本 10、创新 10、生态/采用度 10、可复现性 10。
- 最终分：`价值分 × 证据系数 - 风险扣分`；A=1.00，B=0.90，C=0.75，D=0.50。
- 核心收录：最终分 >=80 且证据 A/B；观察收录：65-79 或价值高但证据不足；候选记录：评价未完成；淘汰：低于 65 或硬门槛失败。

政策配置是评分和状态转换的唯一机器可读依据；本文档中的自然语言规则必须与其一致。每个候选均有复审截止日期，新增证据、许可证变化、官方实现变化或框架能力重叠变化会触发提前复审。

The policy file is the machine-readable source of truth for scoring and state transitions. Every candidate has a review deadline; new evidence, license changes, official implementation changes, or new framework overlap trigger an earlier review.

## Review Register / 复审台账

| Model/version | Last reviewed | Review by | Owner | Early-review trigger |
|---|---|---|---|---|
| RT-DETR (CVPR 2024) | 2026-07-14 | 2027-01-14 | maintainers | benchmark, API, license, or implementation-level change |
| DreamerV3 (Nature 2025 code) | 2026-07-14 | 2027-01-14 | maintainers | independent replication or reference-code change |
| EnCodec 24/48 kHz | 2026-07-14 | 2027-01-14 | maintainers | codec benchmark, API, or license change |
| Mamba-2 / SSD | 2026-10-19 | 2026-11-19 | maintainers | independent comparison or fused-kernel evidence |
| EGNN (ICML 2021) | 2026-07-14 | 2027-01-14 | maintainers | sparse backend, benchmark, or API change |
| FITS (ICLR 2024) | 2026-10-31 | 2027-01-31 | maintainers | independent benchmark, frequency contract, or official code/license change |
| STGCN (IJCAI 2018) | 2026-10-31 | 2027-01-31 | maintainers | graph contract, independent comparison, or official code/license change |
| SAM 2.1 | 2026-10-19 | 2026-11-19 | maintainers | video protocol or independent latency evidence |
| Uni-Mol v1 | 2026-10-19 | 2026-11-19 | maintainers | molecular protocol or checkpoint-policy change |
| TabPFN v2 | 2026-10-19 | 2026-11-19 | maintainers | license, checkpoint, or independent benchmark change |
| Fourier Neural Operator | 2026-10-19 | 2026-11-19 | maintainers | operator-learning protocol or benchmark change |
| V-JEPA 2 | 2026-10-19 | 2026-11-19 | maintainers | permissive assets or broad independent validation |
| RWKV-7 Goose | 2026-10-19 | 2026-11-19 | maintainers | equal-budget independent replication |
| Gated DeltaNet-2 (2026-05) | 2026-10-19 | 2026-11-19 | maintainers | license or independent replication |
| Griffin/RecurrentGemma 1.1 | 2026-10-19 | 2026-11-19 | maintainers | terms, weights, or independent comparison change |
| SegFormer B0-B5 | 2026-07-14 | 2027-01-14 | maintainers | license or material capability-gap change |
| Titans (NeurIPS 2025) | 2026-10-19 | 2026-11-19 | maintainers | official code and explicit license become available |
| DINOv2 with registers | 2026-07-14 | 2027-01-14 | maintainers | non-duplicative random-init API identified |

## 2026-10-19 Source-and-scope Desk Review / 来源及边界复核

**Scope:** This is a limited maintenance review of **identity, current official code/terms pointers, and admission scope**, not a new hardware benchmark, independent model replication, model promotion, or production certification. Ten 2026-10-15 review deadlines had expired and blocked CI. On 2026-10-19, their source pointers and documented adoption boundaries were revisited; the original admission decisions, scores, implementation labels, and evidence grades remain unchanged. The next follow-up is due **2026-11-19** for a deeper evidence review. The 31-day interval is deliberately short because independent benchmarks and license/asset analysis remain open.

| Candidate | Checked public primary/official pointer | Desk-review decision and unresolved issue |
|---|---|---|
| Mamba-2 / SSD | https://github.com/state-spaces/mamba | Official project still supplies Mamba implementations; maintain observation/implemented; framework-side fused kernel and parity evidence remains unverified |
| SAM 2.1 | https://github.com/facebookresearch/sam2 | Official code/checkpoint disclosure visible; maintain observation; no framework video-state task/latency evidence |
| Uni-Mol v1 | https://github.com/deepmodeling/Uni-Mol | Official molecular code carries MIT license; maintain observation; no qualified molecule/geometry task adapter |
| TabPFN v2 | https://github.com/PriorLabs/TabPFN | V2 weights remain subject to version-specific Prior Labs licensing/attribution; maintain observation/no-code; no unrestricted checkpoint integration |
| Fourier Neural Operator | https://github.com/neuraloperator/neuraloperator | Maintained FNO code exists; maintain observation; operator-task contract and resolution-transfer verification remain absent |
| V-JEPA 2 | https://github.com/facebookresearch/vjepa2 | Official implementation and pretrained assets visible, but terms vary by component/asset and the project also discusses V-JEPA 2.1; maintain rejected; no approved pretrained-asset integration |
| RWKV-7 Goose | https://github.com/BlinkDL/RWKV-LM | Official RWKV-7 reference code exists; maintain rejected without a model-specific equal-budget local baseline |
| Gated DeltaNet-2 | https://github.com/NVlabs/GatedDeltaNet-2 | The official project explicitly uses NVIDIA Source Code License-NC; maintain rejected, no general-purpose distribution clearance |
| Griffin/RecurrentGemma 1.1 | https://github.com/google-deepmind/recurrentgemma | Apache-2.0 code and separately gated model weights/Gemma terms are documented; maintain rejected, no pretrained-asset dependency |
| Titans (NeurIPS 2025) | https://arxiv.org/abs/2501.00663 | Paper identified; independent third-party implementations are not evidence of an official, explicitly licensed reference implementation; hard-gate fail/rejected remains |

**Guardrails:** No newly dated source review grants a `core` promotion, new model admission, checkpoint license approval, benchmark evidence, or production readiness. Original detailed records remain authoritative until an independent technical review. The next review MUST confirm reproducibility/license status with direct evidence before any admission or promotion.

## 2026-10-31 Forecasting Admission Review / 时序预测收录评审

中文：本次从 `Workbuddy-TorchLike/report-2026-07-01.md` 的第 3 个正式条目发现 FITS，从 `report-2026-06-13.md` 的第 1 个正式条目发现 STGCN；先登记候选，再独立核验原论文、官方实现与许可证、同类模型差异及外部评价，最后按现有政策评分。6 月 13 日报告把 STGCN 指向了 `Davidham3/STSGCN`；本次以 IJCAI 2018 论文及 `VeritasYin/STGCN_IJCAI-18` 校正身份，STSGCN 不属于本次收录。既有模型的评分和结论不变。

English: FITS was discovered in formal entry 3 of `Workbuddy-TorchLike/report-2026-07-01.md`, and STGCN in formal entry 1 of `report-2026-06-13.md`. Both were recorded as candidates before checking the primary papers, official implementations/licenses, capability gaps, and independent evaluations, then scored under the unchanged policy. The June 13 report incorrectly linked STGCN to `Davidham3/STSGCN`; this review resolves the identity to the IJCAI 2018 paper and `VeritasYin/STGCN_IJCAI-18`. STSGCN is not admitted here. Existing candidate decisions and scores are unchanged.

中文：两项 `core` 结论仅批准进入实现队列。证据 A 表示原论文、官方来源和至少两个独立来源齐备，不表示本仓库已复现实验，也不授予 `benchmarked`、官方数值等价或生产就绪标签。只有导入、构造、前向、梯度、数学性质及边界测试验收后才能登记 `implemented`。

English: These two `core` decisions authorize implementation work only. Literature evidence A means that the primary paper, official source, and at least two independent sources are present; it does not establish a framework-local benchmark, numerical equivalence to the reference, or production readiness. The `implemented` qualifier requires import, construction, forward, gradient, mathematical-property, and boundary-condition acceptance tests.

中文：随后两项实现通过模型级验收，并在各自记录中登记 `implemented` 与真实 Registry 名称。该状态只表示实现与合同测试完成；成熟度和基准证据仍为 `unassessed`，未登记任何数据集性能晋升。

English: Both implementations subsequently passed model-level acceptance and are recorded below as `implemented` with their actual registry names. That status covers implementation and contract tests only; maturity and benchmark evidence remain `unassessed`, with no dataset-performance promotion.

## Hard-Gate Summary / 硬门槛汇总

| Model/version | Identity | Primary source | Relevant | Verifiable | Non-duplicate | License/availability | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| RT-DETR (CVPR 2024) | yes | yes | yes | yes | yes | Apache-2.0 | pass |
| DreamerV3 (Nature 2025 code) | yes | yes | yes | yes | yes | MIT | pass |
| EnCodec 24/48 kHz | yes | yes | yes | yes | yes | MIT | pass |
| Mamba-2 / SSD | yes | yes | yes | yes | yes: SSD is distinct from Mamba-1 | Apache-2.0 | pass |
| EGNN (ICML 2021) | yes | yes | yes | yes | yes: coordinate-equivariant API absent | MIT | pass |
| FITS (ICLR 2024) | yes | yes | yes | yes | yes: compact low-pass complex frequency interpolation absent | Apache-2.0 | pass |
| STGCN (IJCAI 2018) | yes | yes | yes | yes | yes: fixed-graph Chebyshev temporal-convolution baseline absent | BSD-2-Clause | pass |
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
| FITS | 24 | 16 | 15 | 10 | 7 | 8 | 9 | 89 | A | 4 | 85.0 | core, implemented |
| STGCN | 23 | 14 | 14 | 9 | 7 | 10 | 9 | 86 | A | 4 | 82.0 | core, implemented |
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
| FITS | adds compact spectral extrapolation beyond current forecasters | competitive published results, with independent linear-model caveats | small self-contained forecasting operator | FFT plus a truncated complex linear map | explicit low-pass amplitude/phase interpolation, not a new universal function class | independent analysis and maintained TFB integration | Apache-2.0 reference plus independent implementation; local parity pending |
| STGCN | supplied-graph convolutional baseline complements recurrent/adaptive models | established traffic results; newer independent methods often outperform it | reusable temporal/Chebyshev blocks with no pretrained assets | parallel temporal convolutions; dense graph cost must remain explicit | foundational graph-temporal composition, limited current novelty | repeated inclusion in Graph WaveNet and AutoCTS evaluations | BSD-2-Clause graph/training reference and independent comparisons; old TensorFlow portability limits |
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

### FITS (ICLR 2024)

- Category: compact frequency-domain time-series forecasting / 紧凑频域时序预测。
- Core value: low-pass filtering followed by a complex-valued linear frequency interpolation provides an explicit amplitude/phase-aware forecasting baseline with shared or per-channel weights. 中文：补充可解释的低频截断与复数插值能力，而不是增加另一个通用 MLP 预测头。
- Evidence: [primary: ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/701251e1db4a2e4dd2ef23f5265d5936-Abstract-Conference.html), [official: author implementation](https://github.com/VEWOXIC/FITS), [official: Apache-2.0 license](https://github.com/VEWOXIC/FITS/blob/main/LICENSE), [independent: Toner and Darlow, ICML 2024 analysis of linear forecasters](https://proceedings.mlr.press/v235/toner24a.html), and [independent: maintained TFB benchmark integration](https://github.com/decisionintelligence/TFB). TFB's maintained FITS baseline is the evidence here; this is not a claim that FITS was included in the original TFB paper's initial model roster.
- Peer comparison: unlike `DLinear`, it imposes an explicit low-frequency complex interpolation and spectral cutoff rather than a trend/residual time-domain projection; unlike `FreTS`, it is a compact linear spectral interpolation baseline rather than a nonlinear frequency-domain MLP. Toner and Darlow caution that several linear forecasters have closely related function classes, so architectural simplicity is not evidence of universal accuracy superiority.
- Score: value 89; evidence A; risk 4; final 85.
- Decision: core, implemented.
- Conclusion reason: passes all six hard gates and exceeds the core threshold while adding a small, independently studied spectral baseline. The dimension scores are review judgments, not benchmark measurements. 中文：通过硬门槛并补齐紧凑频域基线；文献证据等级与本仓库运行时成熟度分开管理。
- Risk deduction: 2 for cutoff sensitivity and loss of high-frequency/nonperiodic information, 1 for overlap with other linear forecasting function classes, and 1 for reference-to-framework boundary differences such as forecast-only outputs and odd-length inverse FFT handling.
- Implementation boundary: preserve mean/variance normalization, real FFT, low-pass selection, genuine complex-linear interpolation, zero padding, explicit inverse-FFT length, length-ratio amplitude correction, and denormalization. A framework forecast-only output should select the last prediction horizon from the reconstructed-plus-predicted sequence. No anomaly-detection protocol, pretrained checkpoint, official training-recipe equivalence, or local benchmark is admitted by this record.
- Acceptance: `tests/test_fits.py` covers config/state roundtrips, forward/optimizer/gradient behavior, a direct complex-DFT oracle, odd/even FFT lengths, cutoff and constant-input behavior, channel independence, future-input isolation, and invalid-input boundaries. These are mathematical and API checks, not dataset benchmarks.
- Missing information: official numerical/training-recipe parity, controlled local forecasting datasets, and CPU/CUDA cost measurements. Maturity and benchmark evidence remain `unassessed`.
- Registry: `FITS`; integration level `paper_aligned`.

### STGCN (IJCAI 2018)

- Category: fixed-graph spatio-temporal convolutional forecasting / 固定图时空卷积预测。
- Core value: a supplied graph, Chebyshev polynomial graph convolution, and temporal convolution blocks provide a reusable non-recurrent spatio-temporal baseline. 中文：补充明确依赖外部图结构的轻量基线，不学习或伪造传感器拓扑。
- Evidence: [primary: IJCAI 2018 paper](https://www.ijcai.org/proceedings/2018/505), [official: author TensorFlow implementation](https://github.com/VeritasYin/STGCN_IJCAI-18), [official: BSD-2-Clause license](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/LICENSE), [independent: Graph WaveNet, IJCAI 2019](https://www.ijcai.org/proceedings/2019/264), and [independent: AutoCTS, PVLDB 2022](https://www.vldb.org/pvldb/vol15/p971-wu.pdf). The two independent papers contain actual STGCN comparisons and are from distinct author groups.
- Peer comparison: unlike recurrent `DCRNN`, it composes temporal and Chebyshev graph convolutions without a graph-GRU rollout core; unlike `GraphWaveNet`, its graph is supplied rather than adaptively learned; unlike attention-augmented `ASTGCN`, it offers a simpler fixed polynomial graph filter. Independent Graph WaveNet and AutoCTS results also show that STGCN is not a current universal accuracy leader.
- Score: value 86; evidence A; risk 4; final 82.
- Decision: core, implemented.
- Conclusion reason: passes all six hard gates and contributes a well-documented supplied-graph baseline, with the capability/performance score deliberately below newer state-of-the-art candidates. 中文：收录依据是可验证的固定图卷积能力和基线价值，而非把经典方法包装为当前最优模型。
- Risk deduction: 2 for reliance on a meaningful fixed graph and undirected-Laplacian assumptions, 1 for dense graph scaling, and 1 for portability differences from the old TensorFlow reference, including any direct multi-horizon output adaptation.
- Implementation boundary: preserves the scaled normalized graph Laplacian, Chebyshev recurrence, channel-aligned residual temporal/graph blocks, and graph buffers. Symmetric nonnegative adjacency is required; positive edges that would underflow to zero are rejected, and the operator is derived from the same rounded graph stored in config. The official first-temporal GLU / graph ReLU / second-temporal ReLU sandwich is retained, but node-shared channel normalization and a direct multi-horizon head are framework adaptations. Graph propagation is dense and graph eigendecomposition has a one-time O(N^3) setup cost; sparse edge-linear cost and full reference parity are not claimed.
- Acceptance: `tests/test_stgcn.py` covers config/state roundtrips, forward/optimizer/gradient behavior, explicit Chebyshev-polynomial and Laplacian oracles, graph dependence, node permutation, residual alignment, graph alias/underflow/rounding defense, future-input isolation, and graph/sequence boundaries.
- Missing information: official numerical parity and controlled local traffic/sensor and CPU/CUDA benchmarks. Maturity and benchmark evidence remain `unassessed`; no production-readiness claim is made.
- Registry: `STGCN`; integration level `framework_oriented`.

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

No unscored candidate currently remains in this formal pool. New discoveries must enter here before formal scoring; they may not jump directly from a paper announcement into implementation. Discovery ledger: on 2026-10-31, FITS and STGCN were first recorded from the two reports named in the admission review above, then promoted to the Core Inclusion section only after the source, hard-gate, independent-evidence, and score reviews were complete. Neither promotion itself marks an implementation complete.

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
6. 2026-10-31 增量：FITS 与 STGCN 分别以 85 和 82 分获核心收录，随后完成模型级实现验收；不声称本仓库数据集基准已完成。 / Increment: FITS and STGCN receive core admission at 85 and 82 and subsequently pass model-level implementation acceptance; no completed framework dataset benchmark is claimed.

## Admission Order / 实施顺序

1. Mamba-2: **implemented** with full-sequence, chunked, and recurrent-state contracts.
2. EGNN: **implemented** with coordinate-equivariant message passing and E(n) transformation tests.
3. FITS: **implemented** with complex-frequency interpolation, explicit inverse-FFT length, and forecast-horizon slicing.
4. STGCN: **implemented** with supplied-graph Chebyshev/temporal convolutions, node-shared normalization, and a direct multi-horizon head.
5. Do not implement observation/rejected entries without a new evidence review and a domain protocol.
