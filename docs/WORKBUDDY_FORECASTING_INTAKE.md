# Workbuddy Forecasting Intake / Workbuddy 预测算法接入清单

## Scope and Evidence Boundary / 范围与证据边界

中文：本清单依据用户提供的 `Workbuddy-TorchLike` 文件夹中 36 篇报告，整理报告主选模型与 `torch_like` 预测模型目录之间的对应关系。报告日期覆盖 2026-06-09 至 2026-10-31。报告是文献调研，不是本仓库的实现日志、测试记录或基准结果；本清单也不是对所有候选的收录承诺。

English: This intake maps the main selections in 36 user-provided reports from `Workbuddy-TorchLike` to the `torch_like` forecasting catalog. Report dates range from 2026-06-09 to 2026-10-31. The reports are literature surveys, not repository implementation logs, test records, or benchmark results. This inventory does not commit the project to implementing every candidate.

中文：盘点日期为 **2026-10-31**。以下历史映射固定在 `main` 提交 `a403cb7a790136fa5e351a9154030e7a0ece82b9` 的 [forecasting catalog](../torch_like/models/forecasting/_catalog.py)，当时该目录有 43 个注册名称。仅检查名称及别名对应，不据此证明现有实现忠于论文、测试充分或已达到任何性能水平。其他领域的目录不在此次缺口统计范围内。

English: The inventory date is **2026-10-31**. The historical mapping below is pinned to `main` commit `a403cb7a790136fa5e351a9154030e7a0ece82b9`, whose [forecasting catalog](../torch_like/models/forecasting/_catalog.py) contained 43 registered names. A name/alias match does not establish paper fidelity, test coverage, or benchmark maturity. Other model domains are outside this gap count.

| Measure / 统计口径 | Count / 数量 | Interpretation / 解释 |
|---|---:|---|
| Reports read completely / 完整读取的报告 | 36 | Includes zero-selection and summary reports / 包括无主选项和汇总报告 |
| Main selections / 主选模型条目 | 96 | 96 distinct names in the selected sections / 主选章节共有 96 个不同名称 |
| Already mapped at the baseline / 基线已有映射 | 33 | Registered, not independently revalidated here / 已登记，但未在此重新验证 |
| Not mapped at the baseline / 基线未有映射 | 63 | Intake candidates, not 63 approved forecast backbones / 候选线索，并非 63 个已批准的预测骨干 |
| Mapped after this batch / 集成本批后已有映射 | 35 | Baseline 33 plus newly implemented FITS and STGCN / 基线 33 项加本批新实现的 FITS、STGCN |
| Remaining after this batch / 集成本批后尚无映射 | 61 | Still candidates, not promised implementations / 仍是候选线索，不是已承诺的实现 |
| Mapped after R2 / R2 后已有映射 | 36 | R1 added FITS/STGCN, R2 adds Non-stationary Transformers / R1 两项加 R2 一项 |
| Remaining after R2 / R2 后尚无映射 | 60 | Still requires separate admission and task-role review / 仍须逐项准入及任务归类 |
| Mapped after R3 / R3 后已有映射 | 37 | Adds observation-tier FiLM; no benchmark promotion / 新增观察池 FiLM，不代表基准晋升 |
| Remaining after R3 / R3 后尚无映射 | 59 | Complete names and next Pyraformer scope in [R3](WORKBUDDY_FORECASTING_R3.md) / 完整名单及下一项范围见 R3 |
| Honorable-mention occurrences / 附带提及次数 | 89 | Repetitions allowed; excluded from the 96 / 可重复，不计入 96 个主选项 |
| Reports without a main selection / 无主选项的报告 | 3 | `report-2026-06-24.md`, `report-2026-10-24.md`, `report-2026-10-25.md` |

中文：主选项按各报告实际的模型介绍章节统计，不采用汇总报告中可能重复或不一致的计数。下列表中的日期均对应原文件 `report-YYYY-MM-DD.md`；例如 `2026-07-01` 指 `Workbuddy-TorchLike/report-2026-07-01.md`。名称保留报告写法，注册键单独列出。本文只保留必要的来源元数据和工作结论，不复制整篇报告。

English: Main selections are counted from the actual model sections, not from potentially duplicated or inconsistent summary counters. Every date below identifies the original `report-YYYY-MM-DD.md`; for example, `2026-07-01` refers to `Workbuddy-TorchLike/report-2026-07-01.md`. Report names and registry keys are kept separate. Only source metadata and intake conclusions are retained here, not complete report copies.

## Existing Catalog Matches: 33 / 基线已登记的 33 项

中文：本表所有条目的状态均为“已登记、尚未在本次盘点中复核”。同名并不意味着论文复现完成。名称标准化包括 `N-BEATS` → `NBEATS`、`N-HiTS` → `NHITS`、`FC-GAGA` → `FCGAGA`、`Graph WaveNet` → `GraphWaveNet` 和 `iTransformer` → `ITransformer`。

English: Every row in this table means “registered, not revalidated in this intake.” A matching name is not a reproduction claim. Normalization includes `N-BEATS` → `NBEATS`, `N-HiTS` → `NHITS`, `FC-GAGA` → `FCGAGA`, `Graph WaveNet` → `GraphWaveNet`, and `iTransformer` → `ITransformer`.

| Report date / 报告日期 | Report model / 报告模型 | Baseline catalog key / 基线注册键 |
|---|---|---|
| 2026-06-09 | Informer | `Informer` |
| 2026-06-09 | Autoformer | `Autoformer` |
| 2026-06-09 | LogTrans | `LogTrans` |
| 2026-06-11 | MTGNN | `MTGNN` |
| 2026-06-11 | StemGNN | `StemGNN` |
| 2026-06-12 | N-BEATS | `NBEATS` |
| 2026-06-12 | DCRNN | `DCRNN` |
| 2026-06-12 | Graph WaveNet | `GraphWaveNet` |
| 2026-06-14 | AGCRN | `AGCRN` |
| 2026-06-16 | DeepAR | `DeepAR` |
| 2026-06-17 | ASTGCN | `ASTGCN` |
| 2026-06-17 | DMSTGCN | `DMSTGCN` |
| 2026-06-18 | FC-GAGA | `FCGAGA` |
| 2026-06-18 | STDN | `STDN` |
| 2026-06-19 | STFGNN | `STFGNN` |
| 2026-06-19 | TimeGrad | `TimeGrad` |
| 2026-06-20 | AST | `AST` |
| 2026-06-20 | ProTran | `ProTran` |
| 2026-06-20 | MLCNN | `MLCNN` |
| 2026-06-23 | STGODE | `STGODE` |
| 2026-06-23 | GTS | `GTS` |
| 2026-06-25 | FEDformer | `FEDformer` |
| 2026-06-26 | SCINet | `SCINet` |
| 2026-06-29 | PatchTST | `PatchTST` |
| 2026-06-29 | TimesNet | `TimesNet` |
| 2026-06-29 | DLinear | `DLinear` |
| 2026-06-30 | Koopa | `Koopa` |
| 2026-06-30 | Crossformer | `Crossformer` |
| 2026-06-30 | N-HiTS | `NHITS` |
| 2026-07-01 | iTransformer | `ITransformer` |
| 2026-07-01 | TimeMixer | `TimeMixer` |
| 2026-07-06 | DAG | `DAG` |
| 2026-07-06 | PPM | `PPM` |

## Baseline Gaps: 63 / 基线尚未登记的 63 项

中文：`—` 表示在上述基线 forecasting catalog 中没有对应键，不表示整个仓库或其他库不存在该算法。状态缩写仅描述本次工作安排，不替代治理政策的正式收录状态。

English: `—` means no matching key in the pinned forecasting catalog, not that the algorithm is absent from every repository domain or external library. The work-state abbreviations below are intake annotations, not governance admission decisions.

- **I — Implemented / 本批代码已落地：** FITS 与 STGCN 的实现、预测目录和公共按需入口已接入。此状态仅表示代码落地；两项的 `maturity` 仍为 `unassessed`，不表示全量回归、数据集基准或 Stable 验收已完成。
- **I — Implemented:** FITS and STGCN now have implementations, forecasting catalog entries, and public lazy exports. This records code integration only; both retain `maturity=unassessed`. It does not mean full regression, dataset benchmarks, or Stable acceptance is complete.
- **H — Hold / 暂缓：** ST-Norm 的官方来源许可证门槛仍为 `unknown`，须先补证。
- **H — Hold:** The official-source license gate for ST-Norm remains `unknown` and needs evidence before proceeding.
- **U — Unassessed / 未评估：** 其余条目是待评估候选，没有批准、排期或实现承诺；后续优先调研也不改变此状态。
- **U — Unassessed:** All other rows remain unassessed candidates, without approval, a delivery schedule, or an implementation promise. Follow-up research priority does not change this status.

| Report date / 报告日期 | Report model / 报告模型 | Baseline key / 基线注册键 | Work state / 工作状态 |
|---|---|---|---|
| 2026-06-11 | Z-GCNETs | — | U |
| 2026-06-13 | STGCN | — | I |
| 2026-06-13 | GMAN | — | U |
| 2026-06-13 | STSGCN | — | U |
| 2026-06-14 | DA-RNN | — | U |
| 2026-06-14 | DeepGLO | — | U |
| 2026-06-16 | DeepState | — | U |
| 2026-06-16 | STRIPE | — | U |
| 2026-06-17 | ST-MetaNet | — | U |
| 2026-06-18 | TRMF | — | U |
| 2026-06-19 | DSARF | — | U |
| 2026-06-23 | ST-Norm | — | H |
| 2026-06-25 | Pyraformer | — | U |
| 2026-06-25 | Non-stationary Transformers | — | U |
| 2026-06-26 | FiLM | — | U |
| 2026-06-26 | CoST | — | U |
| 2026-07-01 | FITS | — | I |
| 2026-07-02 | Time-LLM | — | U |
| 2026-07-02 | TEMPO | — | U |
| 2026-07-02 | MOIRAI | — | U |
| 2026-07-03 | Time-MoE | — | U |
| 2026-07-03 | Timer-XL | — | U |
| 2026-07-03 | TimeFilter | — | U |
| 2026-07-04 | SEMPO | — | U |
| 2026-07-04 | xLSTM-Mixer | — | U |
| 2026-07-04 | SKOLR | — | U |
| 2026-07-05 | Dynamic-TMoE | — | U |
| 2026-07-05 | SiGMA | — | U |
| 2026-07-05 | LatentTSF | — | U |
| 2026-07-06 | KUP-BI | — | U |
| 2026-07-07 | PULSE | — | U |
| 2026-07-07 | L-Drive | — | U |
| 2026-07-07 | SEER | — | U |
| 2026-10-19 | TStruct | — | U |
| 2026-10-19 | OliO | — | U |
| 2026-10-19 | CTIA-ODE | — | U |
| 2026-10-20 | Paloma | — | U |
| 2026-10-20 | PathFM | — | U |
| 2026-10-21 | TimeES | — | U |
| 2026-10-21 | TempoPFN | — | U |
| 2026-10-21 | Timer-M1 | — | U |
| 2026-10-22 | CVTA | — | U |
| 2026-10-22 | SAGT | — | U |
| 2026-10-22 | TimeFT | — | U |
| 2026-10-23 | Meta-iLaD | — | U |
| 2026-10-23 | DDCI | — | U |
| 2026-10-26 | CRC | — | U |
| 2026-10-26 | Chameleon | — | U |
| 2026-10-26 | CTLite | — | U |
| 2026-10-27 | REMIND | — | U |
| 2026-10-27 | DBCM | — | U |
| 2026-10-28 | TimeMagnifier | — | U |
| 2026-10-28 | PCR | — | U |
| 2026-10-28 | STeP-TS | — | U |
| 2026-10-29 | Timestral | — | U |
| 2026-10-29 | RankFormer | — | U |
| 2026-10-29 | GAS-PFN | — | U |
| 2026-10-30 | TimeH2O-v1 | — | U |
| 2026-10-30 | TimeHAT | — | U |
| 2026-10-30 | MetaARIMA | — | U |
| 2026-10-31 | FARE | — | U |
| 2026-10-31 | IMPACT | — | U |
| 2026-10-31 | IMRouter | — | U |

## First Implementation Batch / 首批实施范围

中文：本批已完成 **FITS 与 STGCN** 的代码实现、目录登记和公共按需导出。正式候选证据、来源版本、许可检查和收录决定由 [MODEL_CANDIDATE_POOL.md](MODEL_CANDIDATE_POOL.md) 记录，并遵守 [model_admission_policy.json](../governance/model_admission_policy.json)。**代码已实现、技术检查通过、基准验证和 Stable 发布是不同状态**；本清单不宣称全量回归已通过、论文指标已复现或 Stable 已发布。

English: This batch has implemented **FITS and STGCN**, registered them in the catalog, and integrated their public lazy exports. Formal candidate evidence, source versions, license checks, and admission decisions belong in [MODEL_CANDIDATE_POOL.md](MODEL_CANDIDATE_POOL.md) under [model_admission_policy.json](../governance/model_admission_policy.json). **Implemented code, passed technical checks, benchmark validation, and a Stable release are distinct states.** This inventory does not claim that full regression has passed, paper metrics have been reproduced, or Stable has been released.

| Model / 模型 | Registry key / 注册键 | Repository files / 仓库文件 | Runtime maturity / 运行时成熟度 |
|---|---|---|---|
| FITS | `FITS` | [implementation / 实现](../torch_like/models/forecasting/fits.py), [tests / 测试](../tests/test_fits.py) | `unassessed` |
| STGCN | `STGCN` | [implementation / 实现](../torch_like/models/forecasting/stgcn.py), [tests / 测试](../tests/test_stgcn.py) | `unassessed` |

中文：以下列出本批保留的核心机制与验收主题，不代表表内每一个验证项目都已完成。

English: The following lists the preserved mechanisms and acceptance topics; it is not a claim that every listed verification item is complete.

| Model / 模型 | Source report / 来源报告 | Implemented mechanism / 已实现机制 | Acceptance topics / 验证关注项 |
|---|---|---|---|
| FITS | `report-2026-07-01.md` | Low-pass complex-valued frequency interpolation, explicit inverse-transform length, and normalization/denormalization / 低通复数频域插值、显式逆变换长度及归一化与反归一化 | Fourier length/amplitude scaling; odd/even lengths; retained frequencies and DC/Nyquist behavior; finite gradients; configuration and state round-trips / 傅里叶长度与幅值尺度、奇偶长度、频率截断及 DC/Nyquist 行为、有限梯度、配置与状态往返 |
| STGCN | `report-2026-06-13.md` | Supplied-graph Chebyshev convolution between temporal GLU/ReLU convolutions, with a direct multi-horizon head / 基于外部图的 Chebyshev 卷积与时间 GLU/ReLU 卷积，采用直接多跨度输出头 | Supplied adjacency contract; Laplacian/operator math; topology sensitivity; temporal receptive field; shape/gradient/state tests / 外部邻接矩阵合同、拉普拉斯与算子数学、拓扑敏感性、时间感受野及形状/梯度/状态测试 |

中文：FITS 不能用现有 FreTS 的频域 MLP 改名替代；STGCN 不能用与输入图无关的时间网络替代。首次实现可明确限定可移植子集，但必须标明与论文机制的对应关系和未覆盖能力，不得继承论文指标或“完整复现”标签。

English: FITS must not be replaced by a renamed FreTS-style spectral MLP; STGCN must not be replaced by a temporal network that ignores the supplied graph. A first implementation may define a portable subset, provided it documents mechanism alignment and omissions without inheriting paper metrics or a “full reproduction” label.

中文：历史表仍保持 **33 已登记 / 63 未登记**。在本批代码集成后的工作树快照中，96 个主选项的对应关系为 **35 已登记 / 61 未登记**：新增的仅为 FITS 和 STGCN，不是全部 96 项均已实现，也不重新认证原有 33 项。两项新增构件均保留 `maturity=unassessed`，没有新增数据集 benchmark 证据；工作树集成也不等于已合并 `main` 或已进入 Stable。最终交付记录应另行列明实际提交、验证命令与结果。

English: The historical tables retain **33 mapped / 63 unmapped**. In the post-integration working-tree snapshot, the 96 main selections now map to **35 registered / 61 unregistered**: only FITS and STGCN are newly implemented. This does not mean all 96 are implemented or re-certify the original 33. Both new components retain `maturity=unassessed`, with no new dataset benchmark evidence. Working-tree integration is also not a merge to `main` or a Stable promotion. The final delivery record should separately identify the actual commit, validation commands, and outcomes.

### Limited Independent FITS QA / FITS 独立技术验真摘要

中文：独立 CPU 审查时，`python -m pytest -q tests/test_fits.py` 的 80 项测试通过。额外的 864 组双精度配置与原生 complex 矩阵参考计算在前向及输入/参数梯度上吻合；4 组联合输入和全部参数的有限差分 `gradcheck` 通过。额外矩阵对照是审查时运行的实验，不表示测试文件中已持久化 864 个用例。

English: During the independent CPU review, `python -m pytest -q tests/test_fits.py` passed 80 tests. An additional 864 double-precision configurations matched a native-complex matrix reference in forward values and input/parameter gradients, and four joint input/all-parameter finite-difference `gradcheck` cases passed. The extra matrix comparisons were review-time experiments, not a claim that 864 cases are persisted in the test file.

中文：该审查使用 PyTorch `2.10.0+cpu`，环境没有 CUDA；因此没有验证 cuFFT、CUDA AMP 或 GPU 设备迁移。上述结果仅支持已检查的数学和工程行为，不构成数据集精度、延迟或显存 benchmark，也不是整个仓库的全量回归或 Stable 验收结论。STGCN 的整体结果及最终集成检查由独立交付记录汇总，本文不推定其通过情况。

English: The review used PyTorch `2.10.0+cpu` without CUDA, so it did not validate cuFFT, CUDA AMP, or GPU device transfers. These results support only the checked mathematical and engineering behavior; they are not dataset-accuracy, latency, or GPU-memory benchmarks, nor a repository-wide regression or Stable acceptance result. The separate delivery record owns STGCN's overall results and final integration checks; this inventory does not presume their outcome.

### STGCN Source Correction / STGCN 来源纠正

中文：`report-2026-06-13.md` 的 STGCN 条目将代码链接指向 `Davidham3/STSGCN`。**STSGCN 与 STGCN 是不同模型**，不能据此前者的实现或许可证替代后者的证据。STGCN 的身份应对齐 Yu、Yin、Zhu 的 [论文 1709.04875](https://arxiv.org/abs/1709.04875)，代码来源纠正目标为 [VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18)。准确版本、实际许可文本和来源核验结果仍以正式候选记录为准；本清单不单独授予复用许可。

English: The STGCN entry in `report-2026-06-13.md` links to `Davidham3/STSGCN`. **STSGCN and STGCN are different models**; the former's implementation or license cannot stand in for the latter's evidence. STGCN identity must align with the Yu, Yin, and Zhu [paper 1709.04875](https://arxiv.org/abs/1709.04875), with the code-source correction targeting [VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18). Exact versions, license text, and verification outcomes remain the responsibility of the formal candidate record; this inventory does not independently grant reuse clearance.

中文：同一报告还同时出现“不需要预定义图”和道路距离邻接矩阵的描述。首批接口须明确接收并实际使用外部图，不能沿用“无图”能力声明。

English: The same report contains conflicting descriptions of requiring no predefined graph and using road-distance adjacency. The first-batch interface must explicitly accept and use its supplied graph; it must not inherit an adjacency-free capability claim.

## Prioritized Follow-up, Not Admission / 后续优先调研，不等于收录

中文：以下顺序是基于机制缺口及可验证性的工程调研顺序，不是论文性能排名。**Non-stationary Transformers 已在 R2 完成核心准入和预测骨干实现，FiLM 已在 R3 完成观察准入和实现**；Pyraformer 已完成只读源码范围核查，仍须正式评分及实现。报告链接本身不代表已核实许可证或性能。

English: The order below is an engineering research priority, not an accuracy ranking. **Non-stationary Transformers has R2 core admission and implementation; FiLM has R3 observation admission and implementation**. Pyraformer's read-only source scope is established, but formal scoring and implementation remain. Report links alone do not certify licenses or performance.

### 1. Non-stationary Transformers

Source / 来源：`report-2026-06-25.md`; [paper / 论文](https://arxiv.org/abs/2205.14415); [report-linked code / 报告所列代码](https://github.com/thuml/Nonstationary_Transformers).

中文：R2 已单独实现真正作用于注意力 logits 的学习式 `tau` 和 `delta`，并用数学对照、输入归一化、因果掩码、梯度及无未来目标泄漏测试验收。实现使用已有 PyTorch 能力，明确限定原始 `ns_Transformer` 风格的可移植子集；详见 [R2 记录](WORKBUDDY_FORECASTING_R2.md)。

English: R2 implements learned `tau` and `delta` inside attention logits and tests them against mathematical oracles, observed-only normalization, causal masking, gradient flow and future-target isolation. It uses existing PyTorch capabilities and is explicitly limited to a portable `ns_Transformer`-style subset; see the [R2 record](WORKBUDDY_FORECASTING_R2.md).

### 2. FiLM

R3 update / R3 更新：已完成观察池准入（A，77 分）及原生可移植实现，保留 `unassessed` 成熟度。新增第二份独立实验 FrAug，但其代码/划分审计尚未完成，不能称为两份完全复核的干净复现。下列原始工程核查要求保留；最新实现边界与完整剩余名单见 [R3 记录](WORKBUDDY_FORECASTING_R3.md)。Observation admission (A,77) and native portable implementation are complete; runtime maturity stays unassessed. FrAug supplies a second actual independent experiment, with code/split audit unresolved; see R3 for current scope.

Source / 来源：`report-2026-06-26.md`; [paper / 论文](https://arxiv.org/abs/2205.08897); [report-linked code / 报告所列代码](https://github.com/tianzhou2011/FiLM).

中文：先复核 Legendre 记忆投影、状态空间离散化、频率增强和重建公式，再设计实现。数学测试须覆盖离散算子、小序列直接计算对照、长度与尺度、频率截断、状态/系数的 dtype 和 device 迁移。显式评估是否需要 SciPy 等额外依赖、是否可预计算为可序列化 buffer；不能用普通注意力或 FFT 线性层替代其核心机制。

English: Review Legendre memory projection, state-space discretization, frequency enhancement, and reconstruction equations before implementation. Mathematical tests must cover the discrete operator, direct small-sequence references, length and scale behavior, frequency truncation, and dtype/device movement of state and coefficients. Explicitly assess additional dependencies such as SciPy and whether constants can be precomputed as serializable buffers. Ordinary attention or an FFT linear layer is not a substitute for the defining mechanism.

### 3. Pyraformer

Source / 来源：`report-2026-06-25.md`; [paper / 论文](https://openreview.net/forum?id=0EXmFzUn5I); [report-linked code / 报告所列代码](https://github.com/ant-research/Pyraformer).

中文：先明确金字塔层级、层内/层间连接、索引映射及任务输出，再检查掩码和不规则序列长度。若参考高效实现依赖专用算子或编译扩展，需审查依赖与许可，并设计经过等价性测试的可选加速路径。可移植的稠密掩码回退可以单独标明，但不能因此宣称保持论文的稀疏计算复杂度或速度。

English: Define the pyramid hierarchy, intra-/inter-scale connections, index mappings, and task output before testing masks and irregular sequence lengths. If the efficient reference path depends on custom operators or compiled extensions, review dependencies and licensing and make acceleration optional with equivalence tests. A portable dense-mask fallback may be documented separately, but it must not inherit sparse-complexity or speed claims from the paper.

### ST-Norm: License Hold / ST-Norm：许可证暂缓项

中文：`report-2026-06-23.md` 将代码指向 [JLDeng/ST-Norm](https://github.com/JLDeng/ST-Norm)，但本次接入记录的官方来源许可证门槛仍为 `unknown`。链接公开可访问不等于已获复用或再分发许可。待直接核对明确的许可证与具体版本后，才重新评估；不得根据报告中“开源”的字样推断许可。若门槛解除，还须验证空间/时间归一化轴、训练与推理统计、因果时间卷积和无测试数据泄漏。

English: `report-2026-06-23.md` points to [JLDeng/ST-Norm](https://github.com/JLDeng/ST-Norm), but the official-source license gate in this intake remains `unknown`. A publicly accessible link is not reuse or redistribution permission. Reassess only after directly checking explicit terms and the relevant version; do not infer a license from the report's “open source” wording. If cleared, subsequent checks must cover spatial/temporal normalization axes, training/inference statistics, causal temporal convolution, and absence of test-data leakage.

## Task and Asset Boundaries / 任务与资产边界

中文：基线的 63 个缺口（R1 后尚余 61 项、R2 后尚余 60 项）并不都是适合直接注册为 `ForecastingModel` 的独立点预测骨干。必须先识别算法真正承担的任务、输入输出及训练/推理状态，再选择正确的组件或任务接口。

English: The 63 baseline gaps (61 after R1; 60 after R2) are not all independent point-forecast backbones suitable for direct `ForecastingModel` registration. Identify each algorithm's actual role, input/output contract, and training/inference state before choosing a component or task interface.

| Category / 类别 | Examples from reports / 报告示例 | Intake boundary / 接入边界 |
|---|---|---|
| Pretrained or foundation-model workflows / 预训练或基础模型流程 | Time-LLM, TEMPO, MOIRAI, Time-MoE, Timer-XL, TempoPFN; later foundation-model claims need separate verification / 后续基础模型声明另行核验 | Separate code, weights, tokenizer/configuration, download behavior, and asset-specific terms; do not treat a random-init stub as a zero-shot model / 分开审查代码、权重、分词器/配置、下载行为和资产条款；随机初始化壳不等于零样本模型 |
| Representation learning / 表征学习 | CoST | Preserve the representation/pretraining objective and define any downstream prediction head explicitly; do not rename a generic forecasting head / 保留表征/预训练目标，明确下游预测头，不能将通用预测头改名冒充 |
| Interval calibration / 区间校准 | DDCI | Requires a calibration/interval contract and held-out-data protocol, not a fake point-forecast backbone / 需要校准/区间合同和留出数据协议，不能伪装成点预测骨干 |
| Residual correction / 残差校正 | CRC; DRiC is an additional mention, not one of the 96 main selections / DRiC 仅属额外提及，不在 96 个主选项内 | Model the wrapped forecaster and residual information boundary explicitly / 明确被包装预测器及残差信息边界 |
| Test-time adaptation / 测试时适应 | DBCM | Requires an explicit state-update protocol; never silently update using test targets / 需要显式状态更新协议，禁止暗中使用测试目标更新 |
| Factorization or other non-backbone algorithms / 分解或其他非骨干算法 | TRMF and other role-uncertain candidates / 以及其他角色待定候选 | Determine a fitting/optimization contract and proper framework location before considering a neural-model registration / 先确定拟合/优化合同与框架归属，再考虑是否适合神经模型注册 |

中文：不能为凑齐清单而把未核验的新论文名称映射到同一通用 MLP、Transformer 或图网络。也不自动下载模型权重、接纳未知条款资产，或把报告中的 SOTA、零样本、速度和内存声明写成仓库能力。LSTNet、TCN 在这些材料中主要是比较基线或组件；NLinear 作为 LTSF-Linear 家族成员出现，均不应被误算成此处 96 个主选项。

English: Do not fill the inventory by mapping unverified paper names to the same generic MLP, Transformer, or graph network. Do not automatically download checkpoints, accept assets under unknown terms, or turn survey SOTA, zero-shot, speed, or memory claims into repository capabilities. LSTNet and TCN occur primarily as comparison baselines or components in these materials; NLinear appears as an LTSF-Linear family member. None should be miscounted as one of these 96 main selections.

## Survey Issues to Resolve / 调研材料中待解决的问题

中文：以下问题来自报告内部核对，只用于提示需要回到一手来源复核；不在本文件中推断正确的论文数值、会议归属或当前代码状态。最新论文的身份、版本、官方实现和许可同样须独立核验。

English: The issues below arise from internal checks of the reports and indicate where primary-source review is required. This document does not infer corrected paper metrics, venues, or current code availability. Recent-paper identities, versions, official implementations, and terms also require independent verification.

| Report / 报告 | Issue and required treatment / 问题与处理 |
|---|---|
| 2026-06-13 | STGCN links to a different model and has contradictory graph assumptions; use the correction above / STGCN 链接错指另一模型且图假设矛盾，按上文纠正 |
| 2026-06-19 | DSARF “best” highlighting conflicts with smaller numbers in its own table; dataset counts also conflict. TimeGrad's stated win count differs from its listed winners / DSARF 最优标记与表内更小数值不符，数据集数也不一致；TimeGrad 胜出数量与列举不符 |
| 2026-06-23 | STGODE's all-best description is not supported by every row in its own table / STGODE 全部最优的表述未获其表格逐行支持 |
| 2026-06-24, 2026-06-25, 2026-06-26, 2026-06-30 | MICN/ETSformer venue or acceptance descriptions vary; establish canonical identities before citing them / MICN、ETSformer 的会议或录用表述不一致，引用前需确认规范身份 |
| 2026-06-26 | FiLM comparisons mix horizon averages and single-horizon results; do not rank them as the same protocol / FiLM 比较混用跨度平均值与单跨度结果，不能视作同一协议排名 |
| 2026-07-01 | The claim that FITS does not report MAE needs a primary-table check / FITS 未报告 MAE 的说法须回查原始表格 |
| 2026-10-25 | The weekly headline count disagrees with its listed selections; this intake counts the source model sections instead / 周报总数与其列举不符，本清单按实际主选章节计数 |
| 2026-10-27 | The report's blanket qualification claim conflicts with its own preprint/public-code caveats; survey selection is not admission approval / 一概满足条件的说法与其预印本、公开代码说明冲突，报告入选不等于准入批准 |
| 2026-10-31 | FARE's broad improvement wording exceeds what every row of the report's own table supports / FARE 的广泛提升描述并非其表格每一行均能支持 |
| All reports / 所有报告 | Dataset, split, forecast horizon, probabilistic versus point metrics, and compute budget must match before comparing scores / 比较指标前必须对齐数据集、划分、预测跨度、概率或点预测指标及计算预算 |

## Completion and Update Protocol / 完成与更新规则

中文：本清单不修改治理阈值，不向历史 baseline 文件追加名称来绕过准入，也不将“能跑通”视作性能证据。每次从候选进入代码或从开发状态进入已验证状态，都应保留可审计的记录。

English: This intake does not change admission thresholds, append names to historical baseline files to bypass governance, or treat a smoke run as performance evidence. Every transition from candidate to code, or from development to validated status, needs an auditable record.

中文：本批代码、接口边界与最终本地全量回归/隔离安装结果见 [R1 接入说明](WORKBUDDY_FORECASTING_R1.md)。本清单中的 `I` 仍只表示实现状态，不自动表示 benchmark 或 Stable 晋升。

English: See the [R1 integration note](WORKBUDDY_FORECASTING_R1.md) for implemented APIs, boundaries, final local regression and isolated-install results. The `I` state here still means implementation only, not benchmark or Stable promotion.

1. 中文：在 [MODEL_CANDIDATE_POOL.md](MODEL_CANDIDATE_POOL.md) 补齐直接来源、精确模型版本、许可/资产边界、至少两个同类比较对象及治理要求的证据与评分；未知项目保持 `unknown`。
   English: Complete direct sources, exact model version, license/asset boundaries, at least two peers, and policy-required evidence and scoring in [MODEL_CANDIDATE_POOL.md](MODEL_CANDIDATE_POOL.md). Keep unresolved items `unknown`.
2. 中文：按 [DEVELOPMENT.md](DEVELOPMENT.md) 保持模型、训练循环和数据解析的职责边界；先验证定义性数学机制，再验证形状、错误输入、有限梯度、配置及 `state_dict` 往返和任务合同。
   English: Follow [DEVELOPMENT.md](DEVELOPMENT.md) to separate model, training-loop, and data-parsing responsibilities. Validate defining mathematics before shape, invalid-input, finite-gradient, configuration, `state_dict` round-trip, and task-contract checks.
3. 中文：完成 catalog、按需导出和相应测试；运行导出生成器及 `--check`。若公共导出发生改变，执行包构建和隔离安装验证，并记录实际命令、环境、结果及跳过项目。
   English: Integrate catalog metadata, lazy exports, and relevant tests; run the export generator and its `--check`. Public-export changes require package-build and isolated-install validation. Record actual commands, environment, outcomes, and skips.
4. 中文：根据 [MODEL_EVIDENCE.md](MODEL_EVIDENCE.md) 和 [CATALOG_SCHEMA_V2.md](CATALOG_SCHEMA_V2.md) 保持实现范围、成熟度和基准证据相互独立。未有对应证据前保持 `unassessed`；论文结果、单元测试和结构性检查不构成本仓库基准。
   English: Keep implementation scope, maturity, and benchmark evidence separate under [MODEL_EVIDENCE.md](MODEL_EVIDENCE.md) and [CATALOG_SCHEMA_V2.md](CATALOG_SCHEMA_V2.md). Retain `unassessed` until supported by the required evidence. Paper results, unit tests, and structural checks are not repository benchmarks.
5. 中文：每次代码集成和验证更新都分别记录对应阶段及结果；保留 33/63 的历史映射与本批后的 35/61 快照，不把代码状态升级为未经证明的 benchmark/Stable 状态，也不把尚未评估的剩余项写成已承诺路线图。
   English: Record code integrations and verification updates as distinct stages with their own outcomes. Preserve the historical 33/63 mapping and this batch's 35/61 snapshot. Do not promote code status into unsupported benchmark/Stable status or turn remaining unassessed entries into a promised roadmap.
