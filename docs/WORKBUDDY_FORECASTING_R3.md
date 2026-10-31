# Workbuddy Forecasting R3 — FiLM / 时序预测第三批

## Outcome and admission / 实现与准入

中文：本轮只新增 `FiLM` 和 `FiLMConfig`，实现固定 Legendre 记忆投影、
频率增强及多尺度重建。证据 A、价值 82、风险扣分 5、最终 77，属于
**观察收录、已实现**，不是核心收录。运行时成熟度和数据集基准证据均保持
`unassessed`；不宣称论文指标、官方 checkpoint 或训练流程复现，也不进入精简 Stable 包。

English: This increment adds only `FiLM` and `FiLMConfig`: fixed Legendre
memory, frequency enhancement and multiscale reconstruction. Evidence A,
value82 minus risk5 gives77: **observation, implemented**, not core admission.
Runtime maturity and dataset benchmark evidence remain `unassessed`. There
is no paper-score, official-checkpoint, training-recipe or slim Stable claim.

The admission record is in [MODEL_CANDIDATE_POOL.md](MODEL_CANDIDATE_POOL.md).
Literature evidence A counts the primary paper, official source and two actual
author-independent experimental sources: TFB and FrAug. FrAug's code/split
audit is incomplete; this is not two independently code-audited clean replications.
The dBG electricity experiment with test-selected early stopping is not counted.

## Source and implementation boundary / 来源及实现边界

- Report: `Workbuddy-TorchLike/report-2026-06-26.md`, formal selection1.
- [FiLM paper v4, NeurIPS2022](https://arxiv.org/abs/2205.08897v4).
- [Official MIT code](https://github.com/tianzhou2011/FiLM/tree/2794355ff6258743a29715263414283782910521),
  pinned at `2794355ff6258743a29715263414283782910521`; `models/FiLM.py`, `utils/op.py` and LICENSE inspected.
- [TFB v2](https://arxiv.org/html/2403.20150v2), Tables6–8, actual external experiments.
- [FrAug v1](https://arxiv.org/pdf/2302.09292), section4.1 and Tables6–7,
  original/augmented FiLM runs. Results are not copied into local evidence metadata.

| Mechanism / 机制 | Implemented contract / 实现合同 | Difference or limitation / 差异或限制 |
|---|---|---|
| LPU | Native bilinear LMU solve, fixed registered A/B buffers, zero-state recurrence | Setup solved in float64 then stored in default floating dtype; dense sequential O(T*N²), not fused scan |
| FEL | Time-axis rFFT of coefficient trajectories, complex input-order→output-order mixing, explicit-length irFFT | Contiguous low modes only; no random modes or low-rank compression |
| Reconstruction | Last H Legendre-grid rows, grid excludes endpoint; filtered state index min(H−1,T−1) | Preserve source index even when a scale contains more than H observations |
| Multiscale | Default scales1/2/4, suffix min(L,sH), dt1/(sH), learned Linear(S,1) | Initialize fusion to equal mean and zero bias; no official-weight parity |
| Normalization | Detached observed-window population mean/variance, inverse normalization | No learned affine; optional `use_norm=False` |
| Capacity | `memory_order=64`, `modes=16` defaults | Official memory order256; configurable capacity, no claimed parameter-count equivalence |
| Short horizon | Effective bins min(modes,max(1,min(L,H)//2)) | H=1 retains DC instead of source's zero-bin degeneration |
| Public API | Tensor forecast, common config/predict/runtime/save-load | No tuple output, checkpoint adapter, >1000-channel compression or source training pipeline |

中文：每个尺度的内存递推计算量为 O(B·C·T·N²)，FFT 为 O(B·C·N·T·logT)，
频率系数混合为 O(B·C·M·N²)，中间轨迹占用 O(B·C·N·T) 内存。
这里 T=min(L,sH)，N 为记忆阶数，M 为保留频点数。不能无条件称为“线性低成本”算法。

English: Per scale, recurrence is O(B*C*T*N²), FFT O(B*C*N*T*logT), spectral
mixing O(B*C*M*N²), and stored trajectories O(B*C*N*T), with T=min(L,sH).
No optimized-kernel latency or unconditional linear-cost claim is made.

## Usage / 使用

```python
import torch
import torch_like as tl

model = tl.create(
    "FiLM", backend="torch_like",
    enc_in=3, dec_in=3, c_out=3,
    seq_len=24, label_len=0, pred_len=8,
    memory_order=16, modes=4, scales=(1, 2, 4),
)
history = torch.randn(2, 24, 3)
forecast = model.predict(model.make_forecast_batch(history))  # (2, 8, 3)
loss = forecast.square().mean()
loss.backward()

tl.save_model(model, "film_model")
restored = tl.load_model("film_model")
```

中文：输入为完整、有限的 `(B,L,C)` 观测窗口；B>0、L≥2、H≥1，输出通道必须
等于输入通道。decoder 数值、日历标记、外部 mask 和通用 attention 参数仅为
接口兼容而接收，实际不使用，不支持掩码观测或未来外生变量。模型和输入需要
相同的 float32/float64 dtype 与 device；内部禁用低精度 AMP。

English: Supply a complete finite `(B,L,C)` observed window, B>0, L≥2, H≥1,
c_out=enc_in. Decoder values, calendar markers, external masks and common
attention settings are accepted for interface compatibility but unused; there
is no masked-observation or future-exogenous support. Model/input dtype and
device must match (float32/float64); internal operations disable low-precision AMP.
If finite observations overflow the mean/variance or normalized history, FiLM
raises a numerical-range `ValueError` before using them. Rescale the observations
or move both data and model to float64; finite-input validation alone is not
a guarantee that all floating-point intermediate operations can be represented.
中文：有限输入也可能使统计量溢出，此时明确报数值范围错误，不静默返回 NaN；
可先缩放数据，或将数据和模型一起改为 float64。
Scales accept tuple or JSON list and serialize canonically as a list. Numerical
stability for arbitrary high orders, extreme data magnitudes and long histories
is not established by the small contract fixtures.

## Acceptance and verification / 验收与验证

| Test suite | Evidence / 证据 |
|---|---|
| `tests/test_film_layers.py` | Rational discretization/recurrence, exact Legendre grid, odd/even inverse lengths, DC cutoff, phase and coefficient orientation, functional parameter/input gradcheck |
| `tests/test_film.py` | Independent multiscale rational forecast and direct-DFT index fixture, short/H>L horizons, gradients/optimizer, normalization/detachment, channel/batch isolation, future poisoning, config and numerical validation |
| `tests/test_workbuddy_film_integration.py` | Public factory/runtime, lazy identity, float32/float64 persistence including sub-float32 fixed-buffer differences, grandfathered baseline unchanged |

Run model tests with `python -m pytest -q tests/test_film_layers.py tests/test_film.py tests/test_workbuddy_film_integration.py`.
Repository gates additionally require full pytest/coverage≥83%, ruff/format,
candidate/API/catalog audits, generated-index check, full/Stable wheel smoke,
independent review and remote Actions. Record actual run results in the PR;
this command list alone does not claim they passed. No dataset benchmark is run.

## Progress and remaining candidates / 进度与剩余候选

The historical intake remains fixed at33 matched/63 unmatched. R1 added FITS
and STGCN, R2 Non-stationary Transformers, R3 FiLM: **37 of96 names mapped,
59 remaining**, with **169 models /47 forecasters** after registration. Mapping
does not certify the old33 implementations. All existing916 protected symbols
and161 grandfathered registry entries must stay unchanged; four additive public
class/config records bring the protected-symbol count to920.

完整剩余名单如下；这是文献待核查清单，不是59项实现承诺或性能排名。
The complete remaining list is an evidence-review inventory, not a delivery
commitment or performance ranking:

| Report period / 报告期 | Count / 数量 | Remaining names / 剩余名称 |
|---|---:|---|
| June / 六月 | 13 | Z-GCNETs, GMAN, STSGCN, DA-RNN, DeepGLO, DeepState, STRIPE, ST-MetaNet, TRMF, DSARF, ST-Norm, Pyraformer, CoST |
| July / 七月 | 16 | Time-LLM, TEMPO, MOIRAI, Time-MoE, Timer-XL, TimeFilter, SEMPO, xLSTM-Mixer, SKOLR, Dynamic-TMoE, SiGMA, LatentTSF, KUP-BI, PULSE, L-Drive, SEER |
| October / 十月 | 30 | TStruct, OliO, CTIA-ODE, Paloma, PathFM, TimeES, TempoPFN, Timer-M1, CVTA, SAGT, TimeFT, Meta-iLaD, DDCI, CRC, Chameleon, CTLite, REMIND, DBCM, TimeMagnifier, PCR, STeP-TS, Timestral, RankFormer, GAS-PFN, TimeH2O-v1, TimeHAT, MetaARIMA, FARE, IMPACT, IMRouter |

### Next bounded increment: Pyraformer / 下一批候选

Read-only source inspection pinned
[`ant-research/Pyraformer@84af4db`](https://github.com/ant-research/Pyraformer/tree/84af4dbd93b7b96975b5034f0dde412005260123),
Apache-2.0. Proposed scope: bottleneck convolutional multiscale construction,
native indexed sparse pyramidal attention and direct FC head. No TVM, bundled
CUDA binary, attention decoder, pretrained assets or training-pipeline import.

Engineering requirements identified, not yet implemented:

- Append a zero prediction token: fine length L+1; use first future calendar
  mark, never future target values. Gather the last token and ancestors for FC.
- Floor-reduced scale lengths; graph's final parent absorbs ragged children,
  while strided coarse convolution drops the tail. Preserve this asymmetry.
- Bounded neighborhood gather can avoid dense S×S attention; prove graph and
  forward/backward equality against a small dense test oracle. No inherited
  TVM speed claim. Decide the batch1/top-length1 BatchNorm behavior explicitly.
- Original `long_range_main.py` uses test data for checkpoint selection; exclude
  that pipeline and use proper validation separation in any future benchmark.
- Two actual independent evaluation leads are LTSF-Linear (AAAI2023) and
  TimesNet (ICLR2023), with differing history/embedding/stationarization setups.
  PatchTST/MTST reuse Pyraformer numbers and must not be counted again. Full
  gate/score review remains required before registration.

ST-Norm remains on its existing license hold. The other57 candidates still
require identity, licensing, mechanism, independent-evidence and scope reviews;
foundation/pretrained models additionally need checkpoint terms and task fit.
