# Workbuddy Forecasting R5 — GMAN / 时序预测第五批

## Outcome and admission / 实现与准入

中文：新增原生 `GMAN`、`GMANConfig`，保留空间/时间注意力、门控残差融合和
历史到未来的变换注意力。固定空间嵌入与未来日历必须由调用方显式提供。
价值76、证据A、风险扣6，最终**70分，观察收录、已实现**，不晋升核心。
运行时成熟度及基准证据保持 `unassessed`，不扩充 Stable，不声称论文成绩复现。

English: Native random-init GMAN with owned fixed spatial embeddings and required
raw calendar. Value76 × A − risk6 =70, **observation, implemented**. Model/API
acceptance is separate from dataset accuracy, training convergence, TF checkpoint
compatibility and measured deployment cost; those remain unassessed.

See [formal admission](MODEL_CANDIDATE_POOL.md#gman-aaai-2020),
[design](superpowers/specs/2026-11-01-gman-design.md) and
[plan](superpowers/plans/2026-11-01-gman.md).

## Evidence and source audit / 证据及来源复核

- Discovery: `Workbuddy-TorchLike/report-2026-06-13.md`.
- [Primary AAAI2020 paper](https://arxiv.org/abs/1911.08415v2), Zheng/Fan/Cheng Wang/Qi.
- [Official source pinned45ed232](https://github.com/zhengchuanpan/GMAN/tree/45ed232f4ee6a7ee9dd248d07f4952b697ecf320), [Apache-2.0 LICENSE](https://github.com/zhengchuanpan/GMAN/blob/45ed232f4ee6a7ee9dd248d07f4952b697ecf320/LICENSE). METR/PeMS models, helpers and loaders were inspected; official runtime is TensorFlow1.14. The README-linked PyTorch implementation is third-party and is not vendored.
- [TESTAM, ICLR2024](https://arxiv.org/html/2403.02600v1): count Appendix B compute and Appendix C new case experiments as **one** independent source. Its main table explicitly uses published baseline values and is excluded from independent numerical evidence.
- [CASAformer,2025](https://doi.org/10.1016/j.commtr.2025.100174): publisher-indexed primary text and original speed-regime baseline tables count as the second independent experiment. Live publisher retrieval was unavailable; a complete manuscript/baseline pipeline was not inspected. Authors do not overlap GMAN; Jianping Wang is not Cheng Wang.

This A grade is **paper-level experimental evidence**, not two reproducible
released GMAN pipelines. Reviewed external trees
[TESTAM20ee781](https://github.com/HyunWookL/TESTAM/tree/20ee7815a053c4a7b64dca676581d88a68ae5078)
and [CASAformer b0c9c12](https://github.com/Yifanny/CASAformer/tree/b0c9c12fcdc4a98cef15bd1dbcb7669fab9ad21a)
do not expose GMAN runners/checkpoints/log chains. TESTAM's generic released
70/10/20 window splitter does not itself implement its paper's EXPY calendar-month
split. CASAformer baseline revisions, embeddings and normalization provenance
remain unaudited. Its future-speed bins are retrospective evaluation, not an
inference-time oracle. No code/assets from these studies are imported.

Requiring executable independent pipeline audits would instead give evidence B:
76×.9−6=**62.4/rejected**, with no score increase to preserve eligibility.
The 6-point risk deduction covers source/variant portability, supplied-embedding
provenance/node alignment, and external baseline provenance,2 points each.

## Architecture and deliberate adaptations / 机制与明确适配

| Component / 组件 | Implemented behavior / 实现 | Boundary / 边界 |
|---|---|---|
| Spatial embedding | Owned persistent SE[N,D], pointwise D→D→D | No synthesized node IDs or automatic node2vec; node order is caller-owned |
| Calendar embedding | One-hot weekday7 then slotT, pointwise(7+T)→D→D | Raw integers, not normalized four-column Informer marks |
| Spatial attention | ReLU Q/K/V from concat(X,STE), contiguous heads, all-node softmax | Dense N×N; includes self, no adjacency mask |
| Temporal attention | Same conditioned Q/K/V, per-node time softmax | Optional lower triangle includes diagonal; no arbitrary external masks |
| Fusion | sigmoid(Ws*HS+Wt*HT+bt), weighted mix, two-layer map, residualX | Spatial gate has no bias; gates have no BN; no LayerNorm |
| Transform bridge | Future STE queries, historical STE keys, encoded traffic values | Per node Q×P attention, no future target values |
| Output | D→D→1 and last-axis squeeze | Dropout only before both output affine maps; direct multihorizon output |

All two-layer maps are affine→optional population BN→ReLU→affine. BN is never
applied to a non-activated affine. It reduces all nonchannel axes, eps1e-3,
gamma1/beta0, without unbiased variance correction; singleton reductions work.
**Initial running variance is1 instead of the TF tensor-EMA zero**, to make cold
eval usable; fixed `bn_decay=.9` replaces the original training scheduler. The
mean/variance update is decay*old+(1−decay)*population_moment. The temporal mask
uses negative infinity rather than the source's finite sentinel.

Default `temporal_mask=False`, `dropout=0`, `e_layers=1` is PeMS-like. Setting
True,.1,5 chooses METR architectural settings. `e_layers` is the block count in
both encoder and decoder; `d_layers`, `d_ff`, `embed`, `freq` are unused common
compatibility fields. `activation` must be `relu`, `output_attention` must beFalse.
These variants are not official weight/training-result parity.

Training BN mixes all times before attention, even when the attention mask is
causal. The full model is **not an online prefix-causal forecaster**. Future
calendar is a legitimate known covariate; decoder traffic never participates.

For an ST block of lengthS, attention costs O(B*D*(S*N²+N*S²)); stored scores
cost O(B*K*(S*N²+N*S²)). Pointwise projections add O(B*S*N*D²). The transform
bridge adds O(B*N*P*Q*D) attention and O(B*K*N*P*Q) scores, plus its projections.
Calendar projection also costs O(B*(P+Q)*(7+T)*D). This is not sparse attention,
an edge-linear implementation, a measured speedup, or a graph-free end-to-end
pipeline: the supplied SE is commonly generated offline from a graph.

## Usage and input safety / 用法与输入安全

```python
import torch
import torch_like as tl

# A small synthetic API example only, NOT useful graph embeddings or a benchmark.
nodes, width, history, horizon = 3, 8, 4, 2
spatial_embedding = torch.linspace(-1, 1, nodes * width).reshape(nodes, width)
model = tl.create(
    "GMAN",
    backend="torch_like",
    enc_in=nodes,
    dec_in=nodes,
    c_out=nodes,
    seq_len=history,
    label_len=0,
    pred_len=horizon,
    d_model=width,
    n_heads=2,
    spatial_embedding=spatial_embedding,
    time_slots_per_day=288,
)
x = torch.randn(2, history, nodes)  # Real applications must use training-fitted scaling.
past = torch.tensor([[[0, 0], [0, 1], [0, 2], [0, 3]]]).repeat(2, 1, 1)
future = torch.tensor([[[0, 4], [0, 5]]]).repeat(2, 1, 1)
batch = model.make_forecast_batch(x, x_mark_enc=past, x_mark_dec=future)
prediction = model.predict(batch)  # (2,2,3)
prediction.square().mean().backward()
model.eval()
tl.save_model(model, "gman_model")
restored = tl.load_model("gman_model").eval()
torch.testing.assert_close(restored.predict(batch), model.predict(batch))
```

- `c_out==enc_in`; one scalar per node, not a hidden node-remapping projection.
  Hidden width is positive and divisible by heads. Label length is0..history.
- SE must be dense, real, finite, non-boolean and exactly[N,D]. Negative/zero
  entries are valid. Reject overflow or nonzero underflow on initial construction-dtype
  conversion. SE is cloned/detached; changing constructor data or returned config
  cannot mutate it. Config captures the current buffer as owned nested lists.
- Input/model must match float32/float64 and device. Autocast is disabled within
  this path. No CUDA/half/bfloat16 performance or parity claim is made.
- Optional `dtype="float64"` (or `torch.float64`; likewise float32) chooses the
  construction dtype without changing global state. DefaultNone uses the process
  default if supported. Config records the current dtype, so valid float64-only
  SE values survive public reload in a normal float32-default process. Random
  parameters are initialized under ambient dtype then converted, as by .double();
  stored weights restore exactly. `GMAN.from_config` does not recursively deepcopy
  computed Tensor embeddings before the constructor detaches and owns them.
- Both calendar tensors are required: past[B,P,2], decoder[B,label_len+Q,2],
  with uint8/int8/int16/int32/int64 weekday0..6 and slot0..T−1. Used integer values
  are promoted before range checks, avoiding fixed-width bound wraparound.
- Only the decoder suffix is used. Its prefix values and all decoder traffic
  values are ignored. Invalid used calendar is rejected **before any BN update**.
  No invented calendar defaults, timezone conversion or cadence validation.
- Finite input is necessary, not sufficient for stability at extreme magnitudes.
  Large finite values can overflow attention or BN statistics. This path does not
  roll back earlier running-stat updates after numeric failure. Scale data; after
  nonfinite output/statistics restore a clean checkpoint or recreate the model.
- Used Integral/Real config scalars are canonical Python numbers before arithmetic
  and JSON output; arbitrary unused compatibility objects are not a JSON contract.

## Verification / 验证

| Suite | Coverage of behavior / 行为验证 |
|---|---|
| test_gman_layers.py | Independent node/time/head loops, transform source binding, fusion/residual, all-parameter gradients, gradcheck, population-BN state, calendar order, output dropout |
| test_gman.py | Full forecast and gradients, optimizer, joint node/SE permutation, future-calendar dependence, decoder-value isolation, singleton/noncontiguous/L5 cold eval, validation before state mutation, fixed-width types and AMP boundary |
| test_workbuddy_gman_integration.py | Public factory/runtime/lazy exports, exact SE/BN/parameter save-load in float32/64 and Python/NumPy widths, unchanged grandfathered baseline |

Run `python -m pytest tests/test_gman_layers.py tests/test_gman.py tests/test_workbuddy_gman_integration.py -q`.
Merge additionally requires full coverage>=83%, lint/format, policy/API/catalog/
lazy-index and AutoML validators, full+Stable installed-wheel smoke, independent
whole-branch review and quality/DDP/Stable workflows. The actual run evidence is
in the PR; this required checklist is not a claim that every gate already ran.

## Progress and remaining candidates / 进度与剩余候选

Historical baseline33 matches from36 reports/96 unique formal names; R1 adds
FITS/STGCN, R2 Non-stationary Transformers, R3 FiLM, R4 Pyraformer, R5 GMAN:
**39/96 mapped,57 remaining**. Catalog171 total/49 forecasting;4 additive API
records give928, with all prior924 unchanged. Grandfathered161, Stable scope,
version, dependencies and workflows are unchanged. Historical33 matches are not
newly verified reproductions.

| Report period / 报告期 | Count | Remaining names / 剩余名称 |
|---|---:|---|
| June / 六月 | 11 | Z-GCNETs, STSGCN, DA-RNN, DeepGLO, DeepState, STRIPE, ST-MetaNet, TRMF, DSARF, ST-Norm, CoST |
| July / 七月 | 16 | Time-LLM, TEMPO, MOIRAI, Time-MoE, Timer-XL, TimeFilter, SEMPO, xLSTM-Mixer, SKOLR, Dynamic-TMoE, SiGMA, LatentTSF, KUP-BI, PULSE, L-Drive, SEER |
| October / 十月 | 30 | TStruct, OliO, CTIA-ODE, Paloma, PathFM, TimeES, TempoPFN, Timer-M1, CVTA, SAGT, TimeFT, Meta-iLaD, DDCI, CRC, Chameleon, CTLite, REMIND, DBCM, TimeMagnifier, PCR, STeP-TS, Timestral, RankFormer, GAS-PFN, TimeH2O-v1, TimeHAT, MetaARIMA, FARE, IMPACT, IMRouter |

These57 are an evidence-review inventory, not promised implementations.
Read-only follow-up STSGCN source is pinned to
[`Davidham3/STSGCN@3f825f6`](https://github.com/Davidham3/STSGCN/tree/3f825f6421afb84fb69ef560179bf799fbf4481c).
The repository license metadata is null and its complete tree contains no
LICENSE/COPYING/NOTICE. Permission is **unknown**, not an inferred MIT/BSD license;
do not implement from this source until that gate is resolved. It remains unscored
and unimplemented. ST-Norm retains its earlier license hold. CoST needs a
representation-learning protocol; pretrained/foundation entries need weight terms
and real task interfaces, not random-init placeholders.
