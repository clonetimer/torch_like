# Workbuddy Forecasting R4 — Pyraformer / 时序预测第四批

## Outcome and admission / 实现与准入

中文：新增 `Pyraformer`、`PyraformerConfig`：瓶颈卷积多尺度构造、原生索引式
稀疏金字塔注意力、直接 FC 预测头。价值76、证据A、风险扣7，最终69分，
**观察收录、已实现**，不是核心收录。运行时成熟度与基准证据仍为 `unassessed`。
不宣称论文指标、官方 checkpoint、TVM 速度或完整训练流程复现，不扩充 Stable 包。

English: Adds a random-init native bottleneck CSCM, indexed sparse pyramidal
attention and FC head. Value76 × evidenceA − risk7 =69: **observation, implemented**.
Runtime/benchmark evidence stays unassessed, with no official-weight, dataset-score,
training-pipeline, optimized-kernel or slim Stable claim.

See [formal admission](MODEL_CANDIDATE_POOL.md#pyraformer-iclr-2022) and
[design](superpowers/specs/2026-10-31-pyraformer-design.md). Evidence A counts
Zeng's source-based evaluation plus TimesNet's **adapted M4** experiment; repeated
long-range values are not counted twice. If exact source-pipeline replication
were required, unchanged scores give B/61.4/rejected. That sensitivity is explicit.

## Sources and scope / 来源与边界

- Discovery: `Workbuddy-TorchLike/report-2026-06-25.md`.
- [Primary ICLR 2022 paper](https://openreview.net/pdf?id=0EXmFzUn5I), Liu et al.
- [Official source, pinned84af4db](https://github.com/ant-research/Pyraformer/tree/84af4dbd93b7b96975b5034f0dde412005260123), Apache-2.0; LICENSE, LEGAL, embedding, layers, LR model and training wrapper inspected.
- [Zeng et al., AAAI2023 v3](https://arxiv.org/pdf/2205.13504v3), external evaluation, unequal lookbacks caveat.
- [TimesNet ICLR2023 v3](https://arxiv.org/pdf/2210.02186v3), separate adapted-model M4 evaluation; not unchanged pipeline parity.

The implementation is independently written from the mathematical contract, not
a copy of ancillary TVM/Informer/Longformer code. The reference training script
selects checkpoints with **test-set** MSE. That procedure and pretrained assets
are not imported; any future benchmark must use separate validation selection.

| Mechanism / 机制 | Implemented behavior / 实现 | Boundary / 边界 |
|---|---|---|
| Prediction token | Append zero raw value plus first future mark | It is not a zero embedding; circular convolution also sees first history value |
| Embedding | Biased circular Conv1d(k3,p1), sinusoidal position, biased Linear(4,D), dropout | No CustomEmbedding or sequence-ID assets; table supports odd D and lengths beyond5000 |
| CSCM | Down projection; three strided Conv→BN→ELU; shared up projection; fine+coarse LayerNorm | Floor lengths; convolution drops remainder; BN and running statistics preserved |
| Pyramid | Local same-level neighbors and bidirectional parent/child edges | Last parent absorbs ragged children, including those dropped by convolution |
| Attention | Indexed K/V gathers, masked softmax, attention/output dropout, postnorm | No production S×S mask; dk=dv=D/heads; no external masks or dense attention output |
| FFN | GELU and two dropout sites; postnorm | Source eps1e-6; CSCM uses eps1e-5 |
| Head | Last fine node plus three ancestors, Linear(4D,H*C), no bias | Four scales only; horizon-major output; no attention/probabilistic decoder |

For total nodes S and padded neighborhood K, attention arithmetic and materialized
gathers cost O(B*S*K*D). For fixed factors/window, K≤inner_size+2*max(factors),
so this is linear in S. Projections/FFN add O(B*S*(D²+D*F)) per encoder layer;
the head adds O(B*D*H*C), with 4D*H*C weights. This is not a measured speedup or
an unconditional constant graph-distance claim for arbitrarily long histories.

## Usage and boundaries / 用法与约束

```python
import torch
import torch_like as tl

model = tl.create(
    "Pyraformer",
    backend="torch_like",
    enc_in=3,
    dec_in=3,
    c_out=3,
    seq_len=95,
    label_len=0,
    pred_len=12,
    d_model=32,
    n_heads=4,
    d_ff=64,
    window_size=(4, 4, 4),
    d_bottleneck=16,
)
history = torch.randn(2, 95, 3)
forecast = model.predict(model.make_forecast_batch(history))  # (2,12,3)
forecast.square().mean().backward()
model.eval()  # Required for batch-one inference when the top level has length1.
tl.save_model(model, "pyraformer_model")
restored = tl.load_model("pyraformer_model").eval()
```

中文：默认四层需要 `seq_len>=63`；恰好三个下采样因子，每个≥2，局部窗口
`inner_size` 必须为正奇数。`d_model>=2` 且能被头数整除；`c_out==enc_in`。
训练要求 `batch_size * 顶层长度 > 1`，不满足则在任何 BN 状态改变前明确报错。
推理可用 batch1。输入和模型需相同 float32/float64 与 device，内部禁用低精度 AMP。

English: Factors accept tuple/JSON list and serialize as a list. Markers are
optional continuous four-feature tensors `(B,L,4)` and `(B,label_len+H,4)`.
Only the decoder mark at index `label_len` is consumed. Decoder values, prefix
marks and later future marks cannot change the forecast. Missing marks become
zeros. External masks, non-GELU activation and `output_attention=True` are rejected.
`d_layers`, `embed`, `freq` are unused compatibility fields. Used values must be
finite; finite-input validation is not a proof of representable intermediates
for arbitrary extreme magnitudes, weights or long training runs.

中文：未来目标值完全不使用；只使用第一个未来时间标记，不支持任意未来外生变量
序列或缺失观测掩码。没有观测窗口归一化；调用方应按训练协议缩放数据。

Numerical boundary observed during review: a float32 input scaled to 1e20 can
produce nonfinite forecasts **and nonfinite BN running variance**. There is no
transactional rollback for numerical overflow. Rescale before training; after
such a failed numerical run, restore a clean checkpoint or recreate the model.
Zero padded-attention weights cannot sanitize nonfinite projected values either.
This is outside the bounded contract fixtures, not a guarantee of safe extremes.
中文：极端有限值也可能污染 BN 运行统计；并非所有错误都能自动回滚。需先缩放数据，
出现非有限结果后恢复干净 checkpoint 或重新建模，不能只更换输入就继续使用损坏状态。

Accepted NumPy integer architecture parameters are converted to Python integers
before geometry and layer arithmetic, and public save/load handles NumPy scalars.
`config()` still retains original scalar types through the common BaseModel
capture; direct `json.dumps(model.config())` with NumPy-valued fields requires
caller conversion (for example `.item()`). Native Python-valued configs round-trip
directly as JSON; the supported `save_model`/`load_model` path needs no such conversion.

## Verification / 验证

| Suite | What it establishes / 验证内容 |
|---|---|
| `test_pyraformer_graph.py` | Literal ragged neighbors/ancestors, interval oracle, symmetry, no duplicates, bounded storage |
| `test_pyraformer_attention.py` | Dense forward/all-parameter gradients, gradcheck, padding invariance and dropout |
| `test_pyraformer.py` | Explicit convolution/BN oracle and state, full dense forecast/gradients, future isolation, boundary errors, float32/64 training, runtime no-S×S allocation check |
| `test_workbuddy_pyraformer_integration.py` | Public factory/runtime/lazy API; exact mixed long/bool/float32/64 buffers and BN state save/load; unchanged grandfathered baseline |

Run `python -m pytest -q tests/test_pyraformer_graph.py tests/test_pyraformer_attention.py tests/test_pyraformer.py tests/test_workbuddy_pyraformer_integration.py`.
Merge additionally requires full pytest/coverage≥83%, Ruff/format, candidate/API/
catalog/index checks, AutoML protocol validation, full+Stable wheel smoke,
independent review and all remote CI workflows. Actual run evidence belongs in
the PR; this checklist is not a claim that every gate has already run. No dataset
benchmark or CUDA runtime test is supplied by these CPU contract fixtures.

## Progress and remaining candidates / 进度与剩余候选

Historical intake:33 matched/63 unmatched names from36 reports and96 unique formal
names. R1 FITS/STGCN, R2 Non-stationary Transformers, R3 FiLM, R4 Pyraformer give
**38/96 mapped,58 remaining**. Catalog: **170 total /48 forecasting**. All920 old
API records remain unchanged;4 additive records give924. The161-entry grandfathered
baseline, Stable list, workflows and package version are not expanded/changed.
Mapping does not certify the original33 implementations.

| Report period / 报告期 | Count | Remaining names / 剩余名称 |
|---|---:|---|
| June / 六月 | 12 | Z-GCNETs, GMAN, STSGCN, DA-RNN, DeepGLO, DeepState, STRIPE, ST-MetaNet, TRMF, DSARF, ST-Norm, CoST |
| July / 七月 | 16 | Time-LLM, TEMPO, MOIRAI, Time-MoE, Timer-XL, TimeFilter, SEMPO, xLSTM-Mixer, SKOLR, Dynamic-TMoE, SiGMA, LatentTSF, KUP-BI, PULSE, L-Drive, SEER |
| October / 十月 | 30 | TStruct, OliO, CTIA-ODE, Paloma, PathFM, TimeES, TempoPFN, Timer-M1, CVTA, SAGT, TimeFT, Meta-iLaD, DDCI, CRC, Chameleon, CTLite, REMIND, DBCM, TimeMagnifier, PCR, STeP-TS, Timestral, RankFormer, GAS-PFN, TimeH2O-v1, TimeHAT, MetaARIMA, FARE, IMPACT, IMRouter |

This is an evidence-review inventory, not58 promised implementations. Next bounded
research target: GMAN's spatial/temporal attention and spatial-embedding contract;
identity, source license, asset boundary and independent evaluation must be
reviewed before formal scoring or code. ST-Norm remains on its existing license
hold. CoST needs a representation-learning contract; pretrained/foundation
models need weight-specific terms and task interfaces, not random-init stubs.

### GMAN read-only reconnaissance / GMAN 只读预研

The next source is pinned, **not formally admitted or implemented**:
[`zhengchuanpan/GMAN@45ed232`](https://github.com/zhengchuanpan/GMAN/tree/45ed232f4ee6a7ee9dd248d07f4952b697ecf320),
Apache-2.0, AAAI2020. The official runtime is TensorFlow1.14; its linked PyTorch
port is third-party. The proposed boundary is a separately reviewed native
spatial/temporal attention, gated fusion and historical-to-future transform
attention model with **supplied fixed spatial embeddings** and integer calendar.
Do not silently bundle node2vec, dataset assets or pretrained weights.

Source issues to resolve before design/scoring:

- `METR/model.py` defaults temporal masking to True; `PeMS/model.py` to False.
  Neither is a universally applicable official default.
- Actual marker order is weekday then slot-of-day, despite a reversed GMAN
  function docstring. Future calendar is a real input; future traffic is not.
- Spatial attention is dense. The offline spatial embeddings are graph-derived;
  no sparse or graph-free end-to-end claim is justified.
- [TESTAM](https://arxiv.org/html/2403.02600v1) has case-specific/compute evaluation
  leads, but its main table uses published numbers after checking reruns.
  [CASAformer](https://doi.org/10.1016/j.commtr.2025.100174) is a second reported
  rerun lead; exact revisions, split/normalization and spatial-embedding handling
  still need source-level audit. These leads are not yet an assigned evidence grade.

中文：下一项 GMAN 已核对官方版本、Apache 许可证和主要接口差异；仍需先明确
时间 mask 变体、空间嵌入来源及日历协议，再做正式准入，当前不增加已实现数量。
