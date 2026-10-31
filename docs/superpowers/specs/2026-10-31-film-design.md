# FiLM portable forecasting design / 可移植预测设计

## Scope / 范围

Implement the FiLM selection in `Workbuddy-TorchLike/report-2026-06-26.md`.
The user has asked to continue implementation and merging without repeating
design-choice questions. This increment is FiLM only; Pyraformer is scoped next.
Keep the existing task checkout on a new feature branch; do not touch PR #30.

Admission is **observation**, evidence **A**, value **82**, risk **5**, final
**77**. A counts primary + official + two author-independent experimental
sources (TFB and FrAug), not two code-audited clean replications. FrAug's
implementation/split audit is unresolved. No score inflation for core entry.

## Source and mechanism / 来源及机制

- Paper: https://arxiv.org/abs/2205.08897v4 (NeurIPS 2022).
- Official MIT source: `tianzhou2011/FiLM` at
  `2794355ff6258743a29715263414283782910521`, `models/FiLM.py` and `utils/op.py`.
- Implement fixed Legendre Projection Unit (LPU), frequency-domain coefficient
  mixing (FEL), polynomial reconstruction and learned multiscale fusion.
- For row i, column j: `A[i,j]=(2*i+1)*(-1 if i<j else (-1)**(i-j+1))`,
  `B[i]=(-1)**i*(2*i+1)`. For window W, dt=1/W and bilinear discretization
  `Ad=solve(I-dt*A/2,I+dt*A/2)`, `Bd=solve(I-dt*A/2,dt*B)`.
- Accumulate zero-initialized states `c_t=Ad@c_(t-1)+Bd*u_t`. Transform their
  time axis with rFFT; mix N input/output coefficients at each retained bin;
  zero other bins, then irFFT with explicit original trajectory length.
- Each scale s uses the last `min(seq_len,s*pred_len)` observed steps, with
  dt=`1/(s*pred_len)`. Select filtered state at `min(pred_len-1,T-1)` (the
  official source's selection, not always the last state). Reconstruct on the
  final pred_len rows of `P_n(1-2*j/W)`, j=0..W-1. Fuse branches by Linear(S,1).

## Public contract / 公共接口

- `FiLMConfig(ForecastingConfig)` and `FiLM(ForecastingModel)`.
- Extra settings: `memory_order=64`, `modes=16`, `scales=(1,2,4)`,
  `use_norm=True`, `eps=1e-5`. Accept a list for scales after JSON roundtrip.
- Positive integer dimensions/order/modes; seq_len>=2; label_len in [0,seq_len];
  c_out==enc_in; dec_in positive but unused. Scales: nonempty unique positive
  integers. Reject bool-as-int. eps must be finite positive and representable
  as a nonzero finite float32. `use_norm` is a bool.
- `forward(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, ...masks)` returns
  `(B,pred_len,enc_in)`. Calendar markers, decoder values, masks and common
  attention/embedding settings are accepted but unused, like FITS. This model
  does not claim masked-input support; full observed windows must be finite.
- Observed-only per-channel population mean/variance normalization without
  affine parameters; mean/scale detached like the official normalization.
  No persistent sample state and no future targets used.
- Input/model must share float32 or float64 dtype and device; nonempty batch,
  configured shape, finite input. Disable lower-precision autocast internally.
- Finite observations that overflow normalization statistics or normalized
  history raise a numerical-range ValueError before use, not a NaN forecast.
- Store paired real/imaginary FEL parameters and persistent real LPU/basis
  buffers. Config, state_dict and public save/load must preserve float64.

## Explicit differences / 明确差异

Configurable order defaults to 64 rather than official 256. Only contiguous
low modes and dense coefficient mixing; no random modes, low-rank option,
channel compression, learned normalization affine, external training recipe,
tuple return, or official checkpoint adapter. Shared retained count is
`min(modes,max(1,min(seq_len,pred_len)//2))`; DC remains for horizon 1, where
the reference otherwise retains zero bins. No SciPy or TVM dependency.

Dense LPU cost per branch is O(B*C*T*N^2); FEL adds O(B*C*N*T*log(T)) plus
O(B*C*M*N^2). Intermediate coefficient histories require O(B*C*N*T) memory.
This is not an efficient parallel scan or an unconditional linear-cost claim.

## Guardrails and acceptance / 边界及验收

- Python >=3.10, PyTorch >=2.0; no new dependency, workflow, version or Stable change.
- Preserve 161 grandfathered registry entries and all existing 916 protected API records.
- Registry key `FiLM`, integration `paper_aligned_portable_subset`; maturity and
  benchmark evidence remain `unassessed`. Observation admission does not block
  a tested implementation and does not mean core or Stable admission.
- Hand-derived discretization, recurrence and Legendre-grid fixtures; phase
  mixing and cutoff oracles; gradcheck and trainable gradients; short/odd/even
  windows, H>L and H=1; branch state selection; channel independence; future
  poisoning; normalization; invalid config/input; AMP and float64 roundtrips.
- Full regression/coverage, catalog/admission/API audits, wheel smoke, independent
  review and all remote CI must pass before the authorized merge.
- Existing report history remains unchanged; append R3 progress: 37/96 mapped,
  59 remaining after integration, without certifying previous implementations.
