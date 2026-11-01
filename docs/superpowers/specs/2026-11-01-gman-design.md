# GMAN R5 design / 图多重注意力设计

## Goal and admission

Continue the authorized Workbuddy intake with native `GMAN` / `GMANConfig`.
Source report: `Workbuddy-TorchLike/report-2026-06-13.md`.
Policy 1.0: R23/P13/U12/E5/I8/A8/X7 =76, evidence A, risk6, final70:
**observation**, not core. Count TESTAM's new compute/case-specific experiments
and CASAformer's speed-regime experiments, not TESTAM's inherited main table.
These are paper-level independent experiments, not two inspected baseline
pipelines. With evidence B, unchanged dimensions/risk give62.4/rejected.
Runtime maturity and dataset benchmark evidence remain `unassessed`.

## Reference and exact retained mechanism

Zheng, Fan, Wang and Qi, AAAI2020, https://arxiv.org/abs/1911.08415v2.
Apache-2.0 reference `zhengchuanpan/GMAN` is pinned to
`45ed232f4ee6a7ee9dd248d07f4952b697ecf320`; inspect METR/PeMS model.py,
tf_utils.py, utils.py and train.py. Independently implement the mathematics;
no TensorFlow, node2vec, third-party port, data or pretrained assets are bundled.

1. Traffic `(B,P,N)` becomes `(B,P,N,1)` then pointwise `1→D→D`.
2. Supplied fixed SE `(N,D)` is projected `D→D→D`. Integer calendar
   `(B,P+Q,2)` is **weekday then slot**; concatenate one-hot depths7 and T,
   then `(7+T)→D→D`. Add both projections by broadcasting over nodes/times.
3. Each encoder/decoder ST block computes dense node attention and per-node
   temporal attention. Each Q/K/V is a distinct `2D→D` activated projection
   of `concat(X,STE)`. Contiguous channel chunks form heads; scale by sqrt(D/K).
   Each attention output is projected `D→D→D`.
4. Fusion gate is sigmoid(Linear_no_bias(HS)+Linear_with_bias(HT)). Mix
   z*HS+(1-z)*HT, project `D→D→D`, add the original X. No LayerNorm or
   post-residual activation. Gate affine maps have no BN.
5. Transform attention uses future STE as Q, history STE as K, encoded history
   alone as V, each activated `D→D`. Attend over history separately per node,
   project `D→D→D`; this produces all Q future states before decoder ST blocks.
6. Output `D→D→1`, squeeze the last channel only. Never read decoder traffic.

All two-layer maps activate only their first affine. An activated affine uses
channel-last population BN, if enabled, then ReLU. BN reduces every nonchannel
axis, epsilon1e-3, affine gamma1/beta0, fixed `bn_decay=.9`. Running moments use
decay*old+(1-decay)*population_moment, without unbiased correction. Singleton
reductions are legal. **Portable adaptation:** initial running variance is1,
not TensorFlow's zero tensor-EMA shadow, so untrained eval is usable; running
mean starts0. No claim of exact TF initial-state/checkpoint/training parity.
The reference training schedule for decay is not reproduced.

## Explicit source-variant choice

Defaults are PeMS-like: `temporal_mask=False`, `dropout=0`, `e_layers=1`.
`e_layers` counts blocks in **both** encoder and decoder, as source L does.
`temporal_mask=True`, `dropout=.1`, `e_layers=5` selects the METR architecture
settings, not its full training recipe. Dropout is before each of the two
output affine layers only. Source finite masking sentinel is replaced by -inf.
BN in training aggregates all times, so a masked full block is not an online
prefix-causal model; mask invariance is asserted only with BN disabled/frozen.

## Public contract

- Standard ForecastingConfig/ForecastingModel dimensions; defaults D64, heads8,
  e_layers1, d_layers1, d_ff64, activation='relu'. `c_out == enc_in`.
- Require explicit `spatial_embedding` of exact shape `(enc_in,d_model)`;
  detach/clone a dense real numeric, finite, non-bool, non-complex buffer.
  Negative/zero values are legal. Reject overflow or nonzero underflow during
  construction-dtype conversion. Sensor order must match traffic feature order.
- `time_slots_per_day=288`, `use_batch_norm=True`, `bn_decay=.9`.
  Used integer dimensions are positive, label_len is in[0,P], D divisible by
  heads. Accept numbers.Integral except bool, canonicalize before arithmetic.
  dropout/decay are finite real numbers in[0,1); bool switches must be bool.
- Input float32/float64 and model dtype/device must match; AMP is disabled in
  this portable path. Require nonempty batch and finite observed traffic.
  Finite checks are not extreme-scale stability guarantees; use scaled data.
- Optional dtype selects float32/float64 (string or torch.dtype); None uses a
  supported process default, without changing global state. Config records the
  current model dtype before reconstructing float64-only SE values. Initial
  random parameters follow ambient initialization then convert, as with .double();
  this is not an upstream initializer-parity claim. GMAN.from_config extracts
  dataclass fields shallowly so non-leaf tensors reach owned-input detachment.
- Mandatory integer marks `(B,P,2)` and `(B,label_len+Q,2)`. Supported integer
  dtypes are uint8/int8/int16/int32/int64. Weekday0..6, slot0..T-1. Validate
  history and used future suffix before any BN mutation. Decoder prefix values
  are unused; no continuity, timezone or calendar inference is performed.
- `make_forecast_batch` refuses missing marks before common zero defaults can
  fabricate an invalid calendar. `predict` consumes the standard batch.
- All decoder traffic is ignored, including NaN or malformed unused objects.
  External masks are rejected; use temporal_mask. output_attention=True and
  activation other than 'relu' are rejected. d_layers/d_ff/embed/freq remain
  compatibility-only fields and do not alter the GMAN operator.
- `config()` emits owned current SE lists and canonical used numeric scalars;
  constructor/config mutation cannot change the model. Public save/load must
  preserve SE, all parameters and BN running buffers exactly in float32/64.

## Cost, acceptance and exclusions

For each ST block at length S, dense spatial+temporal attention arithmetic is
O(B*D*(S*N^2+N*S^2)), with attention storage O(B*K*(S*N^2+N*S^2)); projections
add O(B*S*N*D^2). The transform bridge adds O(B*N*P*Q*D) and corresponding
O(B*K*N*P*Q) logits, plus projections. No sparse or graph-free pipeline claim:
SE is supplied, commonly generated offline from a graph.

Require independent loop/dense output and gradient oracles for every attention,
fusion, embedding and complete forecast; population-BN state oracle; singleton,
noncontiguous and fixed-width integer boundaries; node-order equivariance;
future-traffic isolation; real train/predict; exact public save/load; additive
lazy catalog/API integration; full tests/coverage>=83%; full and Stable wheels;
one fresh branch review and all three PR/postmerge workflows.

Exclude offline embedding generation, automatically synthesized calendars,
per-node multichannel traffic, arbitrary masks, graph assets, dataset training
pipelines, raw-data scaling, official checkpoints, benchmark/speed claims and
Stable expansion. Preserve existing924 API records, grandfathered161 baseline,
version/dependencies/workflows, unrelated PR30 and existing user files.
