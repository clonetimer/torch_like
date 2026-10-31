# Non-stationary Transformer Portable Forecasting Design / 去平稳 Transformer 可移植预测设计

## Intent / 目的

Implement the canonical **Non-stationary Transformers** mechanism from the
2026-06-25 Workbuddy report as one native PyTorch, randomly initialized
encoder–decoder point forecaster. Fill the currently absent learned
de-stationary attention gap without expanding existing attention APIs.

Primary source: Liu et al., NeurIPS 2022,
https://papers.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html .
Official code: https://github.com/thuml/Nonstationary_Transformers at
`c4ec40675d11d50b3d9923657f408d0db6f90f56`, MIT.

## Approaches / 路线

1. **Standalone model-local encoder–decoder (chosen).** Private attention,
   factor projectors, embeddings and Transformer blocks consume the existing
   `ForecastingModel` contract. This preserves the original mechanism and
   isolates portability differences from the existing `Informer` and shared
   attention classes; it costs some extra local code.
2. Generalized de-stationary adapter for every registered Transformer. That
   would need to change shared attention signatures, mask polarity, distilling
   lengths and decoder semantics across unrelated models. Deferred.
3. A normalization wrapper around ordinary attention. That cannot express the
   learned, per-sample `tau` and per-key `delta` and is not this algorithm.

The user asked to continue with the recommended order and authorized GitHub
execution, without per-batch design questions. The chosen portable implementation
is our engineering decision, not a claim of upstream numerical equivalence.

## Public contract / 公共接口

- Export `NonstationaryTransformer` and `NonstationaryTransformerConfig` under
  `torch_like`, `torch_like.models`, and the forecasting facade, with registry
  key `NonstationaryTransformer`. Exact class capitalization is intentional.
- Constructor takes the standard `ForecastingConfig` parameters plus
  `factor_hidden: int = 128` and `eps: float = 1e-5`.
- `enc_in == dec_in == c_out > 0`; `seq_len >= 2`, `0 <= label_len <= seq_len`,
  `pred_len > 0`; `d_model % n_heads == 0`. `e_layers,d_layers >= 1`.
- Point forecast is a single tensor `(batch, pred_len, c_out)`; `predict`,
  `make_forecast_batch`, `.config()`, `from_config()`, `state_dict` and public
  save/load must work. Float32 and float64 CPU are tested; CUDA is unassessed.
- The input is `(B,seq_len,enc_in)` with finite values. Value decoder `x_dec`
  is intentionally ignored; its history prefix is assembled from the final
  normalized `label_len` observations, its future from zeros. This prevents
  target leakage even when the caller provides future targets.
- Optional time markers have 4 features (same as the existing batch helper);
  encoder markers `(B,seq_len,4)` and decoder markers
  `(B,label_len+pred_len,4)` may influence predictions. Missing markers mean
  zero markers. Both future markers and only those known at inference are
  allowed; future target values are never used.
- External boolean attention masks use **True = allowed**. Encoder masks have
  `(seq_len,seq_len)`, decoder self masks `(D,D)`, cross masks `(D,seq_len)`;
  optional batch-aware dimensions are allowed. Decoder self attention always
  intersects its mask with the causal lower triangle. A fully blocked row
  raises `ValueError`; no silent NaNs. Incompatible masks are rejected.
- `output_attention` is accepted as a normal forecasting config parameter,
  but the public `forward/predict` result remains only a tensor. No tuple ABI.

## Mathematical contract / 数学合同

- Per sample/channel, compute `mean = x.mean(time)` and
  `scale = sqrt(x.var(time,unbiased=False) + eps)`. Normalize only the
  observed input; final forecast is `scale * prediction + mean`.
- Two learned projectors each apply `Conv1d(seq_len,1,kernel_size=3,
  padding=1,padding_mode='circular')` to raw `x` across the **feature** axis,
  concatenate projected features with the detached observed scale/mean,
  then feed a two-hidden-layer MLP. The scale projector outputs positive
  `tau = exp(log_tau)`; the bias projector outputs a vector `delta` with
  exactly `seq_len` encoder-key positions. All three raw/statistic inputs to
  the projectors are detached, but projector weights are learned.
- For projected Q/K/V and head dimension `d`:
  `softmax((tau * (Q @ K^T) + delta[key]) / sqrt(d) + mask) @ V`.
  `tau` is `(B,)`; `delta` is `(B,seq_len)`, added along **keys**, not queries.
  Encoder self and decoder cross attention use both factors. Causal decoder
  self attention uses only `tau` and never `delta`.
- Use sinusoidal positional embedding plus learned value/time projections,
  with local residual/normalization/FFN blocks. These choices make this an
  explicitly documented **paper-aligned portable subset**, not a bytewise
  copy of the historical PyTorch 1.9 upstream training program.
- `log_tau` is bounded to `[-15,15]` before exponentiation to avoid a
  needless nonfinite path; this is a stated numerical-portability deviation.
  No attention-sparsity or linear-complexity claim: temporal attention is
  dense/quadratic in sequence length.

## Governance, tests and non-goals / 治理、验收与非目标

- Follow the six hard gates then numeric R/P/U/E/I/A/X score under policy
  1.0 before registering; independent evaluations are TFB (PVLDB 2024,
  *univariate* Table 6 only) and DERITS (IJCAI 2024, Table 1). No author-paper
  scores are treated as local benchmarks or universal superiority.
- Tests must compare de-stationary attention with a small independent
  float64 score/softmax oracle, ensure key-wise delta placement, masking and
  causal isolation, factor/projector gradient and detachment, normalization,
  forecast forward/backward, save/load, lazy public exports and governance.
- No pretrained weight, SciPy, C++/CUDA/TVM dependency, existing model
  registry baseline edit, Stable promotion, dataset benchmark claim, GPU
  parity claim, version bump, or modification of unrelated PRs.

