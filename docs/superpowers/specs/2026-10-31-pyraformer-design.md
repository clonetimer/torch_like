# Pyraformer R4 design / 金字塔注意力设计

## Goal and admission

Continue the authorized Workbuddy forecasting intake with one random-initialized
`Pyraformer` / `PyraformerConfig`, not a generic Transformer renamed as Pyraformer.
Policy 1.0 score: R23/P11/U12/E7/I8/A8/X7 = 76, evidence A, risk 7,
final 69: **observation**. Implementation does not imply core or benchmark maturity.
Evidence A counts Zeng's source-based long-range evaluation and TimesNet's separate
M4 adapted-backbone experiment. TimesNet's matching long-range results are not
counted twice. If unmodified-source replication were required, evidence B would
give 61.4 (rejected); no score is increased to hide this sensitivity.

## Source contract

Primary: Liu et al., ICLR 2022, https://openreview.net/forum?id=0EXmFzUn5I.
Official Apache-2.0 reference: `ant-research/Pyraformer` at
`84af4dbd93b7b96975b5034f0dde412005260123`. Independently implement the mathematics;
do not vendor its TVM/CUDA/Informer/Longformer code or training pipeline.
The reference uses test-set MSE for checkpoint selection; this procedure is
explicitly excluded, and no clean reproduction or checkpoint parity is claimed.

## Architecture

1. Append a zero raw-value prediction token to observed `(B,L,C)`. Append the
   first future time mark at decoder index `label_len`. Embedding is biased
   circular Conv1d(k3,p1), sinusoidal positions, biased Linear(4,D), then dropout.
2. Bottleneck CSCM: Linear(D,db), three Conv1d(db,db,k=f,stride=f) → BatchNorm1d
   → ELU stages, concatenate coarse states, shared Linear(db,D), prepend original
   fine embedding, LayerNorm(eps=1e-5). Convolutions drop ragged tails.
3. Four graph levels have lengths `[L+1, floor(n0/f0), floor(n1/f1), floor(n2/f2)]`.
   Each node attends to itself and nearby same-level nodes, its parent, and its
   children. Last parents absorb remainder children, unlike convolution coverage.
   Build padded neighbor indices/validity directly: never allocate an S-by-S mask.
4. Q/K/V bias-free projections with `dk=dv=D/heads`, indexed K/V gathers,
   scaled dot product, masked softmax, weight dropout, biased output projection,
   output dropout, residual LayerNorm(eps=1e-6). FFN is biased Linear → GELU →
   dropout → Linear → dropout → residual LayerNorm(eps=1e-6). No extra final norm.
5. Select the last fine node and its three ancestors, fine-to-coarse. A bias-free
   Linear(4D,H*C) outputs horizon-major `(B,H,C)`. No autoregressive decoder.

For fixed factors/window, graph and attention storage are O(S*K), K bounded by
`inner_size + 2*max(window_size)`. Gathered K/V additionally carry batch, heads
and head width. Projections/FFN and the forecast head have their own costs; this
is not a measured speedup, an O(1) path guarantee for arbitrary lengths, or TVM.

## Public boundaries

- Standard forecasting constructor/predict interface; `c_out == enc_in`.
- Defaults D64, heads4, encoder layers2, FFN128, bottleneck16, factors(4,4,4),
  inner_size3, dropout0.05. Config dataclass and constructor defaults agree.
- Exactly three integer factors >=2, positive odd inner_size, D>=2 divisible by
  heads, positive layers/FFN/bottleneck/horizon/channels, `0 <= label_len <= L`,
  and every coarse level nonempty. Default factors require `L >= 63`.
- Float32/float64 input and model dtype/device must match; finite observed values
  and used time markers required. AMP is disabled inside this portable path.
- Optional markers are zeros when absent; supplied shapes `(B,L,4)` and
  `(B,label_len+H,4)`. Only the first future mark is used/checked for finiteness;
  decoder prefix and later future marks, and all decoder values, are ignored.
- External masks are rejected explicitly. `activation` must be `gelu`;
  `output_attention=True` is rejected (no public dense attention output).
  `d_layers`, `embed`, `freq` are compatibility fields, unused.
- Preserve BatchNorm running statistics. Before any state mutation, training
  rejects `B * top_length <= 1`; batch-one inference remains valid.
- Runtime-generated sinusoidal positions support odd D and lengths beyond the
  reference's fixed 5000 positions. These are documented portable extensions.
- Persistent graph indices/validity/last ancestors and position buffer migrate
  with the module and round-trip through public save/load; no external assets.

## Acceptance and exclusions

Literal ragged graph fixtures; independent dense attention and full encoder
forward/parameter-gradient comparisons; gradcheck; padded-slot invariance;
runtime allocation-shape checks; direct CSCM oracle including BN state;
end-to-end dense oracle; future-value isolation; input/config rejection;
float32/64 training and public runtime/save-load; unchanged old API/baselines;
full tests, coverage >=83%, packaging and three GitHub CI workflows.

Exclude attention decoder, CustomEmbedding/series IDs, alternative CSCMs,
single-step probability head, separate dk/dv, pretrained weights, real-data
benchmark claims, Stable allowlist expansion and unrelated PR #30.
Runtime maturity and benchmark evidence stay `unassessed`.
