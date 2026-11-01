# Workbuddy R7: native DeepState design

## Intent and authority

Continue the user's authorized feature-branch / test / review / merge workflow
for worthwhile remaining Workbuddy models. R6 is an immutable historical triage,
not an implementation approval. R7 applies the unchanged admission policy before
adding a model. The user delegated value decisions and repeatedly requested
continued execution; the choices below are engineering judgments, not newly
elicited user requirements.

The current main is `43506e4c4621eceedf7310372791984f88425e7e`; its tree
`3c8b2aa3d3c8210b5c7545271107bf8af61cdfb0` equals local base `fe566826`.
Keep unrelated draft PR30 and all user files intact.

## Admission decision

DeepState: value dimensions R/P/U/E/I/A/X = 24/10/12/7/8/8/7, total76;
literature evidence A; risk8; final68; **observation**, not core. The actual
VAEneu baseline reruns and TAGI-LSTM/SSM's explicit report of new Appendix L
GluonTS runs are two independent author/institution sources. TAGI main Table2
reprints prior numbers and is excluded; Appendix L tables were not retrieved.
This is source-structure evidence, not two locally reproduced pipelines.
Risks:3 legacy-port/numerical-state burden,3 incomplete external protocol and
supplement audit,2 unassessed local parity. Weak external accuracy is already
reflected in P10. Do not adjust scores to compensate for an evidence downgrade.

Paloma remains no-code:75*C.75-6=50.25. SEER remains no-code even granting the
scoped external clean-forecasting logbook:75*B.90-7=60.5. Neither clears65.
Their detailed records must separate evidence limitations from scientific value.

## Approaches considered

1. Wrap the MXNet/GluonTS estimator: strongest pipeline reuse, but violates the
   native-PyTorch/no-new-runtime-dependency boundary. Not selected.
2. Add another Gaussian RNN head: small implementation, but lacks the defining
   analytic state posterior and correlated paths. Not selected.
3. Implement the bounded structural Gaussian state-space core in PyTorch with
   explicit inputs: selected by the maintainer. This is more numerical work,
   but contributes the missing capability and fits current native interfaces.

## Mathematical contract

Pinned reference: awslabs/gluonts
`d60556ca71167a3ec1ea807b491dd982d1de2112`, DeepState network/ISSM and LDS files,
Apache-2.0 plus NOTICE. Preserve attribution for the adapted mathematics/code.

Use `z_t = a_t^T l_(t-1) + b_t + sigma_t epsilon_t` followed by
`l_t = F l_(t-1) + g_t nu_t`. Each time has one scalar state innovation:
`g_t = gamma_t a_t`, so `Q_t = g_t g_t^T`, including cross-component terms.
Level coefficients are1; level/trend emission and innovation are[1,1] and
transition is[[1,1],[0,1]]. Seasonal blocks use caller-supplied one-hot indices
and identity transitions. Composite transitions are block diagonal.

The shared residual-stack LSTM consumes only known real/static covariates, not
targets or their scale. The first context output predicts prior mean/diagonal
covariance. Sigmoid-affine heads predict observation standard deviation,
innovation multiplier and prior covariance; an affine head predicts residual.
Default bounds:noise[1e-6,1],innovation[1e-6,.01],prior covariance[1e-6,1].
Upper recurrent layers add their input to their LSTM output. Only dropout0 is
supported: do not replace reference state-zoneout with inter-layer dropout.

Filter in emit -> optional observation update -> transition order. Use Joseph
covariance updates and symmetrize after propagation. A missing observation
contributes no likelihood/update but still transitions. Returned context state
already precedes the first future emission: no extra boundary transition.

Mean absolute scaling uses observed context only and retains input gradients.
Zero-only/all-missing rows use fixed scale1, not a batch-global fallback; positive
scales are floored at1e-10. This deliberate adaptation makes rows independent.
Mask with `where` before arithmetic so masked NaN/Inf never contaminates results.

## Public API

Add `DeepState`, `DeepStateConfig`, `DeepStateBatch` to all three public model
namespaces. `DeepState` subclasses `ProbabilisticForecastingModel`.

Require `enc_in == dec_in == c_out == 1`; batches are independent univariate
series, not jointly multivariate outputs. Positive seq_len/pred_len/d_model/
e_layers/num_samples; integral0<=label_len<=seq_len. Defaults d_model40,e_layers2,
dropout0,distribution='gaussian'. n_heads/d_layers/d_ff are documented unused
compatibility fields, with the inherited defaults8/1/128.

Additional config:time_feature_dim=4 (nonnegative integer),seasonal_periods=(),
add_trend=True,cardinalities=(),embedding_dimensions=(),scaling=True,
minimum_scale=1e-10,noise_std_bounds=(1e-6,1),innovation_bounds=(1e-6,.01),
prior_cov_bounds=(1e-6,1),dtype=None. Lists/tuples are owned and JSON-canonical;
cardinality/embedding lists have equal length and strictly positive integral
entries. Periods are integers>=2. Bounds are finite,positive,ordered and
representable in construction dtype (including squared observation noise).
dtype is float32/float64 or their names; None uses a supported process default.
Config records current dtype, including after `.double()`.

`DeepStateBatch` extends the existing probabilistic batch, without changing it:

| Field | Contract |
|---|---|
| x_enc | float[B,seq_len,1] history |
| x_mark_enc | float[B,seq_len,D] known real covariates |
| x_mark_dec | float[B,label_len+pred_len,D]; use last pred_len only |
| x_dec | optional float[B,label_len+pred_len,1], ignored values |
| observed_enc | optional bool[B,seq_len,1]; omitted means all observed |
| observed_dec | optional bool[B,pred_len,1] training-label mask |
| seasonal_indices_enc | int[B,seq_len,K] within each configured period |
| seasonal_indices_dec | int[B,label_len+pred_len,K]; active future tail only |
| feat_static_cat | int[B,len(cardinalities)] within configured ranges |
| y_true | float[B,pred_len,1], training only |

Require real features when D>0 and calendars when K>0. No inferred time zone,
cadence,phase,continuity or synthetic category IDs. D=0 can omit real features;
if there are also no categorical embeddings, use one explicit constant input1
for the RNN. Decoder prefixes do not add observations or recurrent steps.
Reject nonempty split target/exogenous fields and attention masks; do not silently
reinterpret them. A dedicated batch helper preserves supplied fields without
fabricating calendars. Override predict to forward all subclass fields.

`forward` retains the usual first4 forecasting arguments and mask arguments,
with keyword-only observed_enc,seasonal_indices_enc,seasonal_indices_dec,
feat_static_cat. `predict` returns mean[B,H,1]. `predict_distribution` returns
mean,scale (marginal standard deviations),distribution='gaussian'; marginals do
not assert temporal independence. `sample(batch,num_samples=None)` returns
[samples,B,H,1] using a shared latent trajectory, not independent marginal draws.

`fit_step` is observed-count-mean conditional future NLL in original units:
filter context, then sequentially assimilate observed y_true. Include log(scale)
Jacobian. Require at least one observed future label globally; all-missing rows
contribute zero. This is an explicit objective adaptation to the repository API.
`context_nll(batch)` supplies reference-style context likelihood: per-nonempty-row
mean,then mean across nonempty rows; it requires no future inputs/labels.

## Numerical and persistence boundaries

Dense float32/float64 tensors only,matching model dtype/device; reject empty
batches,unmasked nonfinite values and unsupported storage/layouts. Disable AMP
inside numerical operations. Accumulate filtering, marginals and sampling in
float64, retaining that posterior across the history/future boundary; cast public
outputs to model dtype. A reproduced float32 cancellation at step8 of a low-noise
seasonal example justified this adaptation; it increases compute/storage cost.
Known active features must be finite and calendar/
category tensors integral (not bool). Bound any sampling Cholesky retry jitter
to100*eps*covariance-scale and apply only to failing batch rows; fail explicitly
if still invalid. This is not an arbitrary-indefinite-covariance repair.

Fixed transition is a persistent buffer. Filtered posteriors are recomputed per
call and are not model/session state. Existing model serialization saves learned
parameters/config/buffers exactly; no online continuation/checkpoint API is added.

## Integration and non-goals

Add only a model-specific DeepState template in contracts.inspect, pointing to
DeepStateBatch; shared runtime dispatch and other batch schemas stay unchanged.
Dataclass training batches retain extras through DefaultBatchAdapter and device
movement. Generic(x,y) pairs work only for the feature-free configuration;
calendar/static-feature training needs dataclasses or an explicit callable adapter.

Register integration_level='paper_aligned_portable_subset'; maturity and benchmark
evidence remain unassessed. Do not claim source checkpoint/training-pipeline parity,
dataset accuracy, calibrated intervals, speedups or production readiness. No
pretrained weights,datasets,MXNet,GluonTS,scipy or new dependencies are bundled.
Do not change Stable packaging,version,workflows,policy,grandfathered161 baseline
or PR30. Do not add Paloma/SEER implementations.

## Acceptance

Require literal state-space/Kalman oracles,independent dense-joint-Gaussian NLL,
mask and scaling invariance,finite/checked gradients,empirical cross-time sample
covariance,recurrent continuation,config/state/public-save-load roundtrips,
runtime/dataclass preservation,lazy exports and boundary validation.
Run full existing tests/coverage,lint/format,policy/catalog/API/lazy validators,
independent whole-branch review and all applicable actual PR/main CI before the
already-authorized merge. Record Stable workflow applicability honestly.
