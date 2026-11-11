# Diagnostics R9: Knowledge-Data PINN for decreasing RUL

Date: 2026-11-11

Status: paper-aligned M-PINN subset implemented; benchmark evidence remains **unassessed**.

## Literature basis

Xiaohui Fang, Qinghua Song, Xiaojuan Wang, Zhenyang Li, Haifeng Ma and Zhanqiang Liu, “An intelligent tool wear monitoring model based on knowledge-data-driven physical-informed neural network for digital twin milling”, *Mechanical Systems and Signal Processing* 232 (2025), 112736. DOI: 10.1016/j.ymssp.2025.112736.

The IMA paper describes two coupled networks:

- **Psi** maps sensor features and time to a degradation-state trajectory.
- **Gamma** learns the unknown degradation-rate mechanism from features, time, state and automatic-differentiation derivatives.

The governing-rate residual is combined with supervised data loss and a monotonicity penalty. The paper evaluates MLP-based M-PINN and BiLSTM-based L-PINN variants on HMoTP, PHM 2010 and NASA Ames milling experiments.

## Repository adaptation

The repository implements the MLP-based M-PINN subset as **KnowledgeDataPINNRUL**. The source paper predicts increasing tool wear; the RUL adapter uses the sign-reversed remaining-life form:

\[
\frac{\partial RUL}{\partial t}
+
\Gamma\left(x,t,RUL,\frac{\partial RUL}{\partial x},
\frac{\partial RUL}{\partial t}\right)=0,\qquad \Gamma>0.
\]

The positive rate is enforced by a Softplus output in Gamma. The governing equation is a differentiable residual penalty, not an exact closed-form physical model. The sequence monotonicity term separately penalizes RUL increases within a window.

The total training objective is:

\[
\mathcal{L}
=
\mathcal{L}_{data}
+\lambda_p\mathcal{L}_{physics}
+\lambda_m\mathcal{L}_{monotonicity}.
\]

Default weights are physics_weight=1.0 and monotonicity_weight=0.5, matching the paper's reported settings. The paper's M-PINN uses a tanh MLP; this implementation uses the same decreasing-width MLP pattern with Softplus output constraints for nonnegative RUL. The L-PINN/BiLSTM variant is not included in this round.

## Public API

~~~python
import numpy as np
from torch_like.diagnostics import SignalBatch, create_rul_regressor

# features: (samples, window_length, selected_sensor_features)
signals = SignalBatch(
    features,
    sample_rate_hz=1.0,
    channel_names=feature_names,
    units=feature_units,
)

# One strictly increasing time coordinate per step; normalized [0, 1].
time_marks = np.broadcast_to(
    np.linspace(0.0, 1.0, features.shape[1], dtype=np.float32)[None, :, None],
    (features.shape[0], features.shape[1], 1),
).copy()

model = create_rul_regressor(
    "KnowledgeDataPINNRUL",
    unit="cycles",
    enc_in=features.shape[2],
    seq_len=features.shape[1],
    hidden_dim=32,
    hidden_layers=3,
    physics_weight=1.0,
    monotonicity_weight=0.5,
)

model.fit(
    signals,
    final_window_rul,
    x_mark=time_marks,
    epochs=50,
    batch_size=32,
    learning_rate=5e-4,
)
forecast = model.predict_with_unit(signals, x_mark=time_marks)
~~~

features must be finite and scaled using training-only statistics. final_window_rul contains the nonnegative RUL label at the final time step of each window. The time coordinates must be strictly increasing; the model does not infer calendar/time semantics from arbitrary marker columns.

## Model internals

1. **State network Psi**: pointwise tanh MLP receives the sensor-feature vector and time coordinate and predicts a nonnegative RUL trajectory.
2. **Automatic differentiation**: computes partial derivatives with respect to each input feature and time.
3. **Rate network Gamma**: receives (x, t, RUL, dRUL/dx, dRUL/dt) and predicts a strictly positive rate through Softplus.
4. **Physics loss**: minimizes the squared residual of the sign-adapted RUL governing-rate equation.
5. **Monotonicity loss**: penalizes any adjacent time step where predicted RUL increases.
6. **RUL adapter**: returns the final-step RUL prediction through the existing RULRegressor interface. training_loss is used by the adapter so both physics and monotonicity terms participate in optimization.

loss_components(...) exposes the scalar data, physics, monotonicity and total terms for diagnostics. physics_residual(...) returns the residual at every time step, and predict_trajectory(...) returns the full within-window RUL trajectory.

## Public registration and distribution scope

The full repository registers KnowledgeDataPINNRUL in the rul model domain and exports it through torch_like.models. It can be constructed through create_model("KnowledgeDataPINNRUL", domain="rul", ...) or create_rul_regressor("KnowledgeDataPINNRUL", unit=..., ...).

The separate torch-like-algorithms Stable Wheel remains on its explicit allowlist and does not include knowledge_data_pinn.py in this release. Its list_rul_models() therefore does not advertise this model. The full-framework catalog and the slim-wheel algorithm inventory intentionally differ.

## Deliberate boundaries

- This is a paper-aligned M-PINN subset and a sign-adapted RUL formulation, not a line-by-line reproduction of the tool-wear implementation.
- The original paper's input feature selection (Kendall/Spearman/MIC), Hampel filtering, Min-Max scaling and milling-specific sensor processing are not silently applied. Callers own preprocessing and leakage-free train/test splits.
- The original paper's PDE rate is a learned mechanism Gamma rather than a known closed-form physical equation. A low residual indicates consistency with the learned rate network; it is not independent proof that the rate law is physically correct.
- The sign-reversed RUL equation is appropriate only when remaining life decreases along the supplied within-window time coordinate. It must not be used unchanged for increasing wear/health-index targets.
- Derivative estimates depend on feature scaling and time normalization. No guarantee is made under irregular time sampling, discontinuous event sequences, missing values or strong sensor noise.
- The M-PINN variant is pointwise over time steps; temporal coupling enters through the learned time coordinate, physics residual and monotonicity loss. The paper's BiLSTM L-PINN architecture is deferred.
- No HMoTP, PHM 2010 or NASA Ames data loader, author checkpoint, milling data or published score is bundled. No spacecraft telemetry benchmark has been run.
- The model is excluded from the separate slim Stable Wheel for now; the full framework RUL catalog contains it.

## Verification

tests/test_knowledge_data_pinn_rul.py covers:

- finite nonnegative trajectory and final-step outputs;
- physics residual and all loss-component contracts;
- gradient propagation through both Psi and Gamma networks;
- explicit monotonic time-marker validation;
- MSE/MAE/Huber data-loss paths;
- public model registry and RUL factory construction;
- end-to-end RULRegressor.fit and predict smoke tests.

These are synthetic contract and gradient tests only. Benchmark maturity and evidence remain unassessed.
