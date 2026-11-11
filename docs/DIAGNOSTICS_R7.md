# Diagnostics R7: HSNN-IB hierarchical subgraph RUL model

Date: 2026-11-11

Status: implementation candidate; maturity and benchmark evidence remain **unassessed**.

## Literature basis

Linjie Zheng, Junyu Qi, Dengyu Xiao and Yi Qin, “Hierarchical subgraph neural network with information bottleneck for machine remaining useful life prediction”, Information Fusion 127 (2026), 103953, DOI: 10.1016/j.inffus.2025.103953.

The paper evaluates the method on NASA C-MAPSS FD001–FD004 and a real wind-turbine SCADA dataset. It motivates the architecture by identifying redundant and task-irrelevant information propagation in whole-graph multi-sensor RUL models.

## What is implemented

The portable PyTorch model is exposed through the existing native RUL adapter:

~~~python
from torch_like.diagnostics import create_rul_regressor

rul = create_rul_regressor(
    "HSNNIBRUL",
    unit="cycles",
    enc_in=14,
    seq_len=60,
    hidden_dim=60,
    num_layers=2,
    num_subgraphs=3,
    max_nodes_per_subgraph=5,
    initial_edge_threshold=0.5,
    centrality_alpha=0.5,
    ib_weight=1e-5,
)

rul.fit(
    healthy_and_degradation_windows,
    rul_targets,
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
)
forecast = rul.predict_with_unit(test_windows)
~~~

The model includes the following mechanisms:

1. **Sensor sequence encoding**: one LSTM encodes each sensor's window into a node representation.
2. **Initial graph**: cosine similarity between sensor-window sequences creates a thresholded adjacency, with self-loops retained.
3. **HMGPA-inspired partitioning**: learned edge similarities reweight the previous graph while preserving each node's total incoming connection strength. Centrality combines mean connection strength with negative entropy of the softmax-normalized edge distribution. High-centrality nodes become subgraph centers, and each center selects up to M-1 unused neighbors.
4. **HGSFU-inspired update**: a two-layer tanh network updates features inside each subgraph; learned cross-subgraph similarities and two gates regulate information exchanged between groups.
5. **MFIBM-inspired fusion**: mean node features from each subgraph are accumulated across graph layers, then fused by a Gaussian variational bottleneck with a standard-normal KL penalty.
6. **RUL regression**: a nonnegative scalar head predicts the caller-declared RUL target unit.

The existing RULRegressor now supports an optional model-specific training_loss hook. HSNNIBRUL uses this hook to optimize MSE/MAE/Huber plus the information-bottleneck KL term. Existing RUL models continue using their original loss path.

## Public registration and distribution scope

The full repository registers `HSNNIBRUL` in the `rul` model domain and exports it through `torch_like.models`. It can be constructed through either the model registry (`create_model("HSNNIBRUL", domain="rul", ...)`) or the diagnostic adapter (`create_rul_regressor("HSNNIBRUL", unit=..., ...)`).

The separate `torch-like-algorithms` Stable Wheel remains on its existing explicit allowlist and does not include `hsnn_ib.py` in this release. Its `list_rul_models()` therefore does not advertise HSNN-IB. The full-framework catalog count and the slim-wheel algorithm inventory are intentionally different.

## Deliberate implementation boundaries

This is a paper-aligned implementation of the main architectural ideas, not a line-by-line reproduction of the authors' code or a claim of reproducing their reported scores.

- The article's graph-partitioning equations are translated into a deterministic bounded subgraph procedure. Discrete center/neighbor selection is non-differentiable; gradients flow through the selected edge weights and feature updates.
- The learned pairwise similarity and gate networks use compact MLPs rather than claiming an exact recreation of every 1×1-convolution/BatchNorm detail.
- The number of subgraphs and maximum nodes per subgraph are explicit user settings. Nodes outside the selected K×M budget do not enter the subgraph summaries, consistent with the paper's representative-subgraph design. Users should set a sufficient budget for their sensor count.
- The paper uses dataset-specific preprocessing, window construction, RUL clipping and train/test protocols. The model does not silently fit a global scaler or infer labels. The caller remains responsible for preprocessing, leakage-free splits and RUL target construction.
- The implementation does not include the paper's SCADA data, NASA data loaders, scoring-function evaluation, or published benchmark claims.
- The variational bottleneck samples during training and uses its posterior mean for deterministic inference.

## Initial configurations from the paper

| Setting | C-MAPSS | Wind-turbine SCADA |
|---|---:|---:|
| Window length | 60 | 60 |
| Hidden size | 60 | 60 |
| Subgraph count K | 3 | 4 |
| Maximum nodes M | 5 | 7 |
| Initial edge threshold | 0.5 | 0.5 |
| Centrality balance α | 0.5 | 0.5 |
| Learning rate | 0.001 | 0.001 |
| IB weight λ | Dataset-specific, 1e-5–2e-5 | 5e-5 |

These values are starting points, not universally optimal defaults. In particular, the model constructor defaults to the C-MAPSS sensor-count/window configuration and must be adjusted for other datasets.

## Verification

R7 contract tests cover:

- HMGPA-style partition cardinality, disjointness and isolated-node behavior.
- Finite, deterministic evaluation and nonnegative RUL output.
- Variational bottleneck loss and gradient propagation.
- Native RULRegressor factory, training hook, fitting and prediction.
- Invalid dimensions, sensor graph matrices and nonfinite inputs.

Synthetic tests establish API and numerical behavior only. They do not establish C-MAPSS or SCADA benchmark performance, cross-device generalization, or spacecraft hardware validity. Both HSNNIBRUL and its benchmark evidence remain unassessed.
