# Diagnostics R8: AGIL graph multi-task fault diagnosis and RUL

Date: 2026-11-11

Status: research implementation candidate; benchmark evidence remains **unassessed**.

## Literature basis

J. Qi et al., “Attention-guided graph isomorphism learning: A multi-task framework for fault diagnosis and remaining useful life prediction”, Reliability Engineering & System Safety 263 (2025), 111209.

The paper combines a three-layer GIN/EdgePool shared encoder, task-specific graph self-attention, and separate fault-classification/RUL heads. Its reported experimental data consist of 15 rolling-bearing degradation runs under three operating conditions. The local implementation does not claim to reproduce the paper's reported metrics or to establish broad cross-machine generalization.

## Implemented API

The implementation is available through torch_like.diagnostics:

~~~python
from torch_like.diagnostics import AGILFeatureGraphBuilder, AGILMultiTask

# Fit normalization and the Pearson graph on training windows only.
builder = AGILFeatureGraphBuilder(correlation_threshold=0.8).fit(train_signals)
train_graph = builder.transform(train_signals)
test_graph = builder.transform(test_signals)

x_train, adjacency = train_graph.to_torch()
x_test, _ = test_graph.to_torch()

model = AGILMultiTask(
    num_classes=4,
    hidden_dim=64,
    lambda_fault=0.2,
    lambda_rul=0.8,
)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4)

model.train()
for _ in range(100):
    optimizer.zero_grad()
    loss = model.training_loss(x_train, adjacency, fault_labels, rul_targets)
    loss.backward()
    optimizer.step()

model.eval()
with torch.inference_mode():
    output = model(x_test, adjacency)
    fault_probabilities = output.fault_logits.softmax(dim=-1)
    rul_predictions = output.rul_values
~~~

### Feature graph builder

- Input: a SignalBatch containing one vibration channel per sample/window.
- Feature extraction: the 16 paper indicators F1–F16, including mean, RMS, variance, square-root mean absolute value, peak/peak-to-peak, standardized third/fourth moments, crest/shape/impulse/margin factors and spectral moments.
- Training-only fit: standardization moments and Pearson correlation graph are estimated from the training partition only.
- Adjacency: an undirected edge is retained when the training-set Pearson correlation exceeds the configured threshold (default 0.8); self-loops are always present.
- Test transform: uses frozen training normalization and adjacency; it never refits on held-out data.

The paper describes a per-sample feature graph. The local implementation makes the correlation graph explicit and estimates it from training windows because Pearson correlation requires multiple observations for each feature pair. This is a deliberate, leakage-controlled adaptation rather than an assertion of exact reproduction of the original graph-construction pipeline.

### Multi-task model

1. Three shared GIN layers with learnable epsilon, MLP aggregation, masked batch normalization and ReLU.
2. Learnable EdgePool scores with row-wise softmax, greedy disjoint edge contraction and score-weighted node merging.
3. Global mean readout after each graph layer; all three layer representations are retained.
4. Separate scaled dot-product self-attention modules for fault diagnosis and RUL.
5. Separate classification and nonnegative regression heads.
6. Weighted joint objective: lambda_fault × cross-entropy + lambda_rul × MSE.

The multi-task model returns both outputs in one forward pass. It is a native nn.Module with an explicit training_loss method, not a single-task classifier or RULRegressor adapter.

## Deliberate boundaries

- The feature graph builder currently supports one vibration channel per SignalBatch. Multi-axis signals should be handled by an explicit channel-fusion strategy rather than silently flattened.
- Pearson adjacency is estimated across training windows, not independently for each held-out sample.
- The original paper's raw bearing dataset, RUL label generation, piecewise-linear label configuration, and exact train/test splits are not bundled.
- The paper's EdgePool is implemented as a deterministic greedy disjoint edge contraction. Matching is discrete; gradients flow through selected edge scores and merged node features.
- No automatic class-label schema, data split, normalization fit on test data, or benchmark report is created by the model.
- The implementation is intended for controlled experiments. It is not automatically inserted into the global model catalog or Stable wheel allowlist.

## Verification

R8 contract tests cover all 16 features, zero-signal numerical stability, training-only graph fitting, immutable graph batches, EdgePool contraction, both task outputs, joint-loss gradient flow, and invalid graph/label/target inputs.

Synthetic contract tests establish API and numerical behavior only. They do not establish the paper's reported bearing benchmark scores, cross-machine robustness, or satellite applicability. Maturity and benchmark evidence remain unassessed.
