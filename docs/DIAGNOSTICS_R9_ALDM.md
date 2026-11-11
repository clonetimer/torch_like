# Diagnostics R9: ALDM adaptive latent distribution anomaly detection

Date: 2026-11-11

Status: paper-aligned portable subset; maturity and benchmark evidence remain **unassessed**.

## Literature basis

Yu Liu, Yifan Song, Shaolong Shu, Feng Lin, Jun Wang and Yafeng Guo, “Adaptive Latent Distribution Modeling for Industrial Time Series Anomaly Detection”, IEEE Transactions on Automation Science and Engineering, 23 (2026), 7893–7907. DOI: 10.1109/TASE.2026.3674236.

The paper combines latent contrastive representation learning, adaptive temporal distance, graph-conditioned context and flow-based density scoring. It also introduces an event-wise performance index to reflect anomalies with different durations.

## Reference implementation audit

The official MIT-licensed repository was inspected before implementation. Several issues prevent a direct port:

1. The soft TS2Vec implementation constructs a new SPDLayer inside every metric-learning call. The layer is not a registered parameter of the training model or optimizer.
2. The resulting temporal affinity is detached in the TS2Vec training path before the soft temporal contrastive loss. The adaptive metric therefore receives no gradient from that objective.
3. The SWaT and TEPE loaders fit StandardScaler before the train/validation/test split, allowing future/test distribution statistics into preprocessing.
4. The training script selects checkpoints using test-set ROC-AUC at every epoch rather than a validation-only selection protocol.
5. The evaluation loop does not switch the model to evaluation mode, while the reference TS2Vec path uses stochastic crops, masks and dropout.

These defects are documented as reference-code findings, not attributed to the published paper's mathematical method. This implementation corrects the trainable metric and provides a deterministic inference path; dataset splitting, scaling and checkpoint selection remain caller-owned.

## Public API

~~~python
import torch
from torch_like.models import create_model

detector = create_model(
    "ALDMDetector",
    domain="detection",
    input_dim=51,
    seq_len=60,
    d_model=64,
    metric_dim=32,
    latent_dim=64,
    flow_layers=2,
)

# Train on normal-only windows. Fit preprocessing on the training split only.
batch = detector.make_detection_batch(normal_training_windows)
loss = detector.fit_step(batch)
loss.backward()

# Use model.eval() for deterministic density scoring.
detector.eval()
window_scores = detector.score_batch(
    detector.make_detection_batch(held_out_windows),
    reduction="mean",
)

# Calibrate only on an independent healthy calibration split.
threshold = detector.calibrate_threshold(
    detector.make_detection_batch(healthy_calibration_windows),
    quantile=0.995,
)
~~~

For time/channel-level analysis, score_batch(..., reduction="none") returns shape (samples, time, channels). The caller must align overlapping windows to the original timeline before computing event-level metrics.

## Implemented mechanisms

1. **Sensor graph context**: a shared LSTM encodes each channel; learned query/key attention forms a per-window sensor graph and propagates cross-sensor context.
2. **Latent temporal representation**: a multivariate temporal convolution encoder produces a compact sequence embedding.
3. **Adaptive temporal distance**: a registered lower-triangular factor parameterizes an SPD Mahalanobis metric. The matrix is trace-normalized, and pairwise distances remain in the gradient path of the soft temporal contrastive objective.
4. **Hierarchical contrastive learning**: two masked/noisy views are trained with temporal soft-target contrast and instance contrast at multiple max-pooled time scales.
5. **Conditional density scoring**: each sensor/time latent scalar is transformed by a context-conditioned affine flow; negative log likelihood is the anomaly score.
6. **Event-wise evaluation**: compute_event_wise_metrics implements the reference event recall / point-precision × normal-specificity F1 index with safe handling of empty predictions and normal-only series.

## Distribution scope

ALDMDetector is included in the full torch_like model catalog and public model namespace. The separate slim Stable Wheel remains on its existing explicit allowlist and does not ship ALDMDetector in this release. The full-framework detection catalog and the slim-wheel algorithm inventory are intentionally different.

## Event-wise metric usage

~~~python
from torch_like.diagnostics import compute_event_wise_metrics

# Both arrays must refer to one contiguous, aligned timeline.
metrics = compute_event_wise_metrics(
    ground_truth_labels,
    predicted_labels,
    # Optional half-open intervals preserve adjacent but independent events.
    gt_intervals=((120, 135), (135, 142)),
)
print(metrics.precision, metrics.recall, metrics.f1)
~~~

The reference event precision is point precision multiplied by normal-point specificity; event recall counts a ground-truth event as detected when any predicted point overlaps it. The utility also returns event TP/FP/FN counts. Empty predictions and normal-only series return finite zero-safe metrics instead of division-by-zero errors.

## Fidelity and limits

- This is an independent paper-aligned subset. It does not copy source code from the MIT repository.
- The adaptive metric is genuinely trainable: its SPD factor is a registered parameter and receives gradients through the temporal soft-target loss.
- The conditional density head is a stack of context-conditioned scalar affine transforms, consistent with the reference implementation's one-dimensional per-sensor flow input. It is not a full multivariate cross-sensor MADE.
- The contrastive encoder uses aligned stochastic views and hierarchical max-pooling. It does not reproduce every random-crop and masking detail of the authors' TS2Vec variant.
- Training assumes predominantly healthy/normal windows. Contaminated training data can cause anomalies to be absorbed into the learned normal distribution.
- No scaler is fit internally. Fit normalization only on the training split; use separate validation and healthy calibration data; never select epochs or thresholds on test labels.
- The model does not bundle datasets or author checkpoints. IEEE's abstract reports three widely used benchmarks plus a newly constructed dataset; the official repository README lists SWaT, TEPE, PSM and MSL. The exact mapping between the paper's benchmark description and repository data names remains to be verified against the full experimental protocol. No published paper metric is claimed as a local result.
- The event-wise metric is for one contiguous, already aligned time series. Do not concatenate independent runs or overlapping windows without a documented score-alignment policy.

## Verification

tests/test_aldm_detector.py covers public registry construction, config round-tripping, SPD symmetry/positive definiteness, adaptive-metric gradients, graph/flow/encoder gradients, finite scalar training loss, deterministic evaluation, score tensor shapes, threshold calibration, single-sample batches, invalid inputs and event-wise metric edge cases.

These tests establish numerical/API contracts only. ALDM remains unassessed for SWaT/TEPE/PSM/MSL benchmark reproduction, cross-device generalization, runtime and memory cost, and operational threshold stability.
