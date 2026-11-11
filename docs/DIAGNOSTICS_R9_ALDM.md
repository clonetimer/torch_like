# Diagnostics R9: ALDM latent-distribution anomaly detection

Date: 2026-11-11

Status: paper-aligned portable subset; maturity and benchmark evidence remain **unassessed**.

## Literature basis

Yu Liu, Yifan Song, Shaolong Shu, Feng Lin, Jun Wang and Yafeng Guo, “Adaptive Latent Distribution Modeling for Industrial Time Series Anomaly Detection”, IEEE Transactions on Automation Science and Engineering, 23 (2026), 7893–7907. DOI: 10.1109/TASE.2026.3674236.

- Primary paper: https://doi.org/10.1109/TASE.2026.3674236
- Official implementation: https://github.com/YuLiu-61/ALDM
- Official repository license: MIT

The paper combines sensor-relationship learning, adaptive latent temporal representation, hierarchical contrastive learning and conditional normalizing-flow likelihood estimation. The upstream README lists SWaT, TEPE, PSM and MSL as supported datasets.

## Public API

```python
import torch
from torch_like.models import ALDMDetector

model = ALDMDetector(
    input_dim=8,
    seq_len=64,
    d_model=32,
    metric_dim=24,
    metric_rank=6,
    flow_hidden_dim=32,
    flow_layers=1,
)

# Caller supplies already-windowed, training-only-normalized data.
batch = model.make_detection_batch(train_windows)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
optimizer.zero_grad(set_to_none=True)
loss = model.fit_step(batch)
loss.backward()
optimizer.step()

model.eval()
with torch.no_grad():
    scores = model.score_batch(
        model.make_detection_batch(test_windows),
        reduction="none",
    )
# One negative log-likelihood anomaly score per window; larger means less likely.
```

The model is unsupervised: labels are not required by `fit_step`. Inference is deterministic and returns one scalar negative log-likelihood score per input window. Threshold calibration is a separate validation-stage operation.

## Architecture

1. **Sensor temporal encoding**: a shared LSTM encodes each channel's window into a sensor/time representation.
2. **Sensor graph context**: scaled dot-product attention learns a row-normalized relation matrix between sensor summaries; graph messages and a temporal transition residual update each sensor's hidden sequence.
3. **Adaptive temporal metric**: a low-rank positive-definite metric (W=UU^T+\epsilon I) computes pairwise latent distances for soft temporal targets. The factor is a registered trainable parameter and receives gradients from the contrastive objective.
4. **Hierarchical contrastive learning**: two noisy/masked views pass through a compact dilated temporal encoder. Symmetric instance-level InfoNCE and metric-conditioned temporal soft-target losses are evaluated across max-pooled temporal scales.
5. **Conditional scalar flow**: per-sensor hidden context conditions affine MAF-style transforms of scalar sensor observations. The standard-normal base density and log-Jacobian produce a conditional log-likelihood; its negative is the anomaly score.

The model is registered as `ALDMDetector` in the `detection` domain and uses the existing `DetectionBatch`, `fit_step`, `predict` and `score_batch` contracts.

## Important upstream implementation audit

The official repository is useful as a method reference and is MIT-licensed, but several implementation details should not be copied into a production evaluation pipeline:

- The reference `compute_metric_learning` creates a new `SPDLayer` inside the metric function rather than keeping it as a persistent model parameter. The soft temporal target is subsequently detached in the contrastive training path, so the dynamically created metric is not optimized as an ordinary model parameter.
- The SWaT and TEPE loaders fit `StandardScaler` before the chronological train/test split, allowing test-period statistics to influence preprocessing.
- The main training script compares checkpoints using test-set ROC-AUC, which leaks test labels into model selection.
- The reference `test` path calls the TS2Vec `fit` method, which includes random crop generation and a contrastive training loss; this is not a clean separation between training and deterministic inference.

The portable implementation keeps the metric inside the module/optimizer, does not detach its soft-target path, separates training augmentation from inference, and does not bundle or copy the upstream data loaders or test-AUC checkpoint selection.

## Implementation boundary

- This is an independent paper-aligned portable subset, not a line-by-line reproduction of the authors' repository.
- The contrastive encoder is a compact dilated temporal encoder inspired by TS2Vec; it is not a claim of exact TS2Vec implementation parity.
- The flow operates on scalar sensor observations conditioned on graph context, matching the one-dimensional conditional-flow use in the reference architecture; it is not a general multivariate MAF library.
- The learned sensor attention matrix is a model-internal association, not a physical causal graph or certified fault-propagation graph.
- The model does not include dataset downloads, dataset-specific preprocessing, event-label loaders, threshold selection, benchmark runners, or published checkpoints.
- Callers must fit scalers on training data only, select models and thresholds on validation data only, and keep the final test period sealed. Do not use test AUC for checkpoint selection.
- Published SWaT/TEPE/PSM/MSL results are not reproduced or claimed by this repository.

## Verification

`tests/test_detection_aldm.py` covers:

- Registry and public API construction, configuration and native detection-task contracts.
- Window-level finite anomaly scores and reduction semantics.
- Deterministic evaluation and normalized sensor attention.
- Symmetry, non-negativity and gradients of the positive-definite adaptive metric.
- Gradient propagation through the metric, conditional flow and contrastive encoder.
- Batch-size-one and odd-length windows.
- Invalid dimensions, nonfinite inputs and configuration boundaries.

These tests establish API and numerical contracts only. They do not establish SWaT, TEPE, PSM or MSL benchmark performance, cross-domain generalization, or spacecraft hardware validity.
