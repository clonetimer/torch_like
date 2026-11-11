# Diagnostics R6: scientific dynamics and mask-conditional uncertainty

Date: 2026-11-11

Status: implementation candidate; algorithm maturity and benchmark evidence remain **unassessed**.

## Scope decision

The IMA literature intake is used to fill capability gaps, not to maximize the number of registered neural networks. R6 adds two independent diagnostic capabilities:

1. **SINDyDetector** — sparse nonlinear dynamics identification using sequentially thresholded least squares (STLSQ), plus a derivative-residual anomaly score.
2. **ModalityConditionedConformalClassifier** — a post-hoc Mondrian split-conformal calibrator that groups calibration scores by the exact sensor/modality availability mask.

These additions do not change the 172-model neural model catalog or the Stable package allowlist. SINDy is an explicit classical diagnostic estimator; the conformal component is a calibration utility rather than a trained model.

## Literature-to-capability mapping

| IMA material | Capability gap | R6 disposition |
|---|---|---|
| Brunton, Proctor and Kutz, “Discovering governing equations from data by sparse identification of nonlinear dynamical systems”, PNAS (2016), DOI: 10.1073/pnas.1517384113 | Interpretable sparse dynamics; healthy-model residuals | Implemented as SINDyDetector |
| Karniadakis et al., “Physics-informed machine learning”, Nature Reviews Physics (2021), DOI: 10.1038/s42254-021-00314-5 | Physics/data hybrid learning and inverse problems | Architecture direction; PINN training contract deferred |
| “Universal Differential Equations for Scientific Machine Learning” (2021 preprint) | Known physics plus learned unknown terms | Deferred until ODE solver, residual and adjoint contracts are designed |
| “Conformal Fusion Under Missing Modalities” (MCCF, IMA local arXiv:2608.07183v1 preprint) | Calibrated prediction under changing sensor/modality availability | Implemented only as a mask-conditional split-conformal calibration primitive; not a reproduction of the full fusion network |
| ALDM paper and official repository | Unsupervised multivariate time-series anomaly detection | Deferred pending source correction and independent reproduction |
| AGIL paper notes | Joint graph-based fault classification and RUL estimation | High-priority follow-up; needs a multi-task batch/loss contract |
| HSNN-IB paper notes | Hierarchical subgraph information bottleneck for RUL | Deferred: sensor grouping and external validation evidence need further audit |
| FedMAPS (MSSP, 2026) | Few-shot cross-domain federated diagnosis | Deferred: requires a real client/server federated optimizer and prototype aggregation, not only DDP |
| AI-Feynman, SymMatika, DeepONet, Fourier Neural Operator and Hamiltonian neural networks | Symbolic discovery and scientific neural operators | Retain as research candidates; their assumptions and problem domains need separate admission tests |

## 1. SINDyDetector

### API

- Input: SignalBatch with shape (samples, time, state_channels).
- Optional control input: finite array with shape (samples, time, control_channels).
- Derivative estimation: finite differences on each independent trajectory. No derivative is taken across sample/window boundaries.
- Candidate library: constant, polynomial terms through degree 4, and optional sine/cosine terms.
- Regression: column-normalized STLSQ with optional ridge stabilization.
- Inspectability: coefficients_, feature_names_, equations_, predict_derivative(...).
- Fault scoring: one RMS derivative-residual score per trajectory.
- Alarm threshold: must be calibrated on independent healthy development trajectories with calibrate(...).

### Usage

~~~python
from torch_like.diagnostics import SINDyDetector, SignalBatch

reference = SignalBatch(
    healthy_trajectories,  # (samples, time, state_channels)
    sample_rate_hz=100.0,
    channel_names=("wheel_speed", "motor_current"),
    units=("rad/s", "A"),
)
healthy_development = SignalBatch(
    healthy_development_values,
    sample_rate_hz=100.0,
    channel_names=("wheel_speed", "motor_current"),
    units=("rad/s", "A"),
)
test_batch = SignalBatch(
    test_values,
    sample_rate_hz=100.0,
    channel_names=("wheel_speed", "motor_current"),
    units=("rad/s", "A"),
)
detector = SINDyDetector(
    polynomial_degree=2,
    sparsity_threshold=0.05,
    ridge_alpha=1e-8,
).fit(reference)

print(detector.equations_)
detector.calibrate(healthy_development, quantile=0.99)
scores = detector.score(test_batch)
alarms = detector.alarm_mask(test_batch)
~~~

If known control inputs are available, pass them to fit, calibrate, score and predict_derivative using the same channel order.

### Limits

- Finite-difference derivatives are noise-sensitive. Filtering/denoising must be selected and documented by the caller; total-variation derivative estimation is not included in R6.
- The current candidate library is polynomial plus optional trigonometric terms; it is not an unrestricted symbolic-regression engine.
- A low residual is evidence of consistency with the fitted healthy dynamics, not proof of physical health. Unmodeled operating conditions can also increase residuals.
- Independent healthy calibration is required before alarm_mask; the empirical quantile is not a distribution-free fault guarantee.

## 2. ModalityConditionedConformalClassifier

### API

- Input: class-probability matrix (samples, classes), integer class indices, and a boolean availability mask (samples, modalities).
- Nonconformity: 1 - p(true_class) (LAC).
- Calibration: one finite-sample quantile per exact availability-mask group.
- Prediction: boolean class-inclusion matrix, per-sample quantile, and set sizes.
- No global fallback: an unseen test-time mask raises an explicit error.
- Small groups: the finite-sample correction may produce quantile 1.0 and an all-class set. This is conservative but uninformative; collect more calibration data or raise min_calibration_per_mask.

### Usage

~~~python
from torch_like.diagnostics import ModalityConditionedConformalClassifier

calibrator = ModalityConditionedConformalClassifier(alpha=0.1, min_calibration_per_mask=20)
calibrator.fit(calibration_probabilities, calibration_labels, calibration_mask)
prediction_sets = calibrator.predict_set(test_probabilities, test_mask)

# Each row contains the class indices retained by the conformal set.
class_sets = prediction_sets.class_indices
set_sizes = prediction_sets.set_sizes
~~~

### Statistical scope

The finite-sample coverage statement is marginal **within each calibrated availability-mask group**, assuming exchangeability between calibration and future samples conditional on that exact mask. It does not guarantee coverage under arbitrary temporal, device, class-prior or operating-condition shift. It does not estimate missing modalities, train a fusion backbone, or implement the evidential/Dempster–Shafer fusion and end-to-end set-size objective described in the local preprint.

## Deferred candidates and admission gates

### ALDM

The official repository is MIT-licensed, but the inspected implementation should not be copied as-is. In models/soft_ts2vec.py, compute_metric_learning constructs a fresh SPDLayer inside the function rather than using a registered model parameter; the resulting time-lag matrix is then detached before the contrastive loss. The TS2Vec instance also declares an SPD layer that is not used by that function. This makes the advertised adaptive metric-learning path questionable in the current source. A future port must first verify the paper equations, move the metric parameters into the model/optimizer, provide a stable inference score, and reproduce at least one public benchmark split without test-label leakage.

Source inspected: https://github.com/YuLiu-61/ALDM/blob/main/models/soft_ts2vec.py

### AGIL and HSNN-IB

AGIL is a strong candidate because the current catalog has GIN and separate diagnosis/RUL families but no audited joint diagnosis-and-RUL model. It should be admitted only after a multi-task batch, loss weighting, task-specific output and evaluation contract is defined. HSNN-IB remains a research candidate until its sensor grouping is configurable and its evidence across devices/operating conditions is stronger.

### FedMAPS

FedMAPS combines FOMAML, adaptive cross-domain contrastive learning and global prototype aggregation over clients with partially disjoint class sets. The current DDP runtime is not a federated learning server/client runtime. Do not label a DDP training run as FedMAPS or claim federated privacy/communication behavior without client-local data isolation and server aggregation tests.

### PINN and UDE

PINN and UDE are frameworks, not interchangeable drop-in estimators. Admission requires a public contract for user-supplied physical residuals, initial/boundary conditions, solver tolerances, differentiability/adjoints and hybrid known/learned terms. SINDy is a smaller first step into scientific ML; it does not replace these methods.

## Verification and maturity

R6 adds contract tests for synthetic ODE identification, control-input handling, independent healthy calibration, mask-specific finite-sample quantiles, unseen masks, invalid inputs and immutable outputs.

Synthetic contract tests establish API and numerical behavior only. They do **not** establish benchmark superiority, cross-device generalization, spacecraft flight validity or paper-score reproduction. Both additions remain unassessed until evaluated on public benchmark datasets and independently generated domain-specific data.
