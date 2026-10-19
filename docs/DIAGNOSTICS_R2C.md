# R2c — Diagnostic Window Features to Native Sklearn

This increment provides a minimal but executable bridge between the
statistical feature operators and a native scikit-learn anomaly estimator.

## Pipeline

1. A caller provides labeled or unlabeled **healthy development windows** as
   `SignalBatch(N,L,C)`. The algorithm is not responsible for making a
   train/validation/test split or for checking whether windows overlap.
2. `WindowFeatureTransformer` computes ordered, finite, channel-wise window
   features. The vector is channel-major, with explicit feature names:
   `channel__feature`. Allowed features: mean, std, rms, peak, crest_factor,
   skewness and Pearson kurtosis.
3. `SklearnWindowAnomalyDetector.fit(reference)` trains a native
   `sklearn.ensemble.IsolationForest` by default. A custom estimator with
   compatible `fit` and `decision_function` may also be injected.
4. `score(test)` computes features without modifying the fitted native
   estimator, negates the sklearn decision function so **larger means more
   anomalous**, and returns `DiagnosticResult(N,)` with threshold 0.0.
   `alarm_mask` uses this declared native decision boundary.
5. The native estimator remains accessible as `detector.estimator`. No
   masquerading as a `torch.nn.Module` occurs.

## Boundary checks

The trained reference and inference windows must match **sampling rate,
channel name/order, physical units and window length**. Irregular explicit
timestamps fail the feature-validation guard. Missing values require
upstream cleaning.

Threshold zero is the native IsolationForest decision-function boundary,
affected by its contamination parameter. It is **not** a guaranteed
false-alarm probability nor a calibration from an independent evaluation
dataset. A later stage must introduce operating-condition / group-split
benchmarking and held-out calibration where applicable.

The algorithm never accepts ground truth labels as a fitting argument and
does not select its hyperparameters using sealed test results.

## Verification

Tests check analytical window feature values, channel-major ordering,
deterministic native IsolationForest score sign, a fake estimator proving
no refit during inference, invalid input rejection, and opt-in registry
discovery. Complete quality/coverage/wheel and CPU/Gloo CI remain required.

This is an anomaly detector; it does not identify fault type, isolate fault
origin, or provide root-cause inference. Those are separate diagnostic tasks.
