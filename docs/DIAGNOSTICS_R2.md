# R2 — Statistical Diagnostic Detectors (initial increment)

Three deterministic, multichannel statistical detection baselines are added under torch_like.diagnostics. They are train-only Gaussian-reference methods, not replacements for 0.9.0 AutoML or deep anomaly detection engines.

## Mathematical contracts

Given healthy reference observations x[t,c], fit channel mean mu[c] and population standard deviation sigma[c]. Clamp sigma[c] by user-configured min_std (default 1e-8). Fitting rejects NaNs, infinities and reference histories shorter than two observations.

At inference calculate z[t,c] = (x[t,c]-mu[c])/sigma[c] using frozen mean and standard deviation.

- ZScoreDetector: |z[t,c]|; alarm when score > sigma_threshold.
- CUSUMDetector: Splus[t,c] = max(0,Splus[t-1,c]+z[t,c]-k); Sminus[t,c] = max(0,Sminus[t-1,c]-z[t,c]-k); score = max(Splus,Sminus); alarm when score > h.
- EWMADetector: e[t,c] = alpha*z[t,c]+(1-alpha)*e[t-1,c]; score = |e[t,c]| / sqrt(alpha/(2-alpha) * (1-(1-alpha)^(2*(t+1)))); alarm when score > control_limit.

CUSUM/EWMA initialize at zero and reset for each input sample/window; they have no hidden cross-window state. A continuous online state implementation is reserved for a separate validated protocol.

## Leakage and engineering constraints

1. Fit with a healthy/development baseline chosen externally. Neither labels nor the sealed test subset are accepted or examined by these detectors.
2. Scoring and alarm generation never modify fitted baseline statistics. Thresholds are declared constructor parameters, not fitted using test-set evidence.
3. Channel names, their order and sample rate must match the fitted baseline. Inference refuses missing/non-finite values.
4. Scores are (N,L,C) per-window/per-time/per-channel arrays in standardized units; they are not fault probabilities or root-cause evidence.
5. A constant channel is stabilized by min_std; that numerical choice does not establish physical sensor health.
6. Builtins are registered explicitly via register_statistical_detectors; importing the library does not mutate a global registry.

## Evidence and limitations

Tests exercise hand-calculated Z-score, CUSUM and finite-horizon EWMA results, no-update-on-inference, metadata mismatch, independently reset windows and registry discovery. These are numerical sanity tests, not production false-alarm rate certification.

The next R2 increments should cover robust signal features, PCA T-squared/SPE, out-of-control run rules and cross-dataset independent benchmark validation.
