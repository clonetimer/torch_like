# R2b — Multivariate PCA monitoring and physical signal features

This increment extends the statistical diagnostics library without changing native
PyTorch/sklearn objects or the protected root API baseline.

## PCA T² and SPE/Q

`PCAT2Detector` and `PCASPEDetector` fit channel means, sample standard
deviations, and eigenvectors only from caller-supplied **healthy development**
reference data. The retained subspace has an explicitly configured number
of components smaller than the channel count. A singular retained eigenvalue
is rejected rather than silently inverted.

Given standardized vector `z`, retained loadings `P` and eigenvalues
`lambda`, the statistics are:

- T² = sum((z P)^2 / lambda), sensitive to deviation within the learned subspace.
- SPE/Q = || z - (z P) Pᵀ ||², sensitive to orthogonal residuals.

The `score` method returns sample × time anomaly scores without changing
trained parameters. An alarm threshold is **not** inferred from the test data.
`calibrate(independent_development_batch, quantile=0.99)` is a separate
explicit step; calling `alarm_mask` without calibration fails closed.

The implementation rejects mismatches of channel names/order, physical units,
sampling rate, nonfinite input, and insufficient or rank-deficient references.
It rejects reuse of the identical reference batch object for calibration, but
**cannot independently prove nonoverlap** if the caller sends different views
of the same data. Dataset-level group/time split integrity remains the
responsibility of the evaluation pipeline.

This is empirical development-set calibration, not an analytically guaranteed
false-alarm rate under non-Gaussian or autocorrelated process conditions.

## Signal-domain features

- `time_domain_features`: mean, std, RMS, peak magnitude, crest factor,
  Pearson kurtosis, skewness per sample and channel.
- `fft_amplitude`: one-sided physical-frequency FFT spectrum with coherent
  gain correction; DC and Nyquist components are not doubled.
- `welch_psd`: segmented averaged one-sided power spectral density,
  in physical signal-unit²/Hz, with parameterized segment size and overlap.

Frequency axes use `np.fft.rfftfreq(..., d=1/sample_rate_hz)`. If explicit
timestamps are present they must agree with the sample rate and be evenly
spaced. Irregular telemetry must be resampled explicitly upstream, never
interpreted as uniformly sampled without verification.

Frequency features are deterministic and label-free. They do not imply an
automated fault type, isolation decision, or production-ready false alarm
threshold. Window boundaries remain provided by the caller.

## Verification

Regression tests use a known bin-centered sinusoid to check amplitude in
physical frequency bins, PSD energy conservation for rectangular windows,
multichannel shapes, spectral input validation, analytically solvable
two-channel PCA T²/SPE geometry and independent calibration/frozen scoring.
Full repository Ruff/coverage/wheel and CPU/Gloo checks remain mandatory.

## Next stages

Introduce explicit feature-transform-to-estimator pipelines, held-out
group-based calibration experiments, PCA component selection validation,
and robust thresholds under autocorrelation and operating-condition drift.
