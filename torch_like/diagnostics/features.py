"""Leakage-neutral physical signal features for fixed-rate telemetry windows.

Inputs are immutable SignalBatch objects with (samples, time, channels) axes.
No feature operator estimates parameters from held-out labels or other windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torch_like.diagnostics.types import SignalBatch


@dataclass(frozen=True)
class Spectrum:
    """One-sided amplitude spectrum or power spectral density per sample/channel."""

    frequency_hz: np.ndarray
    values: np.ndarray
    kind: str


def _validate(batch: SignalBatch, *, min_length: int = 2) -> np.ndarray:
    if not isinstance(batch, SignalBatch):
        raise TypeError("batch must be SignalBatch")
    if batch.values.shape[1] < min_length:
        raise ValueError(f"signal window requires at least {min_length} observations")
    if not np.isfinite(batch.values).all():
        raise ValueError("signal features require finite values; handle missing data upstream")
    if batch.timestamps_s is not None:
        actual = np.diff(batch.timestamps_s, axis=1)
        expected = 1.0 / batch.sample_rate_hz
        if not np.allclose(actual, expected, rtol=1e-4, atol=expected * 1e-5):
            raise ValueError("irregular or mismatched timestamps require explicit resampling")
    return batch.values


def time_domain_features(batch: SignalBatch) -> dict[str, np.ndarray]:
    """Return channel-wise features, each with shape (samples, channels).

    Kurtosis is Pearson (normal population=3); zero-variance windows have
    skewness/kurtosis set to zero instead of producing NaN or infinity.
    """
    x = _validate(batch)
    mean = np.mean(x, axis=1)
    centered = x - mean[:, np.newaxis, :]
    std = np.std(x, axis=1, ddof=0)
    rms = np.sqrt(np.mean(x * x, axis=1))
    peak = np.max(np.abs(x), axis=1)
    scale = np.where(std > 0, std, 1.0)
    standardized = centered / scale[:, np.newaxis, :]
    skewness = np.where(std > 0, np.mean(standardized**3, axis=1), 0.0)
    kurtosis = np.where(std > 0, np.mean(standardized**4, axis=1), 0.0)
    crest_factor = np.divide(peak, rms, out=np.zeros_like(peak), where=rms > 0)
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def fft_amplitude(batch: SignalBatch, *, detrend: bool = True, window: str = "boxcar") -> Spectrum:
    """One-sided coherent-gain-corrected FFT amplitude at physical Hz bins.

    Interior frequencies are doubled; the DC and even-length Nyquist bins are
    not doubled. This gives peak amplitude for a bin-centered sinusoid.
    """
    x = _validate(batch)
    length = x.shape[1]
    if window == "boxcar":
        weights = np.ones(length)
    elif window == "hann":
        weights = np.hanning(length)
    else:
        raise ValueError("window must be 'boxcar' or 'hann'")
    coherent_gain = weights.sum()
    if coherent_gain <= 0:
        raise ValueError("window has zero coherent gain")
    if detrend:
        x = x - np.mean(x, axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(x * weights[np.newaxis, :, np.newaxis], axis=1)) / coherent_gain
    if length % 2 == 0:
        spectrum[:, 1:-1, :] *= 2
    else:
        spectrum[:, 1:, :] *= 2
    return Spectrum(
        frequency_hz=np.fft.rfftfreq(length, d=1.0 / batch.sample_rate_hz),
        values=spectrum,
        kind="amplitude",
    )


def welch_psd(
    batch: SignalBatch,
    *,
    segment_length: int = 256,
    overlap: int | None = None,
    window: str = "hann",
) -> Spectrum:
    """Welch one-sided density in input-unit squared per Hz.

    Segments are detrended independently and no zero padding is performed.
    Hann uses the periodic definition, suitable for FFT spectral estimation.
    """
    x = _validate(batch)
    length = x.shape[1]
    if isinstance(segment_length, bool) or not isinstance(segment_length, int):
        raise ValueError("segment_length must be an integer")
    if not 2 <= segment_length <= length:
        raise ValueError("segment_length must be between 2 and the input window length")
    if overlap is None:
        overlap = segment_length // 2
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < segment_length:
        raise ValueError("overlap must be an integer in [0, segment_length)")
    if window == "hann":
        weights = np.hanning(segment_length + 1)[:-1]
    elif window == "boxcar":
        weights = np.ones(segment_length)
    else:
        raise ValueError("window must be 'hann' or 'boxcar'")
    energy = float(np.dot(weights, weights))
    if energy <= 0:
        raise ValueError("spectral window has zero energy")
    starts = range(0, length - segment_length + 1, segment_length - overlap)
    density = np.zeros((x.shape[0], segment_length // 2 + 1, x.shape[2]), dtype=np.float64)
    count = 0
    for start in starts:
        part = x[:, start : start + segment_length, :]
        part = part - np.mean(part, axis=1, keepdims=True)
        transformed = np.fft.rfft(part * weights[np.newaxis, :, np.newaxis], axis=1)
        power = np.abs(transformed) ** 2 / (batch.sample_rate_hz * energy)
        if segment_length % 2 == 0:
            power[:, 1:-1, :] *= 2
        else:
            power[:, 1:, :] *= 2
        density += power
        count += 1
    density /= count
    return Spectrum(
        frequency_hz=np.fft.rfftfreq(segment_length, d=1.0 / batch.sample_rate_hz),
        values=density,
        kind="psd",
    )
