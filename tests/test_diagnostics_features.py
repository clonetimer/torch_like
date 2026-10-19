"""Independent numerical references for frequency and time-domain signal features."""

import unittest

import numpy as np

from torch_like.diagnostics import SignalBatch
from torch_like.diagnostics.features import fft_amplitude, time_domain_features, welch_psd


def sine_batch(*, amplitude=2.0, frequency=8.0, rate=128.0, length=128):
    t = np.arange(length) / rate
    return SignalBatch.from_sequence(
        (amplitude * np.sin(2 * np.pi * frequency * t))[:, None],
        sample_rate_hz=rate,
        channel_names=("vibration",),
        units=("g",),
        timestamps_s=t,
    )


class SignalFeaturesTests(unittest.TestCase):
    def test_sine_fft_frequency_and_one_sided_amplitude_in_hz(self):
        batch = sine_batch(amplitude=2, frequency=8)
        spectrum = fft_amplitude(batch)
        self.assertEqual(spectrum.kind, "amplitude")
        self.assertEqual(spectrum.values.shape, (1, 65, 1))
        self.assertEqual(float(spectrum.frequency_hz[8]), 8.0)
        self.assertEqual(int(np.argmax(spectrum.values[0, :, 0])), 8)
        self.assertAlmostEqual(float(spectrum.values[0, 8, 0]), 2.0, places=12)
        self.assertLess(float(spectrum.values[0, 0, 0]), 1e-12)

    def test_fft_dc_and_nyquist_do_not_double(self):
        rate = 16
        signal = np.full(16, 3.0)
        batch = SignalBatch.from_sequence(signal[:, None], sample_rate_hz=rate)
        self.assertAlmostEqual(float(fft_amplitude(batch, detrend=False).values[0, 0, 0]), 3.0)
        alternating = 2.0 * ((-1.0) ** np.arange(16))
        batch = SignalBatch.from_sequence(alternating[:, None], sample_rate_hz=rate)
        self.assertAlmostEqual(float(fft_amplitude(batch).values[0, -1, 0]), 2.0)

    def test_welch_density_respects_parseval_for_bin_centered_tone(self):
        batch = sine_batch()
        spectrum = welch_psd(batch, segment_length=128, overlap=0, window="boxcar")
        self.assertEqual(spectrum.kind, "psd")
        delta_hz = float(spectrum.frequency_hz[1] - spectrum.frequency_hz[0])
        self.assertAlmostEqual(float(np.sum(spectrum.values[0, :, 0]) * delta_hz), 2.0, places=11)
        self.assertEqual(int(np.argmax(spectrum.values[0, :, 0])), 8)

    def test_welch_multisegment_and_multichannel_shape(self):
        first = sine_batch(rate=128, length=256).values
        batch = SignalBatch(np.concatenate([first, first], axis=2), sample_rate_hz=128)
        result = welch_psd(batch, segment_length=64, overlap=32)
        self.assertEqual(result.values.shape, (1, 33, 2))
        np.testing.assert_allclose(result.values[0, :, 0], result.values[0, :, 1])

    def test_time_statistics_use_time_axis_not_channel_axis(self):
        batch = SignalBatch(
            np.array([[[-1.0, 2.0], [1.0, 2.0], [-1.0, 2.0], [1.0, 2.0]]]),
            sample_rate_hz=10,
        )
        features = time_domain_features(batch)
        for value in features.values():
            self.assertEqual(value.shape, (1, 2))
        np.testing.assert_allclose(features["mean"][0], [0, 2])
        np.testing.assert_allclose(features["std"][0], [1, 0])
        np.testing.assert_allclose(features["rms"][0], [1, 2])
        np.testing.assert_allclose(features["crest_factor"][0], [1, 1])
        np.testing.assert_allclose(features["skewness"][0], [0, 0])
        np.testing.assert_allclose(features["kurtosis"][0], [1, 0])

    def test_fft_rejects_irregular_and_inconsistent_time_axis(self):
        t = np.array([0, 0.1, 0.2, 0.31])
        batch = SignalBatch.from_sequence(np.ones((4, 1)), sample_rate_hz=10, timestamps_s=t)
        with self.assertRaisesRegex(ValueError, "resampling"):
            fft_amplitude(batch)
        with self.assertRaisesRegex(ValueError, "resampling"):
            welch_psd(batch, segment_length=4)
        with self.assertRaisesRegex(ValueError, "resampling"):
            time_domain_features(batch)

    def test_spectrum_parameter_guards_and_missing_data(self):
        data = sine_batch()
        with self.assertRaisesRegex(ValueError, "window"):
            fft_amplitude(data, window="unknown")
        for length in (0, 1, 129, True):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "segment_length"):
                welch_psd(data, segment_length=length)
        for overlap in (-1, 128, True):
            with self.subTest(overlap=overlap), self.assertRaisesRegex(ValueError, "overlap"):
                welch_psd(data, segment_length=128, overlap=overlap)
        with self.assertRaisesRegex(ValueError, "window"):
            welch_psd(data, window="unknown")
        missing = SignalBatch.from_sequence(
            np.array([[np.nan], [1.0]]), sample_rate_hz=100, allow_missing=True
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            fft_amplitude(missing)


if __name__ == "__main__":
    unittest.main()
