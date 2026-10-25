"""Frozen numerical parity vectors from the attached AAS Phase 9 source.

AAS source archive: AAS_code_20261024(1).zip
SHA-256: 86d3134dbe73ba6e9f361cb1dac45fccaaeb2d56f10928a9679999ce485ad8cf
Adapters: AAS/detectors/torch_like_adapters.py
Generated with NumPy default_rng(2060), 2-D input segmented at [5,7], plus
independent 3-D windows. Fixtures are outputs observed by executing those
original AAS classes, not hand-derived expected values.

This test intentionally does not import or ship the AAS platform.
"""

from __future__ import annotations

import unittest

import numpy as np

from torch_like.diagnostics import (
    CUSUMDetector,
    EWMADetector,
    PCASPEDetector,
    SignalBatch,
    project_anomaly_scores,
)

_AAS_GOLDEN = {
    "cusum": {
        "continuous": [
            0.07380162547839741,
            2.623637078189788,
            1.6319903689308735,
            2.7214125669013254,
            2.6874842831615373,
            1.9081107798917705,
            2.3209583713471384,
            1.8849889018304289,
            1.5200101353460163,
            2.4603508843747375,
            4.573971520040817,
            3.252441028143278,
        ],
        "windows": [2.7447084057682436, 0.7996342245802727, 3.692915578881236],
    },
    "ewma": {
        "continuous": [
            0.6738016254783973,
            2.3041357109119063,
            1.1990446149096932,
            2.055015289736991,
            1.8266719760331052,
            2.50811077989177,
            2.3797618327047614,
            1.6706337969326013,
            1.6503205586292808,
            2.165874111758994,
            3.047269331258581,
            2.0834693986313972,
        ],
        "windows": [2.7109071219665095, 1.0293448035353236, 2.8461348564702997],
    },
    "pca_spe": {
        "continuous": [
            0.0006134755872767869,
            2.092160728548987,
            0.013880513955273626,
            3.8445626285023375,
            0.1535320004268827,
            3.165370204758353,
            4.5607203815697686,
            0.026973374943228153,
            1.5676203210680484,
            0.17627797544589685,
            4.538939929056183,
            0.7935118145594608,
        ],
        "windows": [0.7367364642976693, 0.4538038434056465, 0.7937332859640089],
    },
}


class AASPhase9FrozenParityTests(unittest.TestCase):
    @staticmethod
    def _data():
        generator = np.random.default_rng(2060)
        reference = generator.normal(size=(30, 3))
        continuous = generator.normal(size=(12, 3))
        continuous[3:5] += [2.0, -1.0, 0.5]
        continuous[9:11] += [0.0, 3.0, -0.5]
        windows = generator.normal(size=(3, 4, 3))
        windows[0, -1, :] += [2.0, -2.0, 1.0]
        return reference, continuous, windows

    @staticmethod
    def _batch(values):
        if values.ndim == 2:
            return SignalBatch.from_sequence(values, sample_rate_hz=10)
        return SignalBatch(values, sample_rate_hz=10)

    def test_statistical_scores_match_original_aas_golden_vectors(self):
        train, continuous, windows = self._data()
        models = (
            ("cusum", CUSUMDetector(k=0.6, h=4.5), "time_channel", "max"),
            ("ewma", EWMADetector(alpha=0.3, control_limit=2.5), "time_channel", "max"),
            ("pca_spe", PCASPEDetector(n_components=2), "time", None),
        )
        for name, detector, layout, reduction in models:
            with self.subTest(algorithm=name):
                detector.fit(self._batch(train))
                intervals = ((0, 5), (5, 12)) if name != "pca_spe" else ((0, 12),)
                actual = []
                for start, stop in intervals:
                    raw = detector.score(self._batch(continuous[start:stop]))
                    projected = project_anomaly_scores(
                        raw,
                        score_layout=layout,
                        sequence_mode="continuous",
                        channel_reduction=reduction,
                    )
                    self.assertEqual(projected.resolution, "time")
                    actual.extend(projected.values)
                np.testing.assert_allclose(actual, _AAS_GOLDEN[name]["continuous"], rtol=1e-9, atol=1e-9)

                raw = detector.score(self._batch(windows))
                projected = project_anomaly_scores(
                    raw,
                    score_layout=layout,
                    sequence_mode="independent_windows",
                    channel_reduction=reduction,
                )
                self.assertEqual(projected.resolution, "window")
                np.testing.assert_allclose(projected.values, _AAS_GOLDEN[name]["windows"], rtol=1e-9, atol=1e-9)

    def test_reset_of_independent_windows_is_not_stream_continuation(self):
        train, _, windows = self._data()
        for detector in (CUSUMDetector(k=0.6, h=4.5), EWMADetector(alpha=0.3, control_limit=2.5)):
            with self.subTest(detector=type(detector).__name__):
                detector.fit(self._batch(train))
                combined = detector.score(self._batch(windows)).values
                isolated = np.concatenate(
                    [detector.score(self._batch(windows[index : index + 1])).values for index in range(len(windows))],
                    axis=0,
                )
                np.testing.assert_array_equal(combined, isolated)


if __name__ == "__main__":
    unittest.main()
