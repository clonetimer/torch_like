"""AAS numerical score-axis compatibility without importing AAS's platform."""

import json
import unittest

import numpy as np

from torch_like.diagnostics import (
    CUSUMDetector,
    DiagnosticResult,
    EWMADetector,
    PCASPEDetector,
    SignalBatch,
    project_anomaly_scores,
)


class ScoreProjectionTests(unittest.TestCase):
    @staticmethod
    def result(values, name="score"):
        return DiagnosticResult(
            "anomaly_detection", "anomaly_score", np.asarray(values, dtype=np.float64), name
        )

    def test_continuous_time_channel_max_matches_explicit_reference(self):
        x = np.array([[[1., 9.], [4., 2.], [8., 3.]]])
        projected = project_anomaly_scores(
            self.result(x), score_layout="time_channel",
            sequence_mode="continuous", channel_reduction="max",
        )
        np.testing.assert_array_equal(projected.values, [9., 4., 8.])
        self.assertEqual(projected.resolution, "time")
        self.assertEqual(projected.source_shape, (1, 3, 2))
        self.assertEqual(json.loads(json.dumps(projected.to_dict()))["resolution"], "time")
        self.assertFalse(projected.values.flags.writeable)

    def test_independent_windows_select_last_timestep_then_reduce(self):
        x = np.array([
            [[100., 100.], [1., 4.]],
            [[200., 200.], [9., 2.]],
            [[300., 300.], [3., 1.]],
        ])
        p = project_anomaly_scores(
            self.result(x), score_layout="time_channel",
            sequence_mode="independent_windows", channel_reduction="max",
        )
        np.testing.assert_array_equal(p.values, [4., 9., 3.])
        self.assertEqual(p.resolution, "window")
        avg = project_anomaly_scores(
            self.result(x), score_layout="time_channel",
            sequence_mode="independent_windows", channel_reduction="mean",
        )
        np.testing.assert_allclose(avg.values, [2.5, 5.5, 2.0])

    def test_pca_time_layout_and_neural_window_layout(self):
        time = self.result([[1., 2., 3.]], "PCASPEDetector")
        p = project_anomaly_scores(time, score_layout="time", sequence_mode="continuous")
        np.testing.assert_array_equal(p.values, [1., 2., 3.])
        window = self.result([0.25, 0.5], "USAD")
        q = project_anomaly_scores(window, score_layout="window", sequence_mode="independent_windows")
        np.testing.assert_array_equal(q.values, [0.25, 0.5])

    def test_reject_ambiguous_axis_instead_of_guessing(self):
        cases = [
            ([[1., 2.]], "time", "independent_windows", "max"),
            ([[1., 2.]], "window_channel", "continuous", "max"),
            ([[1., 2.]], "window_channel", "independent_windows", None),
            ([1., 2.], "window", "continuous", None),
            ([1., 2.], "time", "independent_windows", None),
            ([[1., 2.]], "time", "continuous", "max"),
        ]
        for values, layout, mode, reduction in cases:
            with self.subTest(layout=layout, mode=mode, reduction=reduction), self.assertRaises(ValueError):
                project_anomaly_scores(
                    self.result(values), score_layout=layout, sequence_mode=mode,
                    channel_reduction=reduction,
                )

    def test_statistical_cusum_matches_independent_reference_recursion(self):
        raw = np.array([[[-2., 1.], [-1., 0.], [0., -1.], [1., 0.], [2., 1.]]])
        batch = SignalBatch(raw, sample_rate_hz=1, channel_names=("a", "b"))
        stream = np.array([[[0., 0.], [1., 0.], [2., 2.], [-3., 0.]]])
        tested = SignalBatch(stream, sample_rate_hz=1, channel_names=("a", "b"))
        algorithm = CUSUMDetector(k=0.5).fit(batch)
        model_scores = algorithm.score(tested)
        p = project_anomaly_scores(
            model_scores, score_layout="time_channel",
            sequence_mode="continuous", channel_reduction="max",
        )
        standardized = (stream[0] - algorithm.mean_) / algorithm.std_
        plus = np.zeros(2)
        minus = np.zeros(2)
        expected = []
        for sample in standardized:
            plus = np.maximum(0, plus + sample - 0.5)
            minus = np.maximum(0, minus - sample - 0.5)
            expected.append(float(np.maximum(plus, minus).max()))
        np.testing.assert_allclose(p.values, expected, atol=1e-12)

        independent = SignalBatch(
            np.concatenate([stream, stream + 0.1], axis=0), sample_rate_hz=1,
            channel_names=("a", "b"),
        )
        scores = algorithm.score(independent)
        q = project_anomaly_scores(
            scores, score_layout="time_channel",
            sequence_mode="independent_windows", channel_reduction="max",
        )
        np.testing.assert_allclose(q.values, scores.values[:, -1].max(axis=1))

    def test_statistical_ewma_and_pca_scores_keep_axis_semantics(self):
        rng = np.random.default_rng(11)
        baseline = SignalBatch(rng.normal(size=(3, 40, 3)), sample_rate_hz=50)
        live = SignalBatch(rng.normal(size=(2, 9, 3)), sample_rate_hz=50)
        ewma = EWMADetector(alpha=0.2).fit(baseline)
        q = project_anomaly_scores(
            ewma.score(live), score_layout="time_channel",
            sequence_mode="independent_windows", channel_reduction="max",
        )
        np.testing.assert_allclose(q.values, ewma.score(live).values[:, -1, :].max(axis=1))
        pca = PCASPEDetector(n_components=2).fit(baseline)
        projection = project_anomaly_scores(
            pca.score(live), score_layout="time", sequence_mode="independent_windows"
        )
        np.testing.assert_allclose(projection.values, pca.score(live).values[:, -1])


if __name__ == "__main__":
    unittest.main()
