"""Streaming threshold continuity, checkpoint and fail-closed behavior."""

import json
import unittest

import numpy as np

from torch_like.diagnostics import DiagnosticResult, StreamingThreshold


def scores(values, *, name="CUSUM"):
    x = np.asarray(values, dtype=float)
    assert x.ndim == 2
    return DiagnosticResult(
        "anomaly_detection", "anomaly_score", x[np.newaxis], name
    )


class StreamingThresholdTests(unittest.TestCase):
    def make_adaptive(self):
        return StreamingThreshold(
            algorithm_name="CUSUM",
            mode="adaptive_ewma",
            initial_mean=(0.0, 1.0),
            initial_std=(1.0, 2.0),
            alpha=0.1,
            sigma=2.0,
            freeze_on_alarm=True,
            channel_names=("wheel", "temperature"),
        )

    @staticmethod
    def input_samples():
        return np.array([
            [0.2, 1.2], [0.1, 0.9], [5.0, 8.0], [0.3, 0.6],
            [0.4, 1.3], [0.0, 1.0], [0.9, 6.0], [0.1, 1.1],
        ], dtype=float)

    def test_continuous_equals_segmented_exactly(self):
        x = self.input_samples()
        kwargs = {"channel_names": ("wheel", "temperature")}
        single = self.make_adaptive()
        full = single.decide_chunk(scores(x), start_index=0, **kwargs)
        segmented = self.make_adaptive()
        left = segmented.decide_chunk(scores(x[:3]), start_index=0, **kwargs)
        right = segmented.decide_chunk(scores(x[3:]), start_index=3, **kwargs)
        np.testing.assert_array_equal(
            full.labels, np.concatenate([left.labels, right.labels], axis=1)
        )
        np.testing.assert_array_equal(
            full.thresholds, np.concatenate([left.thresholds, right.thresholds], axis=1)
        )
        self.assertEqual(single.state_dict(), segmented.state_dict())
        self.assertEqual(segmented.observations, 8)

    def test_checkpoint_json_roundtrip_matches_uninterrupted(self):
        x = self.input_samples()
        kwargs = {"channel_names": ("wheel", "temperature")}
        continuous = self.make_adaptive()
        expected = continuous.decide_chunk(scores(x), start_index=0, **kwargs)
        checkpoint_source = self.make_adaptive()
        checkpoint_source.decide_chunk(scores(x[:4]), start_index=0, **kwargs)
        state = json.loads(json.dumps(checkpoint_source.state_dict()))
        restored = self.make_adaptive()
        restored.load_state_dict(state)
        actual = restored.decide_chunk(scores(x[4:]), start_index=4, **kwargs)
        np.testing.assert_array_equal(actual.labels, expected.labels[:, 4:, :])
        np.testing.assert_array_equal(actual.thresholds, expected.thresholds[:, 4:, :])
        self.assertEqual(restored.state_dict(), continuous.state_dict())
        state["mean"][0] = 999
        self.assertNotEqual(restored.state_dict()["mean"][0], 999)

    def test_predecision_cutoff_and_alarm_freeze(self):
        stream = StreamingThreshold(
            algorithm_name="CUSUM", mode="adaptive_ewma",
            initial_mean=0.0, initial_std=1.0, alpha=0.5,
            sigma=2.0, freeze_on_alarm=True,
        )
        result = stream.decide_chunk(scores([[3.0], [0.0]]), start_index=0)
        self.assertEqual(result.thresholds[0, 0, 0], 2.0)
        self.assertEqual(result.thresholds[0, 1, 0], 2.0)
        self.assertEqual(result.labels.tolist(), [[[1], [0]]])
        self.assertEqual(stream.observations, 2)

    def test_fixed_mode_no_adaptive_mutation_and_reset(self):
        stream = StreamingThreshold(
            algorithm_name="CUSUM", mode="fixed", thresholds=(1.0, 2.0)
        )
        first = stream.decide_chunk(scores([[0.5, 2.5], [2.5, 1.0]]), start_index=0)
        self.assertEqual(first.labels.tolist(), [[[0, 1], [1, 0]]])
        self.assertEqual(stream.state_dict()["mean"], None)
        self.assertEqual(stream.observations, 2)
        checkpoint = json.loads(json.dumps(stream.state_dict()))
        fresh = StreamingThreshold(algorithm_name="CUSUM", mode="fixed", thresholds=(1.0, 2.0))
        fresh.load_state_dict(checkpoint)
        self.assertEqual(fresh.observations, 2)
        fresh.reset()
        self.assertEqual(fresh.observations, 0)
        after_reset = fresh.decide_chunk(scores([[0.5, 2.5], [2.5, 1.0]]), start_index=0)
        np.testing.assert_array_equal(after_reset.labels, first.labels)

    def test_reject_wrong_sample_channel_algorithm_metadata_and_discontinuity(self):
        x = scores([[0.2, 1.2], [0.1, 0.9]])
        stream = self.make_adaptive()
        with self.assertRaisesRegex(ValueError, "channel_names"):
            stream.decide_chunk(x, start_index=0)
        with self.assertRaisesRegex(ValueError, "channel_names"):
            stream.decide_chunk(x, start_index=0, channel_names=("temperature", "wheel"))
        with self.assertRaisesRegex(ValueError, "start_index"):
            stream.decide_chunk(x, start_index=1, channel_names=("wheel", "temperature"))
        with self.assertRaisesRegex(ValueError, "start_index"):
            stream.decide_chunk(x, start_index=True, channel_names=("wheel", "temperature"))
        with self.assertRaisesRegex(ValueError, "algorithm_name"):
            stream.decide_chunk(scores([[0.1, 0.2]], name="EWMA"), start_index=0,
                                channel_names=("wheel", "temperature"))
        with self.assertRaisesRegex(ValueError, "channel count"):
            stream.decide_chunk(scores([[0.1]]), start_index=0,
                                channel_names=("wheel", "temperature"))
        wrong_shape = DiagnosticResult(
            "anomaly_detection", "anomaly_score", np.ones((2, 2, 2)), "CUSUM"
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            stream.decide_chunk(wrong_shape, start_index=0)
        self.assertEqual(stream.observations, 0)

    def test_invalid_or_incompatible_checkpoint_is_atomic(self):
        stream = self.make_adaptive()
        stream.decide_chunk(scores([[0.1, 1.0]]), start_index=0,
                            channel_names=("wheel", "temperature"))
        original = stream.state_dict()
        cases = [
            {"schema_version": 2},
            {"observations": -1},
            {"observations": True},
            {"mean": [float("nan"), 2.0]},
            {"variance": [-1.0, 2.0]},
            {"variance": [1.0]},
            {"config": {**original["config"], "sigma": 9.0}},
        ]
        for edits in cases:
            candidate = json.loads(json.dumps(original))
            candidate.update(edits)
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                stream.load_state_dict(candidate)
            self.assertEqual(stream.state_dict(), original)

    def test_invalid_initial_parameters(self):
        cases = [
            {"mode": "fixed", "thresholds": float("nan")},
            {"mode": "fixed", "thresholds": [1.0, float("inf")]},
            {"mode": "fixed", "thresholds": True},
            {"mode": "fixed", "thresholds": 1.0, "initial_std": 1.0},
            {"mode": "adaptive_ewma", "initial_mean": 0.0, "initial_std": 0.0},
            {"mode": "adaptive_ewma", "initial_mean": (0, 1), "initial_std": (1,)},
            {"mode": "adaptive_ewma", "initial_mean": 0.0, "initial_std": 1.0, "alpha": 0},
        ]
        for params in cases:
            with self.subTest(params=params), self.assertRaises(ValueError):
                StreamingThreshold(algorithm_name="CUSUM", **params)


if __name__ == "__main__":
    unittest.main()
