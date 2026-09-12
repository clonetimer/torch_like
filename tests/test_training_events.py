"""Verify the training-loop lifecycle events emitted by ``torch_backend``."""

import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from torch_like.automl.sklearn_backend import AlgorithmCandidate
from torch_like.automl.specs import TaskType
from torch_like.automl.torch_backend import run_torch_candidate, training_events


class TrainingEventsTests(unittest.TestCase):
    def test_lifecycle_events_fire(self):
        seen: list[tuple[str, dict]] = []
        dispose = training_events.on(lambda name, **kw: seen.append((name, kw)))
        try:
            rng = np.random.RandomState(0)
            x = [rng.rand(10, 1).astype("float32") for _ in range(4)]
            y = rng.rand(4, 3, 1).astype("float32")
            split = types.SimpleNamespace(train=np.array([0, 1]), val=np.array([2, 3]), test=np.array([2, 3]))
            candidate = AlgorithmCandidate(
                name="DLinear",
                preprocessing="none",
                parameters={"moving_avg": 3},
                backend="pytorch",
            )
            with tempfile.TemporaryDirectory() as tmp:
                run_torch_candidate(
                    candidate,
                    TaskType.TIME_SERIES_FORECASTING,
                    x,
                    y,
                    split,
                    seed=0,
                    epochs=2,
                    batch_size=2,
                    learning_rate=1e-3,
                    workers=0,
                    trial_dir=Path(tmp),
                    save_checkpoint=False,
                    device="cpu",
                )
        finally:
            dispose()

        names = [name for name, _ in seen]
        self.assertIn("training_start", names)
        self.assertIn("training_end", names)
        self.assertEqual(names.count("epoch_end"), 2)

        start = dict(seen[names.index("training_start")][1])
        self.assertEqual(start["candidate"], "DLinear")
        self.assertEqual(start["task_type"], TaskType.TIME_SERIES_FORECASTING.value)

        last_epoch = [kw for name, kw in seen if name == "epoch_end"][-1]
        self.assertEqual(last_epoch["epoch"], 2)
        self.assertIn("train_loss", last_epoch)

        end = seen[names.index("training_end")][1]
        self.assertEqual(end["candidate"], "DLinear")
        self.assertIn("summary", end)
        self.assertEqual(end["summary"]["epochs_ran"], 2)

    def test_no_listeners_by_default_is_noop(self):
        # With no subscribers the emitter must not raise and must stay empty.
        self.assertEqual(len(training_events._handlers), 0)


if __name__ == "__main__":
    unittest.main()
