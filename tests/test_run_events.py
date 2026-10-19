"""Training events must survive the spawned trial process and be scoped per run.

Trials run in spawned subprocesses, so a parent-side subscriber sees nothing
unless events are recorded in the trial and replayed onto the run's emitter.

All five backends emit a uniform ``training_start`` / ``training_end`` pair so a
single event stream covers sklearn, baseline, PyTorch, and both anomaly backends.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from torch_like.automl.anomaly_backend import run_anomaly_baseline, run_anomaly_pytorch
from torch_like.automl.data import DataSplit, ValidationFold
from torch_like.automl.events import (
    bind_run_events,
    current_run_events,
    install_event_log,
    read_event_log,
    training_events,
)
from torch_like.automl.runners import run_baseline_single, run_sklearn_cv
from torch_like.automl.sklearn_backend import AlgorithmCandidate
from torch_like.automl.specs import (
    ArtifactSpec,
    DatasetSpec,
    LeaderboardConfig,
    ResourceSpec,
    SearchSpec,
    SplitSpec,
    SplitType,
    TaskSpec,
    TaskType,
)
from torch_like.automl.task import LeaderboardTask
from torch_like.core import Scope


def _write_csv(path: Path) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40)})
    frame["label"] = (frame["f1"] + frame["f2"] > 0).astype(int)
    frame.to_csv(path, index=False)


def _config(
    root: Path,
    *,
    save_training_events: bool,
    algorithms: tuple[str, ...] = ("FTTransformer",),
    torch_epochs: int = 2,
) -> LeaderboardConfig:
    return LeaderboardConfig(
        task_id="event-probe",
        dataset=DatasetSpec(path=str(root / "data.csv"), label_columns=("label",)),
        task=TaskSpec(type="tabular_classification"),
        split=SplitSpec(type=SplitType.STRATIFIED),
        resources=ResourceSpec(timeout_per_trial_minutes=1, max_total_minutes=2),
        search=SearchSpec(
            strategy="none",
            max_trials=1,
            preprocessing=("standard",),
            algorithms=algorithms,
            torch_epochs=torch_epochs,
        ),
        artifacts=ArtifactSpec(output_dir=str(root / "runs"), save_training_events=save_training_events),
    )


def _anomaly_arrays():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(40, 8, 2)).astype(np.float32)
    y = np.zeros((40, 2), dtype=np.uint8)
    y[22:27, 0] = 1
    y[32:37, 1] = 1
    split = DataSplit(np.arange(20), np.arange(20, 30), np.arange(30, 40))
    return x, y, split


class EventLogTests(unittest.TestCase):
    def test_log_records_events_and_stops_when_disposed(self):
        with tempfile.TemporaryDirectory() as directory:
            trial_dir = Path(directory)
            dispose = install_event_log(trial_dir)
            training_events.emit("training_start", candidate="Probe")
            self.assertEqual(read_event_log(trial_dir), [("training_start", {"candidate": "Probe"})])
            dispose()
            training_events.emit("training_end", candidate="Probe")
            self.assertEqual(read_event_log(trial_dir), [("training_start", {"candidate": "Probe"})])

    def test_missing_log_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(read_event_log(Path(directory)), [])


class RunEventScopeTests(unittest.TestCase):
    def test_run_emitter_is_detached_after_the_scope(self):
        self.assertIsNone(current_run_events())
        with Scope("run") as scope:
            emitter = bind_run_events(scope)
            self.assertIs(current_run_events(), emitter)
        self.assertIsNone(current_run_events())

    def test_nested_run_emitters_restore_in_order(self):
        with Scope("outer") as outer:
            first = bind_run_events(outer)
            with Scope("inner") as inner:
                second = bind_run_events(inner)
                self.assertIs(current_run_events(), second)
            self.assertIs(current_run_events(), first)
        self.assertIsNone(current_run_events())


class CrossProcessEventTests(unittest.TestCase):
    def test_torch_trial_events_reach_the_run_callback(self):
        seen: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            LeaderboardTask(_config(root, save_training_events=True)).run(
                on_event=lambda name, **kw: seen.append((name, kw))
            )
            logs = sorted(root.glob("runs/**/events.jsonl"))
        self.assertIsNone(current_run_events())  # run emitter detached with the scope
        names = [name for name, _ in seen]
        self.assertIn("training_start", names)
        self.assertIn("training_end", names)
        self.assertIn("epoch_end", names)
        self.assertTrue(all("trial_id" in payload for _, payload in seen))
        self.assertTrue(logs, "expected a durable per-trial event log")

    def test_sklearn_trial_emits_events_through_the_run_callback(self):
        seen: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            LeaderboardTask(_config(root, save_training_events=True, algorithms=("LogisticRegression",))).run(
                on_event=lambda name, **kw: seen.append((name, kw))
            )
        names = [name for name, _ in seen]
        self.assertIn("training_start", names)
        self.assertIn("training_end", names)
        starts = [payload for name, payload in seen if name == "training_start"]
        self.assertTrue(starts)
        self.assertTrue(all(payload["backend"] == "sklearn" for payload in starts))
        self.assertTrue(all("parameter_count" in payload for name, payload in seen if name == "training_end"))

    def test_nothing_is_recorded_when_the_flag_is_off(self):
        seen: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            LeaderboardTask(_config(root, save_training_events=False)).run(
                on_event=lambda name, **kw: seen.append((name, kw))
            )
            logs = sorted(root.glob("runs/**/events.jsonl"))
        self.assertEqual(logs, [])
        self.assertEqual(seen, [])


class BackendEventTests(unittest.TestCase):
    def _assert_anomaly_events(self, events: list[tuple[str, dict]], backend: str) -> None:
        names = [name for name, _ in events]
        self.assertIn("training_start", names)
        self.assertIn("training_end", names)
        start = next(payload for name, payload in events if name == "training_start")
        self.assertEqual(start["backend"], backend)
        self.assertEqual(start["task_type"], TaskType.ANOMALY_DETECTION.value)
        end = next(payload for name, payload in events if name == "training_end")
        self.assertEqual(end["backend"], backend)
        self.assertIn("parameter_count", end)

    def test_anomaly_baseline_emits_events_in_process(self):
        x, y, split = _anomaly_arrays()
        with tempfile.TemporaryDirectory() as directory:
            trial_dir = Path(directory)
            dispose = install_event_log(trial_dir)
            run_anomaly_baseline(
                AlgorithmCandidate("GlobalSTD", "none", {}, backend="anomaly_baseline"),
                x,
                y,
                split,
                sample_ids=np.arange(40),
                channels=("c1", "c2"),
                seed=0,
                workers=0,
                threshold_candidates=20,
                trial_dir=trial_dir,
                save_predictions=False,
                validation_only=False,
            )
            dispose()
            events = read_event_log(trial_dir)
        self._assert_anomaly_events(events, "anomaly_baseline")

    def test_anomaly_pytorch_emits_events_in_process(self):
        x, y, split = _anomaly_arrays()
        with tempfile.TemporaryDirectory() as directory:
            trial_dir = Path(directory)
            dispose = install_event_log(trial_dir)
            run_anomaly_pytorch(
                AlgorithmCandidate("TranAD", "standard", {}, backend="anomaly_pytorch"),
                x,
                y,
                split,
                sample_ids=np.arange(40),
                channels=("c1", "c2"),
                seed=0,
                workers=0,
                device="cpu",
                epochs=1,
                batch_size=8,
                learning_rate=1e-3,
                optimizer_name="adam",
                weight_decay=0.0,
                grad_clip_norm=None,
                threshold_candidates=20,
                trial_dir=trial_dir,
                save_checkpoint=False,
                save_predictions=False,
                validation_only=False,
            )
            dispose()
            events = read_event_log(trial_dir)
        self._assert_anomaly_events(events, "anomaly_pytorch")

    def test_baseline_single_emits_events_in_process(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(12, 5, 2)).astype(np.float32)
        y = rng.normal(size=(12, 5, 2)).astype(np.float32)
        split = DataSplit(np.arange(6), np.arange(6, 9), np.arange(9, 12))
        payload = {
            "task_type": TaskType.TIME_SERIES_FORECASTING,
            "x": x,
            "y": y,
            "forecast_target_indices": [0, 1],
            "target_weights": [1.0, 1.0],
            "final_only": False,
            "save_checkpoint": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            trial_dir = Path(directory)
            dispose = install_event_log(trial_dir)
            run_baseline_single(
                AlgorithmCandidate("LastValue", "none", {}, backend="baseline"),
                payload,
                split,
                trial_dir,
            )
            dispose()
            events = read_event_log(trial_dir)
        self.assertEqual([name for name, _ in events], ["training_start", "training_end"])
        self.assertEqual(events[0][1]["backend"], "baseline")
        self.assertEqual(events[1][1]["parameter_count"], 0)

    def test_sklearn_cv_emits_events_in_process(self):
        rng = np.random.default_rng(2)
        n = 30
        x = rng.normal(size=(n, 4)).astype(np.float32)
        y = (rng.normal(size=n) > 0).astype(int)
        fold = ValidationFold(train=np.arange(20), val=np.arange(20, 30))
        payload = {
            "task_type": TaskType.TABULAR_CLASSIFICATION,
            "x": x,
            "y": y,
            "target_weights": [1.0],
            "workers": 0,
            "save_checkpoint": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = Path(directory)
            dispose = install_event_log(fold_dir)
            run_sklearn_cv(
                AlgorithmCandidate("LogisticRegression", "standard", {}, backend="sklearn"),
                payload,
                fold,
                fold_dir,
                fold_seed=0,
            )
            dispose()
            events = read_event_log(fold_dir)
        self.assertEqual([name for name, _ in events], ["training_start", "training_end"])
        self.assertEqual(events[0][1]["backend"], "sklearn")
        self.assertIn("parameter_count", events[1][1])


if __name__ == "__main__":
    unittest.main()
