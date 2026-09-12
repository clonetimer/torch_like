"""Handler-driven backend dispatch.

The point of the runner registry: a backend that ``task.py`` has never heard of
can be registered with its own runner and is dispatched to immediately, without
editing any orchestration code.
"""

import queue
import tempfile
import unittest
from pathlib import Path

import numpy as np

from torch_like.automl.backends import BackendSpec, attach_runner, get_backend, has_backend, register_backend
from torch_like.automl.data import DataSplit, ValidationFold
from torch_like.automl.sklearn_backend import AlgorithmCandidate
from torch_like.automl.specs import TaskType
from torch_like.automl.task import _cv_trial_worker, _trial_worker
from torch_like.core import Scope


def _payload(trial_dir: Path, candidate: AlgorithmCandidate, *, folds=None) -> dict:
    payload = {
        "trial_id": "trial-0000",
        "trial_dir": str(trial_dir),
        "candidate": candidate,
        "task_type": TaskType.TABULAR_REGRESSION,
        "seed": 7,
        "candidate_index": 0,
        "workers": 1,
        "device": "cpu",
        "x": np.zeros((12, 3), dtype=np.float64),
        "y": np.zeros(12, dtype=np.float64),
        "sample_ids": np.arange(12),
        "save_checkpoint": False,
        "save_predictions": False,
        "target_weights": None,
        "torch_epochs": 1,
        "torch_batch_size": 4,
        "torch_learning_rate": 0.001,
        "torch_optimizer": "adam",
        "torch_weight_decay": 0.0,
        "torch_loss": "mse",
        "torch_grad_clip_norm": None,
        "torch_lr_scheduler": "none",
        "torch_early_stopping_patience": None,
        "torch_min_delta": 0.0,
        "torch_restore_best": False,
        "torch_target_constraint": "none",
        "forecast_target_indices": [],
        "feature_columns": ("a", "b"),
        "anomaly_threshold_candidates": 50,
    }
    if folds is None:
        payload["split"] = DataSplit(np.arange(6), np.arange(6, 9), np.arange(9, 12))
    else:
        payload["folds"] = folds
        payload["cv_protocol"] = "group_cv"
    return payload


def _noop(*args, **kwargs):
    return None


class HandlerDrivenDispatchTests(unittest.TestCase):
    def test_builtin_backends_have_runners_attached(self):
        for name in ("sklearn", "baseline", "pytorch", "anomaly_baseline", "anomaly_pytorch"):
            self.assertIsNotNone(get_backend(name).runner, name)
        for name in ("sklearn", "baseline", "pytorch"):
            self.assertIsNotNone(get_backend(name).cv_runner, name)

    def test_anomaly_backends_have_distinct_runners(self):
        baseline_runner = get_backend("anomaly_baseline").runner
        pytorch_runner = get_backend("anomaly_pytorch").runner
        self.assertIsNotNone(baseline_runner)
        self.assertIsNotNone(pytorch_runner)
        self.assertIsNot(baseline_runner, pytorch_runner)

    def test_has_backend_reports_registered_names(self):
        self.assertTrue(has_backend("sklearn"))
        self.assertTrue(has_backend("anomaly_pytorch"))
        self.assertFalse(has_backend("definitely_not_a_backend"))

    def test_unknown_backend_is_dispatched_to_its_own_runner(self):
        seen: list[str] = []

        def runner(candidate, payload, split, trial_dir):
            seen.append(candidate.name)
            return ({"rmse": 1.5}, {"rmse": 2.5}, 42, np.zeros(len(split.val)), np.zeros(len(split.test)))

        with tempfile.TemporaryDirectory() as tmp, Scope("plugin") as scope:
            register_backend(BackendSpec("probe_backend", "sklearn", "baseline", runner=runner), scope=scope)
            candidate = AlgorithmCandidate("Probe", "none", {}, backend="probe_backend")
            q: queue.Queue = queue.Queue()
            _trial_worker(_payload(Path(tmp), candidate), q)
            result = q.get(timeout=10)
        self.assertEqual(seen, ["Probe"])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.validation_metrics, {"rmse": 1.5})
        self.assertEqual(result.test_metrics, {"rmse": 2.5})
        self.assertEqual(result.parameter_count, 42)

    def test_unknown_backend_cv_runner_is_dispatched(self):
        seen: list[int] = []

        def cv_runner(candidate, payload, fold, fold_dir, fold_seed):
            seen.append(fold_seed)
            return (None, {"rmse": 3.0}, 7, np.zeros(len(fold.val)), [])

        with tempfile.TemporaryDirectory() as tmp, Scope("plugin") as scope:
            register_backend(BackendSpec("probe_cv", "sklearn", "baseline", cv_runner=cv_runner), scope=scope)
            candidate = AlgorithmCandidate("ProbeCV", "none", {}, backend="probe_cv")
            folds = [ValidationFold(np.arange(6), np.arange(6, 9)), ValidationFold(np.arange(6), np.arange(6, 9))]
            q: queue.Queue = queue.Queue()
            _cv_trial_worker(_payload(Path(tmp), candidate, folds=folds), q)
            result = q.get(timeout=10)
        self.assertEqual(seen, [7, 8])  # payload seed 7 + fold index
        self.assertEqual(result.parameter_count, 7)
        self.assertAlmostEqual(result.validation_metrics["rmse"], 3.0)
        self.assertAlmostEqual(result.validation_metrics["rmse_std"], 0.0)

    def test_backend_without_runner_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp, Scope("plugin") as scope:
            register_backend(BackendSpec("no_runner_backend", "sklearn", "baseline"), scope=scope)
            candidate = AlgorithmCandidate("NoRunner", "none", {}, backend="no_runner_backend")
            q: queue.Queue = queue.Queue()
            _trial_worker(_payload(Path(tmp), candidate), q)
            result = q.get(timeout=10)
        self.assertEqual(result.status, "failed")
        self.assertIn("no single-split runner registered", result.error)

    def test_attach_runner_is_revoked_by_scope_dispose(self):
        original = get_backend("sklearn")
        self.assertIsNotNone(original.runner)
        with Scope("swap") as scope:
            attach_runner("sklearn", runner=_noop, scope=scope)
            self.assertIs(get_backend("sklearn").runner, _noop)
        self.assertIs(get_backend("sklearn").runner, original.runner)
        self.assertIsNot(get_backend("sklearn").runner, _noop)


if __name__ == "__main__":
    unittest.main()
