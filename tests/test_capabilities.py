"""Config-driven capability assembly.

A run mounts only the backends its config allows; everything else is suppressed
for the duration of the run's scope and restored on the way out -- including
when the run fails.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from torch_like.automl.backends import has_backend
from torch_like.automl.candidate_registry import list_candidate_specs
from torch_like.automl.capabilities import assemble_capabilities
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

_PYTORCH_CANDIDATE = "FTTransformer"


def _write_csv(path: Path) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40)})
    frame["label"] = (frame["f1"] + frame["f2"] > 0).astype(int)
    frame.to_csv(path, index=False)


def _config(root: Path, *, backends=None, max_trials: int = 2) -> LeaderboardConfig:
    return LeaderboardConfig(
        task_id="capability-probe",
        dataset=DatasetSpec(path=str(root / "data.csv"), label_columns=("label",)),
        task=TaskSpec(type="tabular_classification"),
        split=SplitSpec(type=SplitType.STRATIFIED),
        resources=ResourceSpec(timeout_per_trial_minutes=1, max_total_minutes=2),
        search=SearchSpec(
            strategy="none",
            max_trials=max_trials,
            preprocessing=("standard",),
            backends=backends,
        ),
        artifacts=ArtifactSpec(output_dir=str(root / "runs")),
    )


def _pytorch_candidates_present() -> bool:
    return any(
        spec.name == _PYTORCH_CANDIDATE
        for spec in list_candidate_specs(
            TaskType.TABULAR_CLASSIFICATION, is_sequence=False, equal_sequence_lengths=True
        )
    )


class CapabilityAssemblyTests(unittest.TestCase):
    def test_unrestricted_config_mounts_an_empty_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            with assemble_capabilities(_config(root)) as scope:
                self.assertTrue(scope.active)
                self.assertTrue(has_backend("pytorch"))
                self.assertTrue(_pytorch_candidates_present())
        self.assertTrue(has_backend("pytorch"))
        self.assertTrue(_pytorch_candidates_present())

    def test_allowlist_suppresses_other_backends_and_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            with assemble_capabilities(_config(root, backends=("sklearn",))):
                self.assertTrue(has_backend("sklearn"))
                self.assertFalse(has_backend("pytorch"))
                self.assertFalse(has_backend("baseline"))
                self.assertFalse(_pytorch_candidates_present())
            # Disposing the scope restores the registry exactly.
            self.assertTrue(has_backend("pytorch"))
            self.assertTrue(has_backend("baseline"))
            self.assertTrue(_pytorch_candidates_present())

    def test_unknown_backend_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            config = _config(root, backends=("sklearn", "definitely_not_a_backend"))
            with self.assertRaisesRegex(ValueError, "Unknown backend"):
                assemble_capabilities(config)
        self.assertTrue(has_backend("pytorch"))

    def test_suppression_is_restored_even_when_the_body_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            with self.assertRaises(RuntimeError):
                with assemble_capabilities(_config(root, backends=("sklearn",))):
                    self.assertFalse(has_backend("pytorch"))
                    raise RuntimeError("boom")
            self.assertTrue(has_backend("pytorch"))
            self.assertTrue(_pytorch_candidates_present())

    def test_search_spec_normalizes_backends_to_a_tuple(self):
        self.assertIsNone(SearchSpec().backends)
        self.assertEqual(SearchSpec(backends=["sklearn"]).backends, ("sklearn",))

    def test_run_restricted_to_sklearn_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            result = LeaderboardTask(_config(root, backends=("sklearn",))).run()
            self.assertFalse(result.leaderboard.empty)
            self.assertEqual(set(result.leaderboard["backend"]), {"sklearn"})
            # The run's scope is disposed before run() returns.
            self.assertTrue(has_backend("pytorch"))
            self.assertTrue(_pytorch_candidates_present())

    def test_scope_can_be_supplied_by_the_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(root / "data.csv")
            with Scope("outer") as outer:
                returned = assemble_capabilities(_config(root, backends=("sklearn",)), outer)
                self.assertIs(returned, outer)
                self.assertFalse(has_backend("pytorch"))
            self.assertTrue(has_backend("pytorch"))


if __name__ == "__main__":
    unittest.main()
