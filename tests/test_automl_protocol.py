import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from torch_like.automl import (
    DatasetSpec,
    DataSplit,
    SampleMode,
    SplitSpec,
    SplitType,
    TaskType,
    audit_split_protocol,
    prepare_csv_dataset,
    split_dataset,
    split_group_cv,
    split_rolling_origin,
)


class AutoMLProtocolAuditTests(unittest.TestCase):
    def _tabular_data(self, root: Path, *, groups: bool = False):
        frame = pd.DataFrame(
            {
                "x": np.arange(30, dtype=np.float32),
                "target": np.arange(30) % 2,
                "group": np.repeat(["a", "b", "c", "d", "e"], 6),
            }
        )
        path = root / "data.csv"
        frame.to_csv(path, index=False)
        return prepare_csv_dataset(
            DatasetSpec(
                path=str(path),
                label_columns=("target",),
                feature_columns=("x",),
                entity_column="group" if groups else None,
            )
        )

    def _forecast_data(self, root: Path):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=80, freq="h"),
                "value": np.sin(np.arange(80) / 5),
            }
        )
        path = root / "forecast.csv"
        frame.to_csv(path, index=False)
        return prepare_csv_dataset(
            DatasetSpec(
                path=str(path),
                label_columns=("value",),
                feature_columns=("value",),
                time_column="time",
                sample_mode=SampleMode.SLIDING_WINDOW,
                sequence_length=8,
                prediction_length=2,
                stride=2,
            )
        )

    def test_single_split_audit_accepts_complete_disjoint_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self._tabular_data(Path(directory))
            spec = SplitSpec(type=SplitType.RANDOM)
            split = split_dataset(data, spec, TaskType.TABULAR_CLASSIFICATION)
            audit = audit_split_protocol(data, split, spec, TaskType.TABULAR_CLASSIFICATION)
        self.assertEqual(audit.status, "passed")
        self.assertIn("sealed_test_isolated", audit.checks)

    def test_single_split_audit_rejects_test_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self._tabular_data(Path(directory))
            split = DataSplit(np.arange(20), np.arange(20, 25), np.arange(24, 30))
            with self.assertRaisesRegex(ValueError, "overlapping samples"):
                audit_split_protocol(data, split, SplitSpec(), TaskType.TABULAR_CLASSIFICATION)

    def test_group_cv_audit_proves_group_isolation_and_exact_cover(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self._tabular_data(Path(directory), groups=True)
            spec = SplitSpec(type=SplitType.GROUP_CV, cv_folds=4, test_groups=("e",))
            split = split_group_cv(data, spec)
            audit = audit_split_protocol(data, split, spec, TaskType.TABULAR_REGRESSION)
        self.assertIn("sealed_test_groups_isolated", audit.checks)
        self.assertIn("validation_exact_cover", audit.checks)

    def test_rolling_origin_audit_proves_chronology(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self._forecast_data(Path(directory))
            spec = SplitSpec(type=SplitType.ROLLING_ORIGIN, rolling_folds=3)
            split = split_rolling_origin(data, spec)
            audit = audit_split_protocol(data, split, spec, TaskType.TIME_SERIES_FORECASTING)
        self.assertEqual(audit.protocol, "rolling_origin_sealed_test")
        self.assertIn("expanding_training_windows", audit.checks)
        self.assertIn("sealed_test_chronological", audit.checks)


if __name__ == "__main__":
    unittest.main()
