import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
from torch.utils.data import RandomSampler, SequentialSampler

from torch_like.data import (
    DetectionDataConfig,
    ForecastDataConfig,
    RULDataConfig,
    XJTUSYDataset,
    XJTUSYFileWindowDataset,
    create_detection_loaders,
    create_forecast_loaders,
    load_detection_data,
    load_forecast_data,
    time_features,
)
from torch_like.data.loader import data_provider


class TaskDataApiTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.tmp_path = Path(self._temporary_directory.name)

    def test_legacy_data_provider_keeps_evaluation_samples(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ett.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=120, freq="h"),
                    "value": np.arange(120, dtype=np.float32),
                }
            ).to_csv(path, index=False)
            args = SimpleNamespace(
                root_path=tmp_dir,
                data_path=path.name,
                seq_len=8,
                label_len=4,
                pred_len=2,
                freq="h",
                batch_size=7,
                num_workers=0,
            )
            _, train_loader = data_provider(args, "train")
            _, val_loader = data_provider(args, "val")
            _, test_loader = data_provider(args, "test")
            self.assertIsInstance(train_loader.sampler, RandomSampler)
            self.assertIsInstance(val_loader.sampler, SequentialSampler)
            self.assertIsInstance(test_loader.sampler, SequentialSampler)
            self.assertFalse(val_loader.drop_last)
            self.assertFalse(test_loader.drop_last)

    def test_time_features_follow_frequency_granularity(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="h")
        self.assertEqual(time_features(dates, freq="h").shape, (4, 4))
        self.assertEqual(time_features(dates, freq="min").shape, (4, 5))
        self.assertEqual(time_features(dates, freq="MS").shape, (4, 1))
        self.assertEqual(time_features(dates, freq="YE").shape, (4, 0))

    def test_forecast_config_rejects_invalid_ranges(self):
        with self.assertRaises(ValueError):
            ForecastDataConfig(seq_len=0)
        with self.assertRaises(ValueError):
            ForecastDataConfig(seq_len=8, label_len=9)
        with self.assertRaises(ValueError):
            ForecastDataConfig(train_ratio=0.8, val_ratio=0.2)
        with self.assertRaises(ValueError):
            ForecastDataConfig(val_ratio=0.0)

    def test_forecast_data_rejects_short_or_invalid_feature_splits(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            load_forecast_data(
                ForecastDataConfig(
                    dataset_type="numpy",
                    numpy_data=np.zeros((20, 3), dtype=np.float32),
                    seq_len=8,
                    label_len=4,
                    pred_len=2,
                )
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_forecast_data(
                ForecastDataConfig(
                    dataset_type="numpy",
                    numpy_data=np.zeros((60, 3), dtype=np.float32),
                    seq_len=8,
                    label_len=4,
                    pred_len=2,
                    target_indices=[0, 1],
                    exogenous_indices=[1, 2],
                )
            )

    def test_graph_feature_splits_use_feature_axis(self):
        data = np.random.rand(60, 3, 2).astype(np.float32)
        bundle = load_forecast_data(
            ForecastDataConfig(
                dataset_type="graph_npy",
                numpy_data=data,
                adjacency=np.eye(3, dtype=np.float32),
                seq_len=8,
                label_len=4,
                pred_len=2,
                target_indices=[1],
                exogenous_indices=[0],
            )
        )
        sample = bundle.train_set[0]
        self.assertEqual(sample["x_enc"].shape, (8, 6))
        self.assertEqual(sample["x_target_enc"].shape, (8, 3))
        self.assertEqual(sample["x_exog_enc"].shape, (8, 3))
        self.assertEqual(sample["x_dec_target"].shape, (6, 3))
        self.assertEqual(bundle.extra["feature_dim"], 2)
        self.assertEqual(bundle.extra["model_num_nodes"], 6)
        self.assertEqual(np.asarray(bundle.extra["adjacency"]).shape, (6, 6))
        self.assertEqual(np.asarray(bundle.extra["node_adjacency"]).shape, (3, 3))

    def test_graph_data_rejects_wrong_adjacency_shape(self):
        with self.assertRaisesRegex(ValueError, "adjacency must have shape"):
            load_forecast_data(
                ForecastDataConfig(
                    dataset_type="graph_npy",
                    numpy_data=np.zeros((60, 3), dtype=np.float32),
                    adjacency=np.eye(2, dtype=np.float32),
                    seq_len=8,
                    label_len=4,
                    pred_len=2,
                )
            )

    def test_small_valid_forecast_split_keeps_training_batch(self):
        _, train_loader, _, _ = create_forecast_loaders(
            ForecastDataConfig(
                dataset_type="numpy",
                numpy_data=np.zeros((45, 2), dtype=np.float32),
                seq_len=10,
                label_len=5,
                pred_len=5,
                batch_size=32,
            )
        )
        self.assertEqual(len(train_loader), 1)

    def test_detection_config_rejects_invalid_stride(self):
        with self.assertRaises(ValueError):
            DetectionDataConfig(stride=0)

    def test_public_forecast_data_api(self):
        data = np.random.rand(120, 3).astype(np.float32)
        bundle = load_forecast_data(
            ForecastDataConfig(
                dataset_type="numpy",
                numpy_data=data,
                seq_len=12,
                label_len=6,
                pred_len=3,
            )
        )
        self.assertGreater(len(bundle.train_set), 0)

    def test_public_detection_data_api(self):
        tmp = self.tmp_path / "runner_csv"
        tmp.mkdir(exist_ok=True)
        train_path = tmp / "task_data_train.csv"
        test_path = tmp / "task_data_test.csv"
        df = pd.DataFrame(
            {
                "timestamp": [f"2000-01-01 00:00:{i:02d}" for i in range(8)],
                "channel_1": np.random.rand(8),
                "channel_2": np.random.rand(8),
                "is_anomaly_channel_1": [0] * 7 + [1],
            }
        )
        df.to_csv(train_path, index=False)
        df.to_csv(test_path, index=False)
        bundle = load_detection_data(
            DetectionDataConfig(
                train_path=str(train_path),
                test_path=str(test_path),
                max_train_rows=8,
                max_test_rows=8,
            )
        )
        self.assertEqual(bundle.train_data.shape[1], 2)
        self.assertEqual(bundle.test_labels[-1], 1)

    def test_public_detection_loader_cuda_kwargs(self):
        tmp = self.tmp_path / "runner_csv"
        tmp.mkdir(exist_ok=True)
        train_path = tmp / "task_data_loader_train.csv"
        test_path = tmp / "task_data_loader_test.csv"
        df = pd.DataFrame(
            {
                "timestamp": [f"2000-01-01 00:00:{i:02d}" for i in range(8)],
                "channel_1": np.random.rand(8),
                "channel_2": np.random.rand(8),
            }
        )
        df.to_csv(train_path, index=False)
        df.to_csv(test_path, index=False)
        _, train_loader, _ = create_detection_loaders(
            DetectionDataConfig(
                train_path=str(train_path),
                test_path=str(test_path),
                seq_len=4,
                batch_size=2,
            ),
            device="cuda",
        )
        self.assertTrue(getattr(train_loader, "pin_memory", False))

    def test_public_forecast_loader_cuda_kwargs(self):
        data = np.random.rand(64, 3).astype(np.float32)
        _, train_loader, _, _ = create_forecast_loaders(
            ForecastDataConfig(
                dataset_type="numpy",
                numpy_data=data,
                seq_len=12,
                label_len=6,
                pred_len=3,
            ),
            device="cuda",
        )
        self.assertTrue(getattr(train_loader, "pin_memory", False))

    def test_public_rul_config_defaults(self):
        cfg = RULDataConfig(root_dir="dummy_root")
        self.assertEqual(cfg.condition, "35Hz12kN")

    def test_xjtu_file_window_dataset_is_bounded_and_labels_lifecycle(self):
        condition = self.tmp_path / "35Hz12kN"
        for bearing_name in ("Bearing1_1", "Bearing1_2"):
            bearing = condition / bearing_name
            bearing.mkdir(parents=True)
            for minute in range(1, 4):
                pd.DataFrame(
                    {
                        "Horizontal_vibration_signals": np.arange(16, dtype=np.float32) + minute,
                        "Vertical_vibration_signals": np.arange(16, dtype=np.float32) - minute,
                    }
                ).to_csv(bearing / f"{minute}.csv", index=False)
        dataset = XJTUSYFileWindowDataset(
            str(self.tmp_path),
            condition="35Hz12kN",
            seq_len=8,
            stride=4,
            max_files_per_bearing=2,
            max_windows_per_file=1,
        )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.num_bearings(), 2)
        self.assertEqual(dataset.num_channels, 2)
        first_x, first_y = dataset[0]
        last_x, last_y = dataset[1]
        self.assertEqual(first_x.shape, (8, 2))
        self.assertAlmostEqual(first_y.item(), 2 / 3)
        self.assertEqual(last_x.shape, (8, 2))
        self.assertEqual(last_y.item(), 0.0)
        dataset[0]
        self.assertEqual(len(dataset._signal_cache), 2)

    def test_xjtu_continuous_window_crosses_files_without_full_life_cache(self):
        bearing = self.tmp_path / "35Hz12kN" / "Bearing1_1"
        bearing.mkdir(parents=True)
        for minute in (1, 2):
            pd.DataFrame(
                {
                    "Horizontal_vibration_signals": np.arange(16, dtype=np.float32) + 100 * minute,
                    "Vertical_vibration_signals": np.arange(16, dtype=np.float32) - 100 * minute,
                }
            ).to_csv(bearing / f"{minute}.csv", index=False)
        dataset = XJTUSYDataset(
            str(self.tmp_path),
            condition="35Hz12kN",
            seq_len=12,
            stride=10,
        )
        window, rul = dataset[1]
        self.assertEqual(window.shape, (12, 2))
        np.testing.assert_array_equal(window[:6, 0].numpy(), np.arange(10, 16) + 100)
        np.testing.assert_array_equal(window[6:, 0].numpy(), np.arange(6) + 200)
        self.assertAlmostEqual(rul.item(), (2 - 22 / 16) / 2)
        self.assertLessEqual(len(dataset._signal_cache), 8)


if __name__ == "__main__":
    unittest.main()
