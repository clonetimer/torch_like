import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from torch_like.automl import (
    ArtifactSpec,
    DatasetFormat,
    DatasetSpec,
    DataSplit,
    GroupCVSplit,
    LeaderboardConfig,
    LeaderboardTask,
    ResourceSpec,
    RollingOriginSplit,
    SampleMode,
    SearchSpec,
    SearchStrategy,
    SplitSpec,
    SplitType,
    TaskSpec,
    TaskType,
    load_config,
    prepare_csv_dataset,
    prepare_dataset,
    split_dataset,
    split_group_cv,
    split_rolling_origin,
)
from torch_like.automl.anomaly_backend import run_anomaly_candidate
from torch_like.automl.anomaly_catalog import discover_detection_model_grids
from torch_like.automl.anomaly_metrics import evaluate_anomaly_scores, select_anomaly_threshold
from torch_like.automl.esa_data import _channel_labels, _windows
from torch_like.automl.sklearn_backend import (
    AlgorithmCandidate,
    SequenceFeatureTransformer,
    build_candidates,
    build_pipeline,
    evaluate_predictions,
)
from torch_like.automl.torch_backend import run_torch_candidate
from torch_like.models.registry import list_models


def _write_esa_archive(root: Path) -> Path:
    mission = root / "source" / "ESA-Mission1"
    channels_dir = mission / "channels"
    channels_dir.mkdir(parents=True)
    timeline = pd.date_range("2020-01-01", "2020-01-06 23:59:30", freq="30s")
    values = np.sin(np.arange(len(timeline)) / 50).astype(np.float32)
    for index, channel in enumerate(("channel_41", "channel_42")):
        channel_values = values + index / 10
        if channel == "channel_41":
            channel_values[(timeline >= "2020-01-03 12:00") & (timeline <= "2020-01-03 18:00")] += 8
        else:
            channel_values[(timeline >= "2020-01-05 12:00") & (timeline <= "2020-01-05 18:00")] += 8
        pd.DataFrame({channel: channel_values}, index=timeline).to_pickle(
            channels_dir / f"{channel}.zip", compression="zip"
        )
    pd.DataFrame(
        {
            "Channel": ("channel_41", "channel_42"),
            "Subsystem": ("subsystem_5", "subsystem_5"),
            "Physical Unit": ("physical_unit_4", "physical_unit_4"),
            "Group": (8, 8),
            "Target": ("YES", "YES"),
        }
    ).to_csv(mission / "channels.csv", index=False)
    pd.DataFrame(
        {
            "ID": ("id_1", "id_2"),
            "Class": ("class_1", "class_2"),
            "Subclass": ("subclass_1", "subclass_1"),
            "Category": ("Anomaly", "Anomaly"),
            "Dimensionality": ("Univariate", "Univariate"),
            "Locality": ("Global", "Global"),
            "Length": ("Subsequence", "Subsequence"),
        }
    ).to_csv(mission / "anomaly_types.csv", index=False)
    pd.DataFrame(
        {
            "ID": ("id_1", "id_2"),
            "Channel": ("channel_41", "channel_42"),
            "StartTime": ("2020-01-03T12:00:00Z", "2020-01-05T12:00:00Z"),
            "EndTime": ("2020-01-03T18:00:00Z", "2020-01-05T18:00:00Z"),
        }
    ).to_csv(mission / "labels.csv", index=False)
    archive = root / "ESA-Mission1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        for path in mission.rglob("*"):
            if path.is_file():
                outer.write(path, path.relative_to(mission.parent).as_posix())
    return archive


def _esa_spec(root: Path) -> DatasetSpec:
    return DatasetSpec(
        path=str(_write_esa_archive(root)),
        format="esa_adb",
        label_columns=("channel_41", "channel_42"),
        feature_columns=("channel_41", "channel_42"),
        sample_mode="sliding_window",
        sequence_length=16,
        stride=4,
        split_time_ranges={
            "train": ("2020-01-01", "2020-01-02 23:59:30"),
            "val": ("2020-01-03", "2020-01-04 23:59:30"),
            "test": ("2020-01-05", "2020-01-06 23:59:30"),
        },
        anomaly_categories=("Anomaly",),
        cache_dir=str(root / "cache"),
    )


class AutoMLDataTests(unittest.TestCase):
    def test_esa_short_events_survive_resampling_and_window_stride(self):
        index = pd.date_range("2020-01-01", periods=120, freq="30s")
        labels = pd.DataFrame(
            {
                "Channel": ["channel_1"],
                "Category": ["Anomaly"],
                "StartTime": [pd.Timestamp("2020-01-01 00:01:45")],
                "EndTime": [pd.Timestamp("2020-01-01 00:01:49")],
            }
        )
        resampled = _channel_labels(index, "channel_1", labels, {"Anomaly"}, pd.Timedelta(seconds=30))
        self.assertEqual(resampled.sum(), 1)
        self.assertEqual(index[np.flatnonzero(resampled)[0]], pd.Timestamp("2020-01-01 00:01:30"))

        values = np.zeros((len(index), 1), dtype=np.float32)
        _, window_labels, _ = _windows(values, resampled[:, None], index.to_numpy(), sequence_length=8, stride=6)
        self.assertEqual(window_labels.sum(), 1)

    def test_esa_archive_adapter_resamples_windows_and_channel_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            data = prepare_dataset(_esa_spec(Path(directory)))
            split = split_dataset(data, SplitSpec(type="predefined"), TaskType.ANOMALY_DETECTION)
            self.assertEqual(data.x.shape[1:], (16, 2))
            self.assertEqual(data.y.shape[1:], (2,))
            self.assertEqual(len(split.train), len(split.val))
            self.assertEqual(len(split.val), len(split.test))
            self.assertGreater(data.y[split.val, 0].sum(), 0)
            self.assertGreater(data.y[split.test, 1].sum(), 0)

    def test_resource_device_accepts_auto_and_rejects_invalid_values(self):
        self.assertEqual(ResourceSpec(device="auto").device, "auto")
        with self.assertRaisesRegex(ValueError, "valid PyTorch device"):
            ResourceSpec(device="not-a-device")

    def test_xjtu_sy_folder_adapter_generates_bounded_rul_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = root / "35Hz12kN"
            for bearing_index in range(5):
                bearing = condition / f"Bearing1_{bearing_index + 1}"
                bearing.mkdir(parents=True)
                for minute in range(1, 4):
                    pd.DataFrame(
                        {
                            "Horizontal_vibration_signals": np.arange(16) + minute,
                            "Vertical_vibration_signals": np.arange(16) - minute,
                        }
                    ).to_csv(bearing / f"{minute}.csv", index=False)
            spec = DatasetSpec(
                path=str(root),
                format=DatasetFormat.XJTU_SY,
                label_columns=("rul",),
                sample_mode="sliding_window",
                sequence_length=8,
                stride=4,
                condition="35Hz12kN",
                max_windows_per_file=1,
            )
            data = prepare_dataset(spec)
            split = split_dataset(data, SplitSpec(type="group"), TaskType.RUL_REGRESSION)
            self.assertEqual(data.x.shape, (15,))
            self.assertEqual(data.x[0].shape, (8, 2))
            np.testing.assert_allclose(data.y[:3], [2 / 3, 1 / 3, 0], atol=1e-6)
            self.assertEqual(data.profile.entities, 5)
            self.assertFalse(set(data.groups[split.train]) & set(data.groups[split.test]))

    def test_xjtu_sy_requires_group_split(self):
        with self.assertRaisesRegex(ValueError, "split.type='group'"):
            LeaderboardConfig(
                task_id="xjtu-leak",
                dataset=DatasetSpec(
                    path="dataset-root",
                    format="xjtu_sy",
                    label_columns=("rul",),
                    sample_mode="sliding_window",
                    sequence_length=8,
                    stride=4,
                ),
                task=TaskSpec(type="rul_regression"),
            )

    def test_sliding_window_requires_entity_length_and_stride(self):
        with self.assertRaisesRegex(ValueError, "entity_column"):
            DatasetSpec(
                path="data.csv",
                label_columns=("rul",),
                sample_mode=SampleMode.SLIDING_WINDOW,
                sequence_length=8,
                stride=4,
            )

    def test_overlapping_windows_cannot_use_random_split(self):
        with self.assertRaisesRegex(ValueError, "would leak raw rows"):
            LeaderboardConfig(
                task_id="leaky-windows",
                dataset=DatasetSpec(
                    path="data.csv",
                    label_columns=("rul",),
                    entity_column="entity",
                    sample_mode="sliding_window",
                    sequence_length=16,
                    stride=8,
                ),
                task=TaskSpec(type="rul_regression"),
                split=SplitSpec(type="random"),
            )

    def test_external_labels_are_joined_by_unique_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"sample_id": ["a", "b"], "f1": [1.0, 2.0]}).to_csv(root / "data.csv", index=False)
            pd.DataFrame({"sample_id": ["a", "b"], "label": [0, 1]}).to_csv(root / "labels.csv", index=False)
            data = prepare_csv_dataset(
                DatasetSpec(
                    path=str(root / "data.csv"),
                    labels_path=str(root / "labels.csv"),
                    label_columns=("label",),
                )
            )
            np.testing.assert_array_equal(data.y, [0, 1])
            self.assertEqual(data.profile.features, 1)

    def test_full_sequence_and_sliding_window_are_distinct_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.csv"
            rows = []
            for entity in ("a", "b"):
                for step in range(6):
                    rows.append({"entity": entity, "time": step, "sensor": step, "rul": 10 - step})
            pd.DataFrame(rows).to_csv(path, index=False)
            full = prepare_csv_dataset(
                DatasetSpec(
                    path=str(path),
                    label_columns=("rul",),
                    entity_column="entity",
                    time_column="time",
                    sample_mode="full_sequence",
                )
            )
            windows = prepare_csv_dataset(
                DatasetSpec(
                    path=str(path),
                    label_columns=("rul",),
                    entity_column="entity",
                    time_column="time",
                    sample_mode="sliding_window",
                    sequence_length=4,
                    stride=2,
                )
            )
            self.assertEqual(len(full.x), 2)
            self.assertEqual(full.sequence_lengths, (6, 6))
            self.assertEqual(len(windows.x), 4)
            self.assertEqual(windows.sequence_lengths, (4, 4, 4, 4))
            np.testing.assert_array_equal(windows.y, [7, 5, 7, 5])

    def test_group_split_keeps_entities_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.csv"
            frame = pd.DataFrame(
                {
                    "entity": np.repeat([f"g{index}" for index in range(12)], 3),
                    "feature": np.arange(36),
                    "label": np.tile([0, 1, 0], 12),
                }
            )
            frame.to_csv(path, index=False)
            data = prepare_csv_dataset(
                DatasetSpec(
                    path=str(path),
                    label_columns=("label",),
                    entity_column="entity",
                )
            )
            split = split_dataset(data, SplitSpec(type="group"), TaskType.TABULAR_CLASSIFICATION)
            train_groups = set(data.groups[split.train])
            val_groups = set(data.groups[split.val])
            test_groups = set(data.groups[split.test])
            self.assertFalse(train_groups & val_groups)
            self.assertFalse(train_groups & test_groups)
            self.assertFalse(val_groups & test_groups)

    def test_group_cv_reserves_test_groups_and_rotates_validation_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.csv"
            frame = pd.DataFrame(
                {
                    "entity": np.repeat([f"g{index}" for index in range(5)], 4),
                    "feature": np.arange(20),
                    "target": np.linspace(1.0, 0.0, 20),
                }
            )
            frame.to_csv(path, index=False)
            data = prepare_csv_dataset(DatasetSpec(path=str(path), label_columns=("target",), entity_column="entity"))
            split = split_group_cv(
                data,
                SplitSpec(type="group_cv", cv_folds=4, test_groups=("g4",)),
            )
            self.assertIsInstance(split, GroupCVSplit)
            self.assertEqual(set(data.groups[split.test]), {"g4"})
            validation_indices = np.concatenate([fold.val for fold in split.folds])
            np.testing.assert_array_equal(np.sort(validation_indices), split.development)
            for fold in split.folds:
                self.assertFalse(set(fold.train) & set(fold.val))
                self.assertFalse(set(fold.train) & set(split.test))
                self.assertFalse(set(fold.val) & set(split.test))

    def test_forecasting_windows_and_rolling_origin_keep_future_test_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.csv"
            steps = np.arange(160)
            pd.DataFrame(
                {
                    "time": steps,
                    "target": np.sin(steps / 8),
                    "exogenous": np.cos(steps / 11),
                }
            ).to_csv(path, index=False)
            data = prepare_csv_dataset(
                DatasetSpec(
                    path=str(path),
                    label_columns=("target",),
                    feature_columns=("target", "exogenous"),
                    time_column="time",
                    sample_mode="sliding_window",
                    sequence_length=24,
                    prediction_length=8,
                    stride=8,
                )
            )
            split = split_rolling_origin(data, SplitSpec(type="rolling_origin", rolling_folds=3))
            self.assertIsInstance(split, RollingOriginSplit)
            self.assertEqual(data.x[0].shape, (24, 2))
            self.assertEqual(data.y[0].shape, (8, 1))
            self.assertGreater(data.timestamps[split.test].min(), data.timestamps[split.development].max())
            for fold in split.folds:
                self.assertLess(data.timestamps[fold.train].max(), data.timestamps[fold.val].min())
                self.assertFalse(set(fold.val) & set(split.test))


class AutoMLBackendTests(unittest.TestCase):
    @staticmethod
    def _anomaly_arrays():
        rng = np.random.default_rng(13)
        x = rng.normal(size=(40, 8, 2)).astype(np.float32)
        y = np.zeros((40, 2), dtype=np.uint8)
        y[22:27, 0] = 1
        y[32:37, 1] = 1
        x[22:27, -1, 0] += 6
        x[32:37, -1, 1] += 6
        split = DataSplit(np.arange(20), np.arange(20, 30), np.arange(30, 40))
        return x, y, split

    def test_anomaly_baselines_execute_directly_with_artifacts(self):
        x, y, split = self._anomaly_arrays()
        candidates = (
            AlgorithmCandidate("GlobalSTD", "none", {}, backend="anomaly_baseline"),
            AlgorithmCandidate(
                "IsolationForest",
                "standard",
                {"n_estimators": 5, "max_samples": 20},
                backend="anomaly_baseline",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, candidate in enumerate(candidates):
                trial_dir = root / str(index)
                trial_dir.mkdir()
                result = run_anomaly_candidate(
                    candidate,
                    x,
                    y,
                    split,
                    sample_ids=np.arange(len(x)),
                    channels=("a", "b"),
                    seed=13,
                    workers=1,
                    device="cpu",
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.001,
                    optimizer_name="adam",
                    weight_decay=0,
                    grad_clip_norm=None,
                    threshold_candidates=10,
                    trial_dir=trial_dir,
                    save_checkpoint=True,
                    save_predictions=True,
                    validation_only=False,
                )
                self.assertIn("event_f0_5_corrected", result.validation_metrics)
                self.assertIn("point_f1", result.test_metrics)
                self.assertTrue((trial_dir / "artifact_manifest.json").is_file())
                self.assertTrue((trial_dir / "predictions.csv").is_file())

    def test_anomaly_pytorch_models_execute_directly(self):
        x, y, split = self._anomaly_arrays()
        candidates = tuple(
            AlgorithmCandidate(name, "standard", parameter_grid[0], backend="anomaly_pytorch")
            for name, parameter_grid in discover_detection_model_grids(include_experimental=True).items()
        )
        self.assertEqual({candidate.name for candidate in candidates}, set(list_models("detection")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, candidate in enumerate(candidates):
                trial_dir = root / str(index)
                trial_dir.mkdir()
                result = run_anomaly_candidate(
                    candidate,
                    x,
                    y,
                    split,
                    sample_ids=np.arange(len(x)),
                    channels=("a", "b"),
                    seed=13,
                    workers=1,
                    device="cpu",
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.001,
                    optimizer_name="adamw",
                    weight_decay=0.001,
                    grad_clip_norm=1.0,
                    threshold_candidates=10,
                    trial_dir=trial_dir,
                    save_checkpoint=True,
                    save_predictions=False,
                    validation_only=False,
                )
                self.assertTrue(np.isfinite(result.validation_scores).all())
                self.assertGreater(result.parameter_count, 0)
                self.assertTrue((trial_dir / "checkpoint.pt").is_file())

    def test_anomaly_candidate_pool_prioritizes_registry_coverage(self):
        default_models = {
            name for name, metadata in list_models("detection").items() if metadata.get("automl_default", True)
        }
        expected_names = {"GlobalSTD", "IsolationForest"} | default_models
        search = SearchSpec(
            strategy="grid",
            max_trials=len(expected_names),
            preprocessing=("standard", "robust"),
        )
        candidates = build_candidates(
            TaskType.ANOMALY_DETECTION,
            search,
            is_sequence=True,
            equal_sequence_lengths=True,
            seed=7,
        )
        self.assertEqual(
            {candidate.name for candidate in candidates},
            expected_names,
        )

    def test_anomaly_threshold_is_selected_from_validation_scores(self):
        truth = np.zeros((100, 2), dtype=np.uint8)
        truth[30:40, 0] = 1
        scores = np.zeros((100, 2), dtype=np.float64)
        scores[30:40, 0] = 5
        threshold, metrics = select_anomaly_threshold(truth, scores, candidates=20)
        self.assertGreater(threshold, 0)
        self.assertEqual(metrics["event_f0_5_corrected"], 1.0)
        self.assertEqual(evaluate_anomaly_scores(truth, scores, threshold)["point_f1"], 1.0)

    def test_vibration_statistics_extract_time_and_frequency_features(self):
        positions = np.linspace(0, 4 * np.pi, 64)
        sequences = np.stack(
            [np.stack((np.sin(positions + phase), np.cos(positions + phase)), axis=-1) for phase in (0.0, 0.2, 0.4)]
        )
        transformer = SequenceFeatureTransformer("vibration_statistics").fit(sequences)
        features = transformer.transform(sequences)
        self.assertEqual(features.shape, (3, 44))
        self.assertTrue(np.isfinite(features).all())

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "lightgbm is not installed")
    def test_lightgbm_rul_candidates_cover_all_sequence_feature_strategies(self):
        candidates = build_candidates(
            TaskType.RUL_REGRESSION,
            SearchSpec(
                strategy="grid",
                max_trials=6,
                preprocessing=("none",),
                algorithms=("LightGBMRegressor",),
            ),
            is_sequence=True,
            equal_sequence_lengths=True,
            seed=42,
        )
        self.assertEqual(len(candidates), 6)
        self.assertEqual(
            {candidate.sequence_features for candidate in candidates},
            {"statistics", "vibration_statistics", "flatten"},
        )

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "lightgbm is not installed")
    def test_lightgbm_regression_pipeline(self):
        rng = np.random.default_rng(3)
        sequences = rng.normal(size=(40, 32, 2))
        targets = sequences[:, :, 0].std(axis=1) + sequences[:, :, 1].mean(axis=1)
        pipeline = build_pipeline(
            AlgorithmCandidate(
                "LightGBMRegressor",
                "none",
                {"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 7},
                sequence_features="vibration_statistics",
            ),
            TaskType.RUL_REGRESSION,
            seed=3,
            workers=1,
        )
        pipeline.fit(sequences, targets)
        prediction = pipeline.predict(sequences)
        self.assertEqual(prediction.shape, (40,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_multilabel_pipeline_and_metrics(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(40, 4))
        y = np.stack([x[:, 0] > 0, x[:, 1] > 0], axis=1).astype(int)
        pipeline = build_pipeline(
            AlgorithmCandidate("LogisticRegression", "standard", {"C": 1.0}),
            TaskType.MULTILABEL_CLASSIFICATION,
            seed=4,
            workers=1,
        )
        pipeline.fit(x, y)
        prediction = pipeline.predict(x)
        metrics = evaluate_predictions(TaskType.MULTILABEL_CLASSIFICATION, y, prediction)
        self.assertIn("f1_macro", metrics)
        self.assertIn("f1_micro", metrics)

    def test_multi_target_pipeline_and_metrics(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(40, 4))
        y = np.stack([2 * x[:, 0], -3 * x[:, 1]], axis=1)
        pipeline = build_pipeline(
            AlgorithmCandidate("Ridge", "standard", {"alpha": 0.1}),
            TaskType.MULTI_TARGET_REGRESSION,
            seed=5,
            workers=1,
        )
        pipeline.fit(x, y)
        metrics = evaluate_predictions(TaskType.MULTI_TARGET_REGRESSION, y, pipeline.predict(x))
        self.assertIn("macro_nrmse", metrics)
        self.assertIn("target_1_rmse", metrics)

    def test_forecasting_candidates_and_horizon_metrics(self):
        candidates = build_candidates(
            TaskType.TIME_SERIES_FORECASTING,
            SearchSpec(
                strategy="grid",
                max_trials=3,
                preprocessing=("standard",),
                algorithms=("DLinear", "PatchTST", "TSMixer"),
            ),
            is_sequence=True,
            equal_sequence_lengths=True,
            seed=5,
        )
        self.assertEqual({candidate.name for candidate in candidates}, {"DLinear", "PatchTST", "TSMixer"})
        y_true = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
        metrics = evaluate_predictions(TaskType.TIME_SERIES_FORECASTING, y_true, y_true + 1)
        self.assertIn("smape", metrics)
        self.assertIn("horizon_4_rmse", metrics)

    def test_torch_backend_contract_in_process(self):
        rng = np.random.default_rng(6)
        sequences = np.empty(12, dtype=object)
        sequences[:] = [rng.normal(size=(8, 2)).astype(np.float32) for _ in range(12)]
        targets = np.linspace(100, 50, 12, dtype=np.float32)
        split = DataSplit(train=np.arange(0, 8), val=np.arange(8, 10), test=np.arange(10, 12))
        with tempfile.TemporaryDirectory() as directory:
            trial_dir = Path(directory)
            validation, test, parameters, _, prediction = run_torch_candidate(
                AlgorithmCandidate(
                    "THGNN",
                    "standard",
                    {"d_model": 16, "graph_depth": 1},
                    backend="pytorch",
                ),
                TaskType.RUL_REGRESSION,
                sequences,
                targets,
                split,
                seed=6,
                epochs=1,
                batch_size=4,
                learning_rate=0.001,
                workers=0,
                trial_dir=trial_dir,
                save_checkpoint=True,
                optimizer_name="adamw",
                weight_decay=0.01,
                loss_name="huber",
                grad_clip_norm=1.0,
                lr_scheduler_name="cosine",
                early_stopping_patience=2,
                restore_best=True,
            )
            self.assertIn("rmse", validation)
            self.assertIn("mae", test)
            self.assertGreater(parameters, 0)
            self.assertEqual(prediction.shape, (2,))
            self.assertTrue((trial_dir / "last_checkpoint.pt").is_file())
            self.assertTrue((trial_dir / "best_checkpoint.pt").is_file())
            self.assertTrue((trial_dir / "training_history.json").is_file())
            self.assertTrue((trial_dir / "artifact_manifest.json").is_file())
            manifest = json.loads((trial_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["seed"], 6)
            self.assertEqual(manifest["training_summary"]["optimizer"], "adamw")
            self.assertEqual(manifest["training_summary"]["loss"], "huber")

    def test_torch_rul_unit_interval_constraint_bounds_predictions(self):
        rng = np.random.default_rng(12)
        sequences = np.empty(12, dtype=object)
        sequences[:] = [rng.normal(size=(8, 2)).astype(np.float32) for _ in range(12)]
        targets = np.linspace(1.0, 0.0, 12, dtype=np.float32)
        split = DataSplit(train=np.arange(0, 8), val=np.arange(8, 10), test=np.arange(10, 12))
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, validation_prediction, test_prediction = run_torch_candidate(
                AlgorithmCandidate(
                    "THGNN",
                    "standard",
                    {"d_model": 16, "graph_depth": 1},
                    backend="pytorch",
                ),
                TaskType.RUL_REGRESSION,
                sequences,
                targets,
                split,
                seed=12,
                epochs=1,
                batch_size=4,
                learning_rate=0.001,
                workers=0,
                trial_dir=Path(directory),
                save_checkpoint=False,
                target_constraint="unit_interval",
            )
            self.assertTrue(((validation_prediction >= 0) & (validation_prediction <= 1)).all())
            self.assertTrue(((test_prediction >= 0) & (test_prediction <= 1)).all())


class AutoMLTaskTests(unittest.TestCase):
    def test_anomaly_run_ranks_validation_then_opens_sealed_test_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = LeaderboardConfig(
                task_id="esa-anomaly-smoke",
                dataset=_esa_spec(root),
                task=TaskSpec(type="anomaly_detection"),
                split=SplitSpec(type="predefined"),
                resources=ResourceSpec(timeout_per_trial_minutes=1, max_total_minutes=2),
                search=SearchSpec(
                    strategy="none",
                    max_trials=1,
                    preprocessing=("none",),
                    algorithms=("GlobalSTD",),
                    anomaly_threshold_candidates=20,
                ),
                artifacts=ArtifactSpec(output_dir=str(root / "runs")),
            )
            result = LeaderboardTask(config).run()
            self.assertEqual(result.leaderboard.iloc[0]["status"], "completed")
            self.assertNotIn("test_point_f1", result.leaderboard)
            sealed = json.loads((result.run_dir / "sealed_test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(sealed["status"], "completed")
            self.assertEqual(sealed["threshold_selected_on"], "validation")
            self.assertIn("point_f1", sealed["test_metrics"])
            self.assertTrue((result.run_dir / "best_model" / "predictions.csv").is_file())

    def _make_config(self, root: Path) -> LeaderboardConfig:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(60, 3))
        frame = pd.DataFrame(x, columns=["a", "b", "c"])
        frame["label"] = (frame["a"] + frame["b"] > 0).astype(int)
        data_path = root / "data.csv"
        frame.to_csv(data_path, index=False)
        return LeaderboardConfig(
            task_id="classification-smoke",
            dataset=DatasetSpec(path=str(data_path), label_columns=("label",)),
            task=TaskSpec(type="tabular_classification"),
            split=SplitSpec(type=SplitType.STRATIFIED),
            resources=ResourceSpec(timeout_per_trial_minutes=1, max_total_minutes=2),
            search=SearchSpec(
                strategy=SearchStrategy.NONE,
                max_trials=1,
                preprocessing=("standard",),
                algorithms=("LogisticRegression",),
            ),
            artifacts=ArtifactSpec(output_dir=str(root / "runs")),
        )

    def test_plan_is_deterministic_and_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            task = LeaderboardTask(self._make_config(Path(directory)))
            task.validate()
            first = task.build_plan()
            second = task.build_plan()
            self.assertEqual(first, second)
            self.assertEqual(first["candidate_count"], 1)
            self.assertEqual(sum(first["split_sizes"].values()), 60)

    def test_invalid_ranking_metric_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._make_config(Path(directory))
            payload = config.to_dict()
            payload["ranking"] = {"primary_metric": "rmse"}
            task = LeaderboardTask(LeaderboardConfig.from_dict(payload))
            with self.assertRaisesRegex(ValueError, "Unsupported ranking metrics"):
                task.build_plan()

    def test_yaml_configuration_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._make_config(root)
            config_path = root / "task.yaml"
            payload = json.loads(json.dumps(config.to_dict(), default=str))
            import yaml

            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            loaded = load_config(config_path)
            self.assertEqual(loaded.task_id, config.task_id)
            self.assertEqual(loaded.task.type, TaskType.TABULAR_CLASSIFICATION)

    def test_end_to_end_run_writes_all_core_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            task = LeaderboardTask(self._make_config(Path(directory)))
            result = task.run()
            self.assertEqual(len(result.leaderboard), 1)
            self.assertEqual(result.leaderboard.iloc[0]["status"], "completed")
            for name in (
                "config.json",
                "dataset_profile.json",
                "environment.json",
                "data_split.json",
                "leaderboard.csv",
                "final_report.md",
            ):
                self.assertTrue((result.run_dir / name).is_file(), name)
            self.assertTrue((result.run_dir / "best_model" / "pipeline.joblib").is_file())
            self.assertTrue((result.run_dir / "trials" / "trial-0000" / "artifact_manifest.json").is_file())
            trial_config = json.loads(
                (result.run_dir / "trials" / "trial-0000" / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(trial_config["seed"], task.config.seed)
            report = (result.run_dir / "final_report.md").read_text(encoding="utf-8")
            self.assertIn("模型排行榜报告", report)

    def test_group_cv_run_ranks_validation_and_evaluates_sealed_test_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for group_index in range(6):
                for sample_index in range(6):
                    feature = group_index + sample_index / 10
                    rows.append(
                        {
                            "entity": f"g{group_index}",
                            "feature": feature,
                            "target": 2.0 * feature + 1.0,
                        }
                    )
            data_path = root / "regression.csv"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config = LeaderboardConfig(
                task_id="group-cv-smoke",
                dataset=DatasetSpec(
                    path=str(data_path),
                    label_columns=("target",),
                    entity_column="entity",
                ),
                task=TaskSpec(type="tabular_regression"),
                split=SplitSpec(type="group_cv", cv_folds=3, test_groups=("g5",)),
                resources=ResourceSpec(timeout_per_trial_minutes=1, max_total_minutes=2),
                search=SearchSpec(
                    strategy="none",
                    max_trials=1,
                    preprocessing=("standard",),
                    algorithms=("Ridge",),
                ),
                artifacts=ArtifactSpec(output_dir=str(root / "runs")),
            )
            result = LeaderboardTask(config).run()
            self.assertEqual(result.leaderboard.iloc[0]["status"], "completed")
            self.assertIn("val_rmse", result.leaderboard)
            self.assertIn("val_rmse_std", result.leaderboard)
            self.assertNotIn("test_rmse", result.leaderboard)
            sealed = json.loads((result.run_dir / "sealed_test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(sealed["status"], "completed")
            self.assertEqual(sealed["test_groups"], ["g5"])
            self.assertIn("rmse", sealed["test_metrics"])
            self.assertTrue((result.run_dir / "best_model" / "pipeline.joblib").is_file())
            fold_dirs = list((result.run_dir / "trials" / "trial-0000").glob("fold-*"))
            self.assertEqual(len(fold_dirs), 3)

    def test_forecasting_run_uses_rolling_origin_and_multistep_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = np.arange(240)
            data_path = root / "forecast.csv"
            pd.DataFrame(
                {
                    "time": steps,
                    "target": np.sin(steps / 9) + steps / 200,
                    "exogenous": np.cos(steps / 13),
                }
            ).to_csv(data_path, index=False)
            config = LeaderboardConfig(
                task_id="forecasting-smoke",
                dataset=DatasetSpec(
                    path=str(data_path),
                    label_columns=("target",),
                    feature_columns=("target", "exogenous"),
                    time_column="time",
                    sample_mode="sliding_window",
                    sequence_length=32,
                    prediction_length=8,
                    stride=8,
                ),
                task=TaskSpec(type="time_series_forecasting"),
                split=SplitSpec(type="rolling_origin", rolling_folds=3),
                resources=ResourceSpec(timeout_per_trial_minutes=1, max_total_minutes=2),
                search=SearchSpec(
                    strategy="grid",
                    max_trials=2,
                    preprocessing=("standard",),
                    algorithms=("LastValue", "DLinear"),
                    torch_epochs=1,
                    torch_batch_size=8,
                ),
                artifacts=ArtifactSpec(output_dir=str(root / "runs")),
            )
            result = LeaderboardTask(config).run()
            self.assertEqual(set(result.leaderboard["status"]), {"completed"})
            self.assertIn("baseline", " ".join(result.leaderboard["pipeline"]))
            self.assertEqual(result.leaderboard.iloc[0]["status"], "completed")
            self.assertIn("val_horizon_8_rmse", result.leaderboard)
            sealed = json.loads((result.run_dir / "sealed_test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(sealed["status"], "completed")
            self.assertIsNotNone(sealed["test_timestamp_range"])
            predictions = pd.read_csv(result.run_dir / "best_model" / "predictions.csv")
            self.assertEqual(len([column for column in predictions if column.startswith("prediction_")]), 8)
            preprocessor = joblib.load(result.run_dir / "trials" / "trial-0001" / "fold-00" / "preprocessor.joblib")
            self.assertIsNotNone(preprocessor["target_scaler"])

    def test_torch_like_rul_candidate_uses_same_leaderboard_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for entity_index in range(12):
                for step in range(8):
                    rows.append(
                        {
                            "entity": f"bearing-{entity_index}",
                            "time": step,
                            "sensor_a": entity_index + step / 10,
                            "sensor_b": entity_index - step / 20,
                            "rul": 100 - entity_index * 4 - step,
                        }
                    )
            data_path = root / "rul.csv"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config = LeaderboardConfig(
                task_id="rul-torch-smoke",
                dataset=DatasetSpec(
                    path=str(data_path),
                    label_columns=("rul",),
                    entity_column="entity",
                    time_column="time",
                    sample_mode="full_sequence",
                ),
                task=TaskSpec(type="rul_regression"),
                split=SplitSpec(type="group_cv", cv_folds=3, test_groups=("bearing-11",)),
                resources=ResourceSpec(timeout_per_trial_minutes=1, max_total_minutes=2),
                search=SearchSpec(
                    strategy="none",
                    max_trials=1,
                    preprocessing=("standard",),
                    algorithms=("THGNN",),
                    torch_epochs=1,
                    torch_batch_size=4,
                ),
                artifacts=ArtifactSpec(output_dir=str(root / "runs")),
            )
            result = LeaderboardTask(config).run()
            self.assertEqual(result.leaderboard.iloc[0]["status"], "completed")
            self.assertTrue((result.run_dir / "best_model" / "last_checkpoint.pt").is_file())
            self.assertTrue((result.run_dir / "best_model" / "preprocessor.joblib").is_file())
            fold_records = json.loads(
                (result.run_dir / "trials" / "trial-0000" / "fold_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(fold_records), 3)
            self.assertTrue(all(len(record["inner_validation_groups"]) == 1 for record in fold_records))
            self.assertTrue(all(record["training"]["best_epoch"] == 1 for record in fold_records))


if __name__ == "__main__":
    unittest.main()
