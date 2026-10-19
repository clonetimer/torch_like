import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch_like as tl
from runner.benchmark_compile import BENCHMARK_MODELS, benchmark_one
from runner.checkpoints import load_checkpoint, save_checkpoint
from runner.common import average_total, build_run_payload
from runner.compile_utils import CompileConfig, maybe_compile_model
from runner.detection_data import DetectionDataConfig, load_detection_data
from runner.detection_runner import DetectionRunConfig, default_model_specs
from runner.detection_runner import main as run_detection_main
from runner.forecast_comparison_runner import ForecastComparisonConfig
from runner.forecast_comparison_runner import default_model_specs as forecast_default_model_specs
from runner.forecast_comparison_runner import main as run_forecast_comparison_main
from runner.forecast_data import ForecastDataConfig, create_forecast_loaders, load_forecast_data
from runner.forecast_runner import (
    ForecastRunConfig,
    _make_decoder_input,
    _to_forecasting_batch,
)
from runner.forecast_runner import (
    main as run_forecast_main,
)
from runner.performance import ExecutionConfig, autocast_context, create_grad_scaler, move_to_device
from runner.report import load_result_jsons, summarize_results
from runner.rul_data import RULDataConfig, create_rul_loaders
from runner.rul_runner import RULRunConfig, _development_loaders
from runner.rul_runner import run_experiment as run_rul_experiment
from torch_like.utils.tools import EarlyStopping, dotdict


class RunnerApiTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.tmp_path = Path(self._temporary_directory.name)

    def test_early_stopping_creates_checkpoint_directory(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "model.pt"
            stopper = EarlyStopping(patience=2)
            stopper(1.0, nn.Linear(2, 1), path)
            self.assertTrue(path.is_file())

    def test_dotdict_missing_attribute_is_not_silent(self):
        config = dotdict(value=1)
        self.assertEqual(config.value, 1)
        with self.assertRaises(AttributeError):
            _ = config.missing

    def test_compile_helper_disabled_returns_same_model(self):
        model = nn.Linear(2, 1)
        compiled = maybe_compile_model(model, CompileConfig(enabled=False))
        self.assertIs(compiled, model)

    def test_compile_helper_prefers_module_compile(self):
        model = nn.Linear(2, 1)
        with patch.object(model, "compile") as module_compile:
            compiled = maybe_compile_model(model, CompileConfig(enabled=True, backend="eager"))
        self.assertIs(compiled, model)
        module_compile.assert_called_once_with(fullgraph=False, backend="eager")

    def test_execution_config_validates_amp_dtype(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(amp_dtype="float32")

    def test_execution_helpers_on_cpu(self):
        config = ExecutionConfig(mixed_precision=True, amp_dtype="bfloat16")
        tensor = move_to_device(torch.randn(2, 2), "cpu", config)
        with autocast_context("cpu", config):
            output = nn.Linear(2, 2)(tensor)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertIsNone(create_grad_scaler("cpu", config))

    def test_load_esa_style_csv(self):
        tmp = self.tmp_path / "runner_csv"
        tmp.mkdir(exist_ok=True)
        train_path = tmp / "train.csv"
        test_path = tmp / "test.csv"
        df = pd.DataFrame(
            {
                "timestamp": ["2000-01-01 00:00:00", "2000-01-01 00:00:30", "2000-01-01 00:01:00"],
                "channel_1": [0.1, 0.2, 0.3],
                "channel_2": [0.4, 0.5, 0.6],
                "is_anomaly_channel_1": [0, 0, 1],
                "is_anomaly_channel_2": [0, 0, 0],
            }
        )
        df.to_csv(train_path, index=False)
        df.to_csv(test_path, index=False)
        cfg = DetectionDataConfig(
            train_path=str(train_path), test_path=str(test_path), max_train_rows=3, max_test_rows=3
        )
        bundle = load_detection_data(cfg)
        self.assertEqual(bundle.train_data.shape, (3, 2))
        self.assertEqual(bundle.test_data.shape, (3, 2))
        self.assertEqual(bundle.test_labels.tolist(), [0, 0, 1])
        self.assertEqual(len(bundle.extra["columns"]), 2)

    def test_rul_data_config_defaults(self):
        cfg = RULDataConfig(root_dir="dummy_root")
        self.assertEqual(cfg.condition, "35Hz12kN")
        self.assertEqual(cfg.seq_len, 2048)
        self.assertEqual(cfg.batch_size, 32)

    def test_forecast_numpy_split_loader(self):
        data = np.random.rand(200, 3).astype(np.float32)
        cfg = ForecastDataConfig(
            dataset_type="numpy",
            numpy_data=data,
            seq_len=12,
            label_len=6,
            pred_len=3,
        )
        bundle = load_forecast_data(cfg)
        self.assertGreater(len(bundle.train_set), 0)
        self.assertGreater(len(bundle.val_set), 0)
        self.assertGreater(len(bundle.test_set), 0)
        sample = bundle.train_set[0]
        self.assertIn("x_enc", sample)
        self.assertIn("x_dec_target", sample)

    def test_graph_forecast_split_loader(self):
        data = np.random.rand(240, 5).astype(np.float32)
        adjacency = np.eye(5, dtype=np.float32)
        cfg = ForecastDataConfig(
            dataset_type="graph_npy",
            numpy_data=data,
            adjacency=adjacency,
            seq_len=24,
            label_len=12,
            pred_len=6,
        )
        bundle = load_forecast_data(cfg)
        self.assertGreater(len(bundle.train_set), 0)
        self.assertEqual(bundle.extra["num_nodes"], 5)
        self.assertEqual(bundle.extra["feature_dim"], 1)
        self.assertEqual(len(bundle.extra["adjacency"]), 5)

    def test_forecast_target_exogenous_split_loader(self):
        data = np.random.rand(200, 4).astype(np.float32)
        cfg = ForecastDataConfig(
            dataset_type="numpy",
            numpy_data=data,
            seq_len=12,
            label_len=6,
            pred_len=3,
            target_indices=[0, 1],
            exogenous_indices=[2, 3],
        )
        bundle = load_forecast_data(cfg)
        sample = bundle.train_set[0]
        self.assertEqual(sample["x_enc"].shape[-1], 4)
        self.assertEqual(sample["x_dec_target"].shape[-1], 2)
        self.assertEqual(sample["x_target_enc"].shape[-1], 2)
        self.assertEqual(sample["x_exog_enc"].shape[-1], 2)
        self.assertEqual(bundle.extra["target_indices"], [0, 1])
        self.assertEqual(bundle.extra["exogenous_indices"], [2, 3])

    def test_forecast_batch_conversion_preserves_optional_fields(self):
        batch = (
            torch.randn(2, 8, 3),
            torch.randn(2, 6, 2),
            torch.zeros(2, 8, 4),
            torch.zeros(2, 6, 4),
        )
        converted = _to_forecasting_batch(batch, "cpu", label_len=2)
        self.assertIsNone(converted.x_target_enc)
        self.assertEqual(converted.x_dec.shape, (2, 6, 2))

    def test_decoder_input_uses_explicit_label_length(self):
        target = torch.arange(8.0).view(1, 8, 1)
        decoder = _make_decoder_input(target, label_len=3)
        self.assertEqual(decoder[0, :3, 0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(decoder[0, 3:, 0].tolist(), [0.0] * 5)

    def test_forecast_loader_cuda_kwargs(self):
        data = np.random.rand(64, 3).astype(np.float32)
        cfg = ForecastDataConfig(
            dataset_type="numpy",
            numpy_data=data,
            seq_len=12,
            label_len=6,
            pred_len=3,
        )
        _, train_loader, _, _ = create_forecast_loaders(cfg, device="cuda")
        self.assertTrue(getattr(train_loader, "pin_memory", False))

    def test_rul_loader_cuda_kwargs(self):
        cfg = RULDataConfig(root_dir="dummy_root")
        dataset = torch.utils.data.TensorDataset(torch.randn(64, 8, 2), torch.randn(64))
        with patch("torch_like.data.task_data.create_train_test_split", return_value=(dataset, dataset)):
            _, train_loader, _ = create_rul_loaders(cfg, device="cuda")
        self.assertTrue(getattr(train_loader, "pin_memory", False))

    def test_graph_forecast_runner_injects_prior_adjacency(self):
        data = np.random.rand(160, 4).astype(np.float32)
        adjacency = np.eye(4, dtype=np.float32)
        config = ForecastRunConfig(
            data=ForecastDataConfig(
                dataset_type="graph_npy",
                numpy_data=data,
                adjacency=adjacency,
                seq_len=16,
                label_len=8,
                pred_len=4,
                batch_size=8,
            ),
            model_name="MTGNN",
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            model_kwargs={"d_model": 16, "e_layers": 1, "d_ff": 32, "dropout": 0.1, "graph_mode": "prior"},
        )
        model, _, _ = run_forecast_main(config)
        self.assertEqual(model.graph_mode, "prior")
        self.assertIsNotNone(model.prior_adjacency)

    def test_graphwavenet_runner_injects_prior_adjacency(self):
        data = np.random.rand(160, 4).astype(np.float32)
        adjacency = np.eye(4, dtype=np.float32)
        config = ForecastRunConfig(
            data=ForecastDataConfig(
                dataset_type="graph_npy",
                numpy_data=data,
                adjacency=adjacency,
                seq_len=16,
                label_len=8,
                pred_len=4,
                batch_size=8,
            ),
            model_name="GraphWaveNet",
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            model_kwargs={
                "e_layers": 2,
                "d_ff": 32,
                "dropout": 0.1,
                "graph_mode": "prior",
                "residual_channels": 16,
                "skip_channels": 32,
                "end_channels": 64,
            },
        )
        model, _, _ = run_forecast_main(config)
        self.assertEqual(model.graph_mode, "prior")
        self.assertIsNotNone(model.prior_adjacency)

    def test_dcrnn_runner_injects_prior_adjacency(self):
        data = np.random.rand(160, 4).astype(np.float32)
        adjacency = np.eye(4, dtype=np.float32)
        config = ForecastRunConfig(
            data=ForecastDataConfig(
                dataset_type="graph_npy",
                numpy_data=data,
                adjacency=adjacency,
                seq_len=12,
                label_len=6,
                pred_len=3,
                batch_size=8,
            ),
            model_name="DCRNN",
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            model_kwargs={"d_model": 16, "e_layers": 2, "dropout": 0.1, "graph_mode": "prior", "diffusion_steps": 2},
        )
        model, _, _ = run_forecast_main(config)
        self.assertEqual(model.graph_mode, "prior")
        self.assertIsNotNone(model.prior_adjacency)

    def test_forecast_runner_compile_config_smoke(self):
        data = np.random.rand(96, 3).astype(np.float32)
        config = ForecastRunConfig(
            data=ForecastDataConfig(
                dataset_type="numpy",
                numpy_data=data,
                seq_len=12,
                label_len=6,
                pred_len=3,
                batch_size=8,
            ),
            model_name="PatchTST",
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            compile=CompileConfig(enabled=False),
            model_kwargs={"d_model": 16, "n_heads": 4, "e_layers": 1, "d_ff": 32, "dropout": 0.1, "patch_len": 4},
        )
        model, _, _ = run_forecast_main(config)
        self.assertIsNotNone(model)

    def test_forecast_runner_target_exogenous_smoke(self):
        data = np.random.rand(96, 4).astype(np.float32)
        config = ForecastRunConfig(
            data=ForecastDataConfig(
                dataset_type="numpy",
                numpy_data=data,
                seq_len=12,
                label_len=4,
                pred_len=3,
                batch_size=8,
                target_indices=[0, 1],
                exogenous_indices=[2, 3],
            ),
            model_name="Informer",
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            model_kwargs={"d_model": 8, "n_heads": 2, "e_layers": 1, "d_layers": 1, "d_ff": 16},
        )
        model, _, metrics = run_forecast_main(config)
        self.assertEqual(model.config()["dec_in"], 2)
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_forecast_runner_three_dimensional_graph_data_smoke(self):
        data = np.random.rand(72, 3, 2).astype(np.float32)
        config = ForecastRunConfig(
            data=ForecastDataConfig(
                dataset_type="graph_npy",
                numpy_data=data,
                adjacency=np.eye(3, dtype=np.float32),
                seq_len=8,
                label_len=4,
                pred_len=2,
                batch_size=8,
                target_indices=[1],
                exogenous_indices=[0],
            ),
            model_name="DCRNN",
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            model_kwargs={
                "d_model": 4,
                "e_layers": 1,
                "diffusion_steps": 1,
                "graph_mode": "prior",
            },
        )
        model, _, metrics = run_forecast_main(config)
        self.assertEqual(model.config()["enc_in"], 6)
        self.assertEqual(model.config()["c_out"], 3)
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_tft_runner_target_exogenous_dimension_mapping(self):
        data = np.random.rand(96, 4).astype(np.float32)
        config = ForecastRunConfig(
            data=ForecastDataConfig(
                dataset_type="numpy",
                numpy_data=data,
                seq_len=12,
                label_len=4,
                pred_len=3,
                batch_size=8,
                target_indices=[0, 1],
                exogenous_indices=[2, 3],
            ),
            model_name="TFT",
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            model_kwargs={"d_model": 8, "n_heads": 2, "e_layers": 1, "d_layers": 1, "d_ff": 16},
        )
        model, _, metrics = run_forecast_main(config)
        self.assertEqual(model.config()["enc_in"], 8)
        self.assertEqual(model.config()["dec_in"], 6)
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_forecast_comparison_default_specs(self):
        cfg = ForecastComparisonConfig(
            data=ForecastDataConfig(
                dataset_type="numpy",
                numpy_data=np.random.rand(96, 3).astype(np.float32),
                seq_len=12,
                label_len=6,
                pred_len=3,
                batch_size=8,
            ),
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
        )
        specs = forecast_default_model_specs(cfg, feature_dim=3, out_dim=3, adjacency=None)
        names = [spec["name"] for spec in specs]
        self.assertIn("NBEATS", names)
        self.assertIn("GraphWaveNet", names)
        self.assertIn("DCRNN", names)

    def test_forecast_comparison_smoke(self):
        data = np.random.rand(120, 3).astype(np.float32)
        config = ForecastComparisonConfig(
            data=ForecastDataConfig(
                dataset_type="numpy",
                numpy_data=data,
                seq_len=12,
                label_len=6,
                pred_len=3,
                batch_size=8,
            ),
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            compile=CompileConfig(enabled=False),
            model_specs=[
                {
                    "name": "NBEATS",
                    "kwargs": {
                        "enc_in": 3,
                        "dec_in": 3,
                        "c_out": 3,
                        "seq_len": 12,
                        "label_len": 6,
                        "pred_len": 3,
                        "d_model": 16,
                        "e_layers": 2,
                        "d_ff": 32,
                        "dropout": 0.1,
                        "stack_layers": 2,
                    },
                },
                {
                    "name": "PatchTST",
                    "kwargs": {
                        "enc_in": 3,
                        "dec_in": 3,
                        "c_out": 3,
                        "seq_len": 12,
                        "label_len": 6,
                        "pred_len": 3,
                        "d_model": 16,
                        "n_heads": 1,
                        "e_layers": 1,
                        "d_ff": 32,
                        "dropout": 0.1,
                        "patch_len": 3,
                    },
                },
            ],
        )
        results = run_forecast_comparison_main(config)
        self.assertEqual(len(results), 2)

    def test_compile_benchmark_smoke(self):
        result = benchmark_one(
            "PatchTST",
            BENCHMARK_MODELS["PatchTST"],
            device="cpu",
            warmup=1,
            iters=1,
            compile_cfg=CompileConfig(enabled=False),
        )
        self.assertEqual(result["model"], "PatchTST")
        self.assertIn("eager_ms", result)

    def test_build_run_payload_shape(self):
        cfg = RULDataConfig(root_dir="dummy_root")
        payload = build_run_payload(config=cfg, history=[{"epoch": 1}], extra={"task": "rul"}, metrics={"rmse": 0.1})
        self.assertIn("config", payload)
        self.assertIn("history", payload)
        self.assertIn("extra", payload)
        self.assertIn("metrics", payload)

    def test_average_total_rejects_empty_loader(self):
        with self.assertRaises(ValueError):
            average_total(0.0, 0)

    def test_build_run_payload_summarizes_array_config(self):
        cfg = ForecastDataConfig(dataset_type="numpy", numpy_data=np.zeros((10, 3), dtype=np.float32))
        payload = build_run_payload(config=cfg, history=[])
        self.assertEqual(payload["config"]["numpy_data"]["shape"], [10, 3])
        self.assertEqual(payload["config"]["numpy_data"]["dtype"], "float32")

    def test_checkpoint_roundtrip(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir) / "nested" / "checkpoint.pt"
            model = nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            save_checkpoint(tmp, model, optimizer=optimizer, epoch=3, history=[{"epoch": 3}], extra={"task": "test"})
            loaded = load_checkpoint(tmp, model, optimizer=optimizer, map_location="cpu")
            self.assertEqual(loaded["epoch"], 3)
            self.assertEqual(loaded["extra"]["task"], "test")

    def test_checkpoint_rejects_missing_model_state(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid.pt"
            torch.save({"epoch": 1}, path)
            with self.assertRaises(KeyError):
                load_checkpoint(path, nn.Linear(2, 1), map_location="cpu")

    def test_report_summary(self):
        tmp_dir = self.tmp_path / "runner_report"
        tmp_dir.mkdir(exist_ok=True)
        payload = build_run_payload(
            config=RULDataConfig(root_dir="dummy_root"),
            history=[{"epoch": 1}],
            extra={"task": "rul"},
            metrics={"test": {"rmse": 0.1}},
        )
        report_path = tmp_dir / "sample.json"
        report_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
        loaded = load_result_jsons(tmp_dir)
        summary = summarize_results(tmp_dir)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(summary), 1)
        self.assertTrue(summary[0]["has_metrics"])

    def test_rul_development_holdout_is_bearing_disjoint(self):
        class BearingDataset(torch.utils.data.Dataset):
            bearing_names = ["B1", "B2", "B3"]
            index_map = [(bearing, index) for bearing in range(3) for index in range(4)]

            def __len__(self):
                return len(self.index_map)

            def __getitem__(self, index):
                return torch.zeros(32, 2), torch.tensor(0.5)

        dataset = BearingDataset()
        bundle = type("Bundle", (), {"train_set": dataset})()
        cfg = RULRunConfig(data=RULDataConfig(root_dir="dummy", batch_size=2), device="cpu")
        train_loader, validation_loader, name = _development_loaders(bundle, cfg)
        training_groups = {dataset.index_map[index][0] for index in train_loader.dataset.indices}
        validation_groups = {dataset.index_map[index][0] for index in validation_loader.dataset.indices}
        self.assertEqual(len(training_groups), 2)
        self.assertEqual(len(validation_groups), 1)
        self.assertFalse(training_groups & validation_groups)
        self.assertIn(name, dataset.bearing_names)

    def test_rul_main_uses_sealed_test_only_for_validation_winner(self):
        from runner import rul_runner

        config = RULRunConfig(
            data=RULDataConfig(root_dir="dummy_root", condition="35Hz12kN"),
            device="cpu",
            results_dir=self.tmp_path,
        )
        bundle = type("Bundle", (), {"train_set": object(), "test_set": object()})()
        train_loader, val_loader, test_loader = object(), object(), object()
        scores = {"InformerRUL": 0.7, "AutoformerRUL": 0.9, "LogTransRUL": 0.4, "THGNN": 0.6}

        def experiment(model_cls, name, provided_bundle, train, validation, experiment_config):
            self.assertIs(provided_bundle, bundle)
            self.assertIs(train, train_loader)
            self.assertIs(validation, val_loader)
            self.assertIs(experiment_config, config)
            return {
                "model": name,
                "params": 10,
                "best_validation_rmse": scores[name],
                "best_validation_mae": scores[name],
                "best_epoch": 1,
                "train_time_s": 0.0,
                "history": {},
                "_trained_model": name,
            }

        with (
            patch("runner.rul_runner.create_rul_loaders", return_value=(bundle, None, test_loader)),
            patch(
                "runner.rul_runner._development_loaders",
                return_value=(train_loader, val_loader, "Bearing1_1"),
            ),
            patch("runner.rul_runner.run_experiment", side_effect=experiment) as run_model,
            patch("runner.rul_runner.evaluate", return_value=(0.1, 0.2, 0.3)) as evaluate_test,
            patch("runner.rul_runner.save_json") as save_results,
        ):
            results = rul_runner.main(config)

        self.assertEqual(run_model.call_count, 4)
        evaluate_test.assert_called_once()
        self.assertEqual(evaluate_test.call_args.args[0], "LogTransRUL")
        self.assertIs(evaluate_test.call_args.args[1], test_loader)
        self.assertEqual(len(results), 4)
        self.assertEqual([item["model"] for item in results if "sealed_test_rmse" in item], ["LogTransRUL"])
        save_results.assert_called_once()
        payload = save_results.call_args.args[1]
        self.assertEqual(payload["extra"]["sealed_test_evaluation_count"], 1)
        self.assertEqual(payload["extra"]["selected_model"], "LogTransRUL")

    def test_thgnn_rul_experiment_smoke(self):
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 12

            def __getitem__(self, index):
                return torch.randn(32, 2), torch.tensor(0.5, dtype=torch.float32)

        bundle = type("Bundle", (), {"train_set": DummyDataset(), "test_set": DummyDataset()})()
        train_loader = torch.utils.data.DataLoader(bundle.train_set, batch_size=4, shuffle=False)
        test_loader = torch.utils.data.DataLoader(bundle.test_set, batch_size=4, shuffle=False)
        config = RULRunConfig(
            data=RULDataConfig(root_dir="dummy_root", seq_len=32, batch_size=4),
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            model_shared={"enc_in": 2, "d_model": 16, "n_heads": 4, "e_layers": 1, "d_ff": 32, "dropout": 0.1},
        )
        result = run_rul_experiment(tl.THGNN, "THGNN", bundle, train_loader, test_loader, config)
        self.assertEqual(result["model"], "THGNN")

    def test_detection_comparison_smoke(self):
        tmp = self.tmp_path / "detection_runner"
        tmp.mkdir(exist_ok=True)
        train_path = tmp / "train.csv"
        test_path = tmp / "test.csv"
        df = pd.DataFrame(
            {
                "timestamp": [f"2000-01-01 00:00:{i:02d}" for i in range(12)],
                "channel_1": np.random.rand(12),
                "channel_2": np.random.rand(12),
                "is_anomaly_channel_1": [0] * 11 + [1],
            }
        )
        df.to_csv(train_path, index=False)
        df.to_csv(test_path, index=False)
        config = DetectionRunConfig(
            data=DetectionDataConfig(
                train_path=str(train_path),
                test_path=str(test_path),
                seq_len=4,
                batch_size=4,
                max_train_rows=12,
                max_test_rows=12,
            ),
            epochs=1,
            device="cpu",
            results_dir=self.tmp_path / "results",
            compile=CompileConfig(enabled=False),
            model_specs=default_model_specs(
                DetectionRunConfig(
                    data=DetectionDataConfig(seq_len=4, batch_size=4),
                    epochs=1,
                    device="cpu",
                    compile=CompileConfig(enabled=False),
                ),
                input_dim=2,
            ),
        )
        results = run_detection_main(config)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
