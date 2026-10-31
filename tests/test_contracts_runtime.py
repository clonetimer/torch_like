import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import torch_like as tl
from torch_like.models.base import BaseModel
from torch_like.runtime import UnsupportedOperationError


class _MixedBufferCheckpointModel(BaseModel):
    """Test-only model with independently typed persistent/nonpersistent buffers."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float32))
        self.register_buffer("precise", torch.tensor([1.0 + 2**-40], dtype=torch.float64))
        self.register_buffer("phase", torch.tensor([1 + 2j], dtype=torch.complex64))
        self.register_buffer("ephemeral", torch.tensor([3 + 4j], dtype=torch.complex128), persistent=False)


class ModelContractTests(unittest.TestCase):
    def test_builtin_contract_imports_only_requested_model(self):
        sys.modules.pop("torch_like.models.forecasting.autoformer", None)
        spec = tl.get_model_spec("PatchTST")
        self.assertEqual(spec.backend, tl.BackendKind.TORCH_LIKE)
        self.assertEqual(spec.domain, "forecasting")
        self.assertEqual(spec.task.kind, tl.TaskKind.TIME_SERIES_FORECASTING)
        self.assertTrue(spec.supports("predict"))
        self.assertFalse(spec.supports("fit"))
        self.assertNotIn("torch_like.models.forecasting.autoformer", sys.modules)

    def test_probabilistic_contract_discovers_concrete_capabilities(self):
        spec = tl.describe("DeepAR")
        self.assertEqual(spec.task.kind, tl.TaskKind.PROBABILISTIC_FORECASTING)
        self.assertEqual(set(spec.operations), {"predict", "loss", "sample", "distribution"})

    def test_sequence_contract_maps_next_token_logits_to_predict(self):
        spec = tl.get_model_spec("GPT")
        self.assertEqual(spec.task.kind, tl.TaskKind.CAUSAL_LANGUAGE_MODELING)
        self.assertTrue(spec.supports("predict"))
        self.assertTrue(spec.supports("loss"))
        self.assertTrue(spec.supports("generate"))

    def test_sklearn_contract_is_native_and_truthful(self):
        classifier = tl.get_model_spec("RandomForestClassifier")
        self.assertEqual(classifier.backend, tl.BackendKind.SKLEARN)
        self.assertEqual(classifier.task.kind, tl.TaskKind.CLASSIFICATION)
        self.assertEqual(classifier.operations, ("fit", "predict"))
        self.assertEqual(tl.get_model_spec("PCA").operations, ("fit", "transform"))
        self.assertEqual(tl.get_model_spec("DBSCAN").operations, ("fit", "fit_predict"))

    def test_partial_contract_fails_closed_for_required_model_specific_inputs(self):
        spec = tl.get_model_spec("ControlNet")
        self.assertEqual(spec.contract_status, tl.ContractStatus.PARTIAL)
        self.assertFalse(spec.supports("predict"))
        self.assertTrue(spec.supports("loss"))
        self.assertTrue(spec.supports("sample"))

    def test_contract_serializes_to_plain_data(self):
        data = tl.get_model_spec("DLinear").to_dict()
        self.assertEqual(data["backend"], "torch_like")
        self.assertEqual(data["task"]["kind"], "time_series_forecasting")
        self.assertEqual(data["contract_version"], "1.0")
        self.assertIsInstance(data["task"]["input"]["fields"], list)


class RuntimeAdapterTests(unittest.TestCase):
    def test_diagnosis_make_batch_predict_and_loss(self):
        model = tl.InceptionTime(
            input_dim=3,
            num_classes=4,
            seq_len=16,
            d_model=8,
            num_blocks=2,
            bottleneck_dim=4,
            kernel_sizes=(3, 5, 7),
        )
        runtime = tl.as_runtime(model)
        batch = runtime.make_batch(x=torch.randn(2, 16, 3), labels=torch.tensor([1, 2]))
        prediction = runtime.predict(batch)
        loss = runtime.loss(batch)
        self.assertEqual(prediction.shape, (2, 4))
        self.assertEqual(loss.ndim, 0)

    def test_sequence_predict_uses_next_token_contract(self):
        model = tl.GPT(vocab_size=16, max_seq_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0)
        runtime = tl.as_runtime(model)
        ids = torch.randint(0, 16, (2, 5))
        batch = runtime.make_batch(input_ids=ids, labels=ids.clone())
        self.assertEqual(runtime.predict(batch).shape, (2, 5, 16))
        self.assertEqual(runtime.loss(batch).ndim, 0)

    def test_sklearn_fit_predict_without_wrapping_native_estimator(self):
        estimator = tl.create("RandomForestClassifier", backend="sklearn", n_estimators=5, random_state=0)
        runtime = tl.as_runtime(estimator)
        self.assertIs(runtime.model, estimator)
        X = np.asarray([[0.0], [1.0], [0.1], [0.9]])
        y = np.asarray([0, 1, 0, 1])
        runtime.fit(X, y)
        self.assertEqual(runtime.predict(np.asarray([[0.05]])).shape, (1,))
        with self.assertRaises(UnsupportedOperationError):
            runtime.sample(1)

    def test_runtime_reports_unsupported_operation_before_dispatch(self):
        model = tl.DLinear(enc_in=2, dec_in=2, c_out=2, seq_len=8, label_len=4, pred_len=2)
        runtime = tl.as_runtime(model)
        with self.assertRaisesRegex(UnsupportedOperationError, "supported operations"):
            runtime.fit(torch.zeros(1, 2, 2))


class SerializationTests(unittest.TestCase):
    def test_torch_model_round_trip_preserves_config_state_and_prediction(self):
        torch.manual_seed(7)
        model = tl.DLinear(enc_in=2, dec_in=2, c_out=2, seq_len=8, label_len=4, pred_len=2).eval()
        x = torch.randn(2, 8, 2)
        batch = model.make_forecast_batch(x)
        expected = model.predict(batch)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            saved = tl.save_model(model, path)
            self.assertEqual(saved, path.resolve())
            self.assertTrue((path / "model.json").is_file())
            self.assertTrue((path / "weights.pt").is_file())
            loaded = tl.load_model(path)
            loaded.eval()
            actual = loaded.predict(loaded.make_forecast_batch(x))

        self.assertEqual(loaded.config(), model.config())
        self.assertTrue(torch.allclose(actual, expected))

    def test_tuple_constructor_values_round_trip(self):
        model = tl.ResNet(
            in_channels=3,
            num_classes=3,
            image_size=32,
            d_model=4,
            layers=(1, 1, 1, 1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resnet"
            tl.save_model(model, path)
            loaded = tl.load_model(path)
        self.assertEqual(loaded.config()["layers"], (1, 1, 1, 1))

    def test_double_checkpoint_preserves_parameters_and_integer_buffers(self):
        model = tl.ResNet(
            in_channels=3,
            num_classes=3,
            image_size=32,
            d_model=4,
            layers=(1, 1, 1, 1),
        ).double()
        saved_state = model.state_dict()
        self.assertTrue(any(t.dtype == torch.int64 for t in saved_state.values()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resnet-double"
            tl.save_model(model, path)
            restored = tl.load_model(path)
        self.assertTrue(all(param.dtype == torch.float64 for param in restored.parameters()))
        for name, saved in saved_state.items():
            actual = restored.state_dict()[name]
            self.assertEqual(actual.dtype, saved.dtype, name)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)

    def test_loader_preserves_mixed_floating_complex_and_nonpersistent_buffers(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(parameter_dtype=dtype):
                model = _MixedBufferCheckpointModel()
                model.weight = torch.nn.Parameter(model.weight.detach().to(dtype=dtype))
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "mixed-buffers"
                    tl.save_model(model, path)
                    restored = tl.load_model(path)
                self.assertEqual(restored.weight.dtype, dtype)
                for name in ("precise", "phase", "ephemeral"):
                    original = getattr(model, name)
                    actual = getattr(restored, name)
                    self.assertEqual(actual.dtype, original.dtype, name)
                    torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_sklearn_round_trip(self):
        estimator = tl.create("RandomForestRegressor", backend="sklearn", n_estimators=5, random_state=0)
        X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        y = np.asarray([0.0, 1.0, 2.0, 3.0])
        estimator.fit(X, y)
        expected = estimator.predict([[1.5]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rf"
            tl.save_model(estimator, path)
            loaded = tl.load_model(path)
            actual = loaded.predict([[1.5]])
        np.testing.assert_allclose(actual, expected)

    def test_save_refuses_accidental_overwrite(self):
        model = tl.DLinear(enc_in=1, dec_in=1, c_out=1, seq_len=4, label_len=2, pred_len=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            tl.save_model(model, path)
            with self.assertRaises(FileExistsError):
                tl.save_model(model, path)
            tl.save_model(model, path, overwrite=True)


class AutoMLContractIntegrationTests(unittest.TestCase):
    def test_candidate_exposes_shared_model_spec(self):
        from torch_like.automl.candidate_registry import list_candidate_specs
        from torch_like.automl.specs import TaskType

        specs = list_candidate_specs(
            TaskType.TIME_SERIES_FORECASTING,
            is_sequence=True,
            equal_sequence_lengths=True,
        )
        patchtst = next(spec for spec in specs if spec.name == "PatchTST")
        model_spec = patchtst.model_spec()
        self.assertIsNotNone(model_spec)
        self.assertEqual(model_spec.domain, "forecasting")
        self.assertEqual(model_spec.task.kind, tl.TaskKind.TIME_SERIES_FORECASTING)

    def test_torch_backend_builds_through_shared_registry(self):
        from torch_like.automl.sklearn_backend import AlgorithmCandidate
        from torch_like.automl.specs import TaskType
        from torch_like.automl.torch_backend import _build_model

        candidate = AlgorithmCandidate(
            name="DLinear",
            preprocessing="none",
            parameters={"moving_avg": 3, "individual": False, "use_norm": True},
            backend="pytorch",
        )
        y = np.zeros((4, 2, 1), dtype=np.float32)
        model, kwargs = _build_model(candidate, TaskType.TIME_SERIES_FORECASTING, 8, 1, y)
        self.assertEqual(model.__class__.__name__, "DLinear")
        self.assertEqual(kwargs["pred_len"], 2)

    def test_torch_backend_rejects_contract_task_mismatch(self):
        from torch_like.automl.sklearn_backend import AlgorithmCandidate
        from torch_like.automl.specs import TaskType
        from torch_like.automl.torch_backend import _build_model

        candidate = AlgorithmCandidate(
            name="FTTransformer",
            preprocessing="none",
            parameters={"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16},
            backend="pytorch",
        )
        y = np.zeros((4, 2, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            _build_model(candidate, TaskType.TIME_SERIES_FORECASTING, 8, 3, y)


if __name__ == "__main__":
    unittest.main()
