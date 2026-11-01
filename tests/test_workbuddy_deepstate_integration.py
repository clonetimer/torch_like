"""DeepState factory, runtime, training, lazy exports and exact persistence."""

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from test_deepstate import batch_for, config

import torch_like as tl
from tools.validate_candidate_pool import load_registry_baseline, validate_document
from torch_like.training import Trainer, TrainerConfig
from torch_like.training.adapters import DefaultBatchAdapter
from torch_like.training.data import move_to_device


def model_factory(**overrides):
    assert "DeepState" in tl.list_models("forecasting"), "DeepState registration is missing"
    return tl.create("DeepState", backend="torch_like", **config(**overrides))


def test_factory_runtime_uses_deepstate_batch_and_all_public_operations():
    model = model_factory(time_feature_dim=2, seasonal_periods=(3,), cardinalities=(2,), embedding_dimensions=(2,))
    assert type(model) is tl.DeepState
    descriptor = tl.describe("DeepState")
    assert descriptor.task.input.batch_type == "torch_like.models.forecasting.DeepStateBatch"
    assert descriptor.task.kind == tl.TaskKind.PROBABILISTIC_FORECASTING
    assert set(descriptor.operations) == {"predict", "distribution", "sample", "loss"}
    info = tl.get_model_info("DeepState", domain="forecasting")
    assert info["integration_level"] == "paper_aligned_portable_subset"
    assert info["maturity"] == info["benchmark_evidence"] == "unassessed"
    runtime = tl.as_runtime(model)
    supplied = batch_for(model)
    supplied.observed_enc = torch.ones_like(supplied.x_enc, dtype=torch.bool)
    supplied.observed_enc[:, 1] = False
    batch = runtime.make_batch(**supplied.__dict__)
    assert type(batch) is tl.DeepStateBatch
    torch.testing.assert_close(runtime.predict(batch), model.predict(batch), rtol=0, atol=0)
    torch.testing.assert_close(runtime.distribution(batch)["mean"], model.predict(batch), rtol=0, atol=0)
    torch.testing.assert_close(runtime.loss(batch), model.fit_step(batch), rtol=0, atol=0)
    assert runtime.sample(batch, num_samples=3).shape == (3, 2, 3, 1)


def test_training_and_device_adapter_preserve_custom_fields():
    model = model_factory(time_feature_dim=2, seasonal_periods=(3,), cardinalities=(2,), embedding_dimensions=(2,))
    batch = batch_for(model)
    batch.observed_enc = torch.ones_like(batch.x_enc, dtype=torch.bool)
    batch.observed_dec = torch.ones_like(batch.y_true, dtype=torch.bool)
    batch.observed_enc[:, 1] = False
    batch.observed_dec[:, 1] = False
    batch.x_enc[:, 1] = float("nan")
    batch.y_true[:, 1] = float("nan")
    adapted = DefaultBatchAdapter().adapt(model, batch)
    moved = move_to_device(adapted.model_batch, torch.device("cpu"))
    assert type(moved) is tl.DeepStateBatch
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(moved, field.name), value, rtol=0, atol=0, equal_nan=True)
    before = model.predict(moved).detach().clone()
    result = Trainer(model, TrainerConfig(epochs=1, device="cpu")).fit([moved])
    assert result.epochs_ran == 1
    assert torch.isfinite(torch.tensor(result.best_monitored_loss))
    assert not torch.equal(model.predict(moved), before)


def test_generic_training_pair_works_only_for_feature_free_configuration():
    model = model_factory()
    batch = batch_for(model)
    pair = (batch.x_enc, batch.y_true)
    adapted = DefaultBatchAdapter().adapt(model, pair)
    assert type(adapted.model_batch) is tl.DeepStateBatch
    assert adapted.model_batch.x_dec is None and adapted.model_batch.x_mark_enc is None
    assert torch.isfinite(tl.as_runtime(model).loss(adapted.model_batch))
    minimal = tl.as_runtime(model).make_batch(x_enc=batch.x_enc, y_true=batch.y_true)
    torch.testing.assert_close(model.predict(minimal), model.predict(adapted.model_batch), rtol=0, atol=0)
    requiring_features = model_factory(time_feature_dim=2)
    incomplete = DefaultBatchAdapter().adapt(requiring_features, pair)
    with pytest.raises(ValueError, match="x_mark_enc"):
        requiring_features.fit_step(incomplete.model_batch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("integer_type", [int, np.int8, np.uint8])
def test_public_save_load_preserves_config_state_and_predictions(tmp_path, dtype, integer_type):
    model = model_factory(
        seq_len=integer_type(4),
        label_len=integer_type(1),
        pred_len=integer_type(3),
        seasonal_periods=[integer_type(3)],
        time_feature_dim=integer_type(2),
        dtype=dtype,
    )
    batch = batch_for(model)
    loss = model.fit_step(batch)
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.001).step()
    model.eval()
    expected = model.predict_distribution(batch)
    path = tmp_path / "deepstate"
    tl.save_model(model, path)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64 if dtype == torch.float32 else torch.float32)
        restored = tl.load_model(path).eval()
        assert torch.get_default_dtype() != dtype
    finally:
        torch.set_default_dtype(original)
    assert restored.config() == model.config() == json.loads(json.dumps(model.config()))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for name in ("mean", "scale"):
        torch.testing.assert_close(restored.predict_distribution(batch)[name], expected[name], rtol=0, atol=0)
    torch.manual_seed(41)
    samples = model.sample(batch, 9)
    torch.manual_seed(41)
    reloaded = restored.sample(batch, 9)
    torch.testing.assert_close(samples, reloaded, rtol=0, atol=0)


@pytest.mark.parametrize("namespace", ["torch_like", "torch_like.models", "torch_like.models.forecasting"])
def test_all_namespaces_export_public_classes(namespace):
    module = __import__(namespace, fromlist=["DeepState"])
    assert hasattr(module, "DeepState"), "DeepState exports are missing"
    assert module.DeepState is tl.DeepState
    assert module.DeepStateConfig is tl.DeepStateConfig
    assert module.DeepStateBatch is tl.DeepStateBatch


def test_exports_are_lazy_and_forecasting_batch_contract_is_not_globally_changed():
    model_factory()
    script = """
import sys
import torch_like as tl
assert 'torch_like.models.forecasting.deepstate' not in sys.modules
assert 'torch_like.models.forecasting.deepstate_layers' not in sys.modules
assert tl.DeepState.__name__ == 'DeepState'
assert tl.DeepStateConfig.__name__ == 'DeepStateConfig'
assert tl.DeepStateBatch.__name__ == 'DeepStateBatch'
assert 'torch_like.models.vision.vit' not in sys.modules
assert tl.describe('DeepAR').task.input.batch_type.endswith('.ProbabilisticForecastingBatch')
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_deepstate_is_admitted_without_expanding_grandfathered_registry():
    model = model_factory()
    assert model.param_count() > 0
    baseline = load_registry_baseline()
    assert len(baseline) == 161 and ("forecasting", "DeepState") not in baseline
    assert validate_document(Path("docs/MODEL_CANDIDATE_POOL.md")) == []
    assert "Paloma" not in tl.list_models() and "SEER" not in tl.list_models()
