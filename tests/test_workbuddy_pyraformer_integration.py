"""Public Pyraformer registry/runtime/lazy API and exact mixed-buffer persistence."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_pyraformer import config

import torch_like as tl
from tools.validate_candidate_pool import load_registry_baseline, validate_document


def test_factory_runtime_prediction_and_unassessed_catalog():
    model = tl.create("Pyraformer", backend="torch_like", **config()).eval()
    assert type(model) is tl.Pyraformer
    assert "Pyraformer" in tl.list_models("forecasting")
    info = tl.get_model_info("Pyraformer", domain="forecasting")
    assert info["integration_level"] == "paper_aligned_portable_subset"
    assert info["maturity"] == info["benchmark_evidence"] == "unassessed"
    descriptor = tl.describe("Pyraformer")
    assert descriptor.task.kind == tl.TaskKind.TIME_SERIES_FORECASTING
    assert descriptor.supports("predict")
    runtime = tl.as_runtime(model)
    x = torch.randn(2, 22, 2)
    batch = runtime.make_batch(x_enc=x, x_mark_enc=None, x_dec=torch.zeros(2, 5, 2), x_mark_dec=None)
    actual = runtime.predict(batch)
    assert actual.shape == (2, 3, 2)
    torch.testing.assert_close(actual, model(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_save_load_retains_float_precision_and_integer_boolean_graph(tmp_path, dtype):
    model = tl.Pyraformer.from_config(tl.PyraformerConfig(**config())).to(dtype=dtype)
    x = torch.randn(2, 22, 2, dtype=dtype)
    model(x).square().mean().backward()  # Populate running statistics and counters.
    model.eval()
    if dtype == torch.float64:
        model.position[0, 0, 0] += 1e-10
        model.cscm.bns[0].running_mean[0] += 1e-10
    expected = model.predict(model.make_forecast_batch(x))
    target = tmp_path / "pyraformer"
    tl.save_model(model, target)
    restored = tl.load_model(target).eval()
    assert restored.config() == model.config()
    for key, value in model.state_dict().items():
        assert restored.state_dict()[key].dtype == value.dtype
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert restored.neighbors.dtype == restored.last_ancestors.dtype == torch.long
    assert restored.neighbor_valid.dtype == torch.bool
    assert restored.cscm.bns[0].num_batches_tracked.dtype == torch.long
    assert restored.position.dtype == dtype
    torch.testing.assert_close(restored.predict(restored.make_forecast_batch(x)), expected, rtol=0, atol=0)


@pytest.mark.parametrize("integer_type", [np.int8, np.uint8])
def test_public_save_load_of_fixed_width_integer_model_width(tmp_path, integer_type):
    model = tl.Pyraformer(**config(seq_len=63, d_model=integer_type(64), n_heads=4)).double().eval()
    target = tmp_path / "integer_width"
    tl.save_model(model, target)
    restored = tl.load_model(target).eval()
    x = torch.randn(2, 63, 2, dtype=torch.float64)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    assert restored.head.in_features == model.head.in_features == 256


@pytest.mark.parametrize("namespace", ["torch_like", "torch_like.models", "torch_like.models.forecasting"])
def test_public_export_identity(namespace):
    module = __import__(namespace, fromlist=["Pyraformer"])
    assert module.Pyraformer is tl.Pyraformer and module.PyraformerConfig is tl.PyraformerConfig


def test_exports_are_lazy_in_fresh_interpreter():
    source = """
import sys
import torch_like as tl
assert 'torch_like.models.forecasting.pyraformer' not in sys.modules
assert 'torch_like.models.forecasting.pyraformer_layers' not in sys.modules
assert tl.Pyraformer.__name__ == 'Pyraformer'
assert tl.PyraformerConfig.__name__ == 'PyraformerConfig'
assert 'torch_like.models.vision.vit' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_observation_implementation_leaves_grandfathered_baseline_unchanged():
    model = tl.create("Pyraformer", backend="torch_like", **config())
    assert model.param_count() > 0
    baseline = load_registry_baseline()
    assert len(baseline) == 161 and ("forecasting", "Pyraformer") not in baseline
    assert validate_document(Path("docs/MODEL_CANDIDATE_POOL.md")) == []


def test_public_batch_preserves_first_future_marker_and_ignores_decoder_values():
    model = tl.Pyraformer(**config()).eval()
    future = torch.randn(2, 5, 4)
    batch = model.make_forecast_batch(torch.randn(2, 22, 2), x_mark_dec=future)
    expected = model.predict(batch)
    batch.x_dec.fill_(float("nan"))
    batch.x_mark_dec[:, :2] = float("nan")
    batch.x_mark_dec[:, 3:] = float("nan")
    torch.testing.assert_close(model.predict(batch), expected, rtol=0, atol=0)
