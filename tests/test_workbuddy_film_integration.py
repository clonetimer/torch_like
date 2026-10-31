"""Public FiLM factory, runtime, persistence, lazy API and governance contracts."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

import torch_like as tl
from tools.validate_candidate_pool import load_registry_baseline, validate_document


def config():
    return dict(enc_in=2, dec_in=2, c_out=2, seq_len=7, label_len=0, pred_len=3, memory_order=4, scales=(1, 2))


def test_factory_runtime_prediction_and_unassessed_catalog():
    model = tl.create("FiLM", backend="torch_like", **config()).eval()
    assert type(model) is tl.FiLM
    assert "FiLM" in tl.list_models("forecasting")
    info = tl.get_model_info("FiLM", domain="forecasting")
    assert info["integration_level"] == "paper_aligned_portable_subset"
    assert info["maturity"] == info["benchmark_evidence"] == "unassessed"
    spec = tl.describe("FiLM")
    assert spec.task.kind == tl.TaskKind.TIME_SERIES_FORECASTING
    assert spec.supports("predict")
    runtime = tl.as_runtime(model)
    x = torch.randn(2, 7, 2)
    batch = runtime.make_batch(x_enc=x, x_mark_enc=None, x_dec=torch.zeros(2, 3, 2), x_mark_dec=None)
    actual = runtime.predict(batch)
    assert actual.shape == (2, 3, 2)
    torch.testing.assert_close(actual, model(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_save_load_preserves_parameters_fixed_buffers_and_config(tmp_path, dtype):
    model = tl.FiLM.from_config(tl.FiLMConfig(**config())).to(dtype=dtype).eval()
    if dtype == torch.float64:
        # This precision must not be rounded through a reconstructed float32 buffer.
        model.projections[0].A[0, 0] += 1e-10
    x = torch.randn(2, 7, 2, dtype=dtype)
    expected = model.predict(model.make_forecast_batch(x))
    target = tmp_path / "film"
    tl.save_model(model, target)
    restored = tl.load_model(target).eval()
    assert restored.config() == model.config()
    for key, value in model.state_dict().items():
        assert restored.state_dict()[key].dtype == dtype
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(restored.predict(restored.make_forecast_batch(x)), expected, rtol=0, atol=0)


@pytest.mark.parametrize("namespace", ["torch_like", "torch_like.models", "torch_like.models.forecasting"])
def test_public_export_identity(namespace):
    module = __import__(namespace, fromlist=["FiLM"])
    assert module.FiLM is tl.FiLM and module.FiLMConfig is tl.FiLMConfig


def test_exports_are_lazy_in_a_fresh_interpreter():
    source = """
import sys
import torch_like as tl
assert 'torch_like.models.forecasting.film' not in sys.modules
assert 'torch_like.models.forecasting.film_layers' not in sys.modules
assert tl.FiLM.__name__ == 'FiLM'
assert tl.FiLMConfig.__name__ == 'FiLMConfig'
assert 'torch_like.models.vision.vit' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_observation_implementation_does_not_rewrite_grandfathered_baseline():
    baseline = load_registry_baseline()
    assert len(baseline) == 161 and ("forecasting", "FiLM") not in baseline
    assert validate_document(Path("docs/MODEL_CANDIDATE_POOL.md")) == []


def test_public_batch_future_values_cannot_change_prediction():
    model = tl.FiLM(**config()).eval()
    batch = model.make_forecast_batch(torch.randn(2, 7, 2))
    expected = model.predict(batch)
    batch.x_dec.fill_(float("nan"))
    torch.testing.assert_close(model.predict(batch), expected, rtol=0, atol=0)
