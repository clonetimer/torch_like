"""Public GMAN registry/runtime/lazy entrypoints and exact owned-asset persistence."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_gman import config, inputs

import torch_like as tl
from tools.validate_candidate_pool import load_registry_baseline, validate_document


def test_factory_and_runtime_use_required_calendar_and_unassessed_catalog():
    assert "GMAN" in tl.list_models("forecasting"), "GMAN is not registered"
    model = tl.create("GMAN", backend="torch_like", **config()).eval()
    assert type(model) is tl.GMAN
    info = tl.get_model_info("GMAN", domain="forecasting")
    assert info["integration_level"] == "paper_aligned_portable_subset"
    assert info["maturity"] == info["benchmark_evidence"] == "unassessed"
    descriptor = tl.describe("GMAN")
    assert descriptor.task.kind == tl.TaskKind.TIME_SERIES_FORECASTING
    runtime = tl.as_runtime(model)
    x, past, future = inputs(model)
    batch = runtime.make_batch(x_enc=x, x_mark_enc=past, x_dec=torch.zeros(2, 4, 3), x_mark_dec=future)
    actual = runtime.predict(batch)
    assert actual.shape == (2, 2, 3)
    torch.testing.assert_close(actual, model(x, past, None, future), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("integer_type", [int, np.int8, np.uint8])
def test_public_save_load_preserves_embedding_parameters_bn_and_precision(tmp_path, dtype, integer_type):
    assert hasattr(tl, "GMAN"), "GMAN public export is missing"
    torch.manual_seed(81)
    cfg = config(
        d_model=integer_type(64),
        n_heads=integer_type(4),
        time_slots_per_day=np.uint8(250),
        spatial_embedding=np.linspace(-1, 1, 192).reshape(3, 64),
        bn_decay=np.float32(0.9),
    )
    model = tl.GMAN.from_config(tl.GMANConfig(**cfg)).to(dtype=dtype)
    x, past, future = inputs(model, dtype=dtype)
    model(x, past, None, future).square().mean().backward()
    model.eval()
    if dtype == torch.float64:
        model.spatial_embedding[0, 0] += 1e-10
        model.embedding.spatial[0].norm.running_mean[0] += 1e-10
    expected = model.predict(model.make_forecast_batch(x, x_mark_enc=past, x_mark_dec=future))
    path = tmp_path / "gman"
    tl.save_model(model, path)
    restored = tl.load_model(path).eval()
    assert restored.config() == model.config() == json.loads(json.dumps(model.config()))
    for name, tensor in model.state_dict().items():
        actual = restored.state_dict()[name]
        assert actual.dtype == tensor.dtype
        torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
    assert restored.spatial_embedding.dtype == dtype and not restored.spatial_embedding.requires_grad
    assert restored.embedding.spatial[0].norm.num_batches_tracked.dtype == torch.long
    torch.testing.assert_close(
        restored.predict(restored.make_forecast_batch(x, x_mark_enc=past, x_mark_dec=future)), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("namespace", ["torch_like", "torch_like.models", "torch_like.models.forecasting"])
def test_public_namespaces_export_the_same_model_and_config(namespace):
    module = __import__(namespace, fromlist=["GMAN"])
    assert hasattr(module, "GMAN") and hasattr(module, "GMANConfig"), "GMAN exports are missing"
    assert module.GMAN is tl.GMAN and module.GMANConfig is tl.GMANConfig


def test_public_gman_exports_are_lazy_in_fresh_interpreter():
    assert hasattr(tl, "GMAN"), "GMAN public export is missing"
    script = """
import sys
import torch_like as tl
assert 'torch_like.models.forecasting.gman' not in sys.modules
assert 'torch_like.models.forecasting.gman_layers' not in sys.modules
assert tl.GMAN.__name__ == 'GMAN'
assert tl.GMANConfig.__name__ == 'GMANConfig'
assert 'torch_like.models.vision.vit' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, text=True, capture_output=True)


def test_gman_implementation_does_not_expand_grandfathered_baseline():
    assert "GMAN" in tl.list_models("forecasting"), "GMAN is not registered"
    assert tl.create("GMAN", backend="torch_like", **config()).param_count() > 0
    baseline = load_registry_baseline()
    assert len(baseline) == 161 and ("forecasting", "GMAN") not in baseline
    assert validate_document(Path("docs/MODEL_CANDIDATE_POOL.md")) == []


@pytest.mark.parametrize("coordinate", [1e-50, -1e-50, 1e50])
def test_public_float64_only_embedding_roundtrip_survives_changed_default(tmp_path, coordinate):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(103)
        model = tl.GMAN(
            **config(enc_in=1, dec_in=1, c_out=1, d_model=2, n_heads=1, spatial_embedding=[[coordinate, 0.5]])
        ).eval()
        x, past, future = inputs(model, dtype=torch.float64)
        expected = model(x, past, None, future)
        assert torch.isfinite(expected).all()
        path = tmp_path / "double_only_embedding"
        tl.save_model(model, path)
        torch.set_default_dtype(torch.float32)
        restored = tl.load_model(path).eval()
        assert torch.get_default_dtype() == torch.float32
        assert restored.config() == model.config()
        for name, tensor in model.state_dict().items():
            assert restored.state_dict()[name].dtype == tensor.dtype
            torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
        torch.testing.assert_close(restored(x, past, None, future), expected, rtol=0, atol=0)
    finally:
        torch.set_default_dtype(original_dtype)
