"""Registry, runtime, serialization and public-API contract of Workbuddy R2."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import torch_like as tl
from tools.validate_candidate_pool import load_registry_baseline, validate_document


def _config():
    return dict(
        enc_in=3,
        dec_in=3,
        c_out=3,
        seq_len=8,
        label_len=0,
        pred_len=3,
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=16,
        factor_hidden=8,
        dropout=0,
    )


def test_public_registry_factory_runtime_and_unassessed_maturity():
    model = tl.create("NonstationaryTransformer", backend="torch_like", **_config()).eval()
    assert type(model) is tl.NonstationaryTransformer
    assert "NonstationaryTransformer" in tl.list_models("forecasting")
    spec = tl.describe("NonstationaryTransformer")
    assert spec.task.kind == tl.TaskKind.TIME_SERIES_FORECASTING
    assert spec.supports("predict")
    info = tl.get_model_info("NonstationaryTransformer", domain="forecasting")
    assert info["integration_level"] == "paper_aligned_portable_subset"
    assert info["maturity"] == "unassessed" and info["benchmark_evidence"] == "unassessed"
    runtime = tl.as_runtime(model)
    x = torch.randn(2, 8, 3)
    batch = runtime.make_batch(x_enc=x, x_mark_enc=None, x_dec=torch.zeros(2, 3, 3), x_mark_dec=None)
    actual = runtime.predict(batch)
    assert actual.shape == (2, 3, 3)
    torch.testing.assert_close(actual, model(x, None, None, None))


def test_public_save_load_roundtrip_and_from_dataclass_config():
    torch.manual_seed(232)
    config = tl.NonstationaryTransformerConfig(**_config())
    model = tl.NonstationaryTransformer.from_config(config).eval()
    x = torch.randn(2, 8, 3)
    expected = model.predict(model.make_forecast_batch(x))
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "nonstationary"
        tl.save_model(model, target)
        restored = tl.load_model(target).eval()
        assert restored.config() == model.config()
        torch.testing.assert_close(restored.predict(restored.make_forecast_batch(x)), expected)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)


def test_public_float64_save_load_preserves_forecast_precision():
    model = tl.NonstationaryTransformer(**_config()).double().eval()
    x = torch.randn(2, 8, 3, dtype=torch.float64)
    expected = model.predict(model.make_forecast_batch(x))
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "nonstationary64"
        tl.save_model(model, target)
        restored = tl.load_model(target).eval()
        assert next(restored.parameters()).dtype == torch.float64
        actual = restored.predict(restored.make_forecast_batch(x))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("class_location", ["torch_like", "torch_like.models", "torch_like.models.forecasting"])
def test_public_export_identity_across_facades(class_location):
    module = __import__(class_location, fromlist=["NonstationaryTransformer"])
    assert module.NonstationaryTransformer is tl.NonstationaryTransformer
    assert module.NonstationaryTransformerConfig is tl.NonstationaryTransformerConfig


def test_public_export_is_lazy_until_first_access():
    source = """
import sys
import torch_like as tl
assert 'torch_like.models.forecasting.nonstationary_transformer' not in sys.modules
assert 'torch_like.models.forecasting.nonstationary_attention' not in sys.modules
assert tl.NonstationaryTransformer.__name__ == 'NonstationaryTransformer'
assert tl.NonstationaryTransformerConfig.__name__ == 'NonstationaryTransformerConfig'
assert 'torch_like.models.vision.vit' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_core_admission_did_not_mutate_grandfathered_registry():
    baseline = load_registry_baseline()
    assert len(baseline) == 161
    assert ("forecasting", "NonstationaryTransformer") not in baseline
    assert validate_document(Path("docs/MODEL_CANDIDATE_POOL.md")) == []


def test_future_target_values_never_enter_public_prediction():
    model = tl.NonstationaryTransformer(**_config()).eval()
    x = torch.randn(2, 8, 3)
    good = model.predict(model.make_forecast_batch(x))
    batch = model.make_forecast_batch(x)
    batch.x_dec.fill_(float("nan"))
    torch.testing.assert_close(model.predict(batch), good, rtol=0, atol=0)
