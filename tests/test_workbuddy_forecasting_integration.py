"""Public API acceptance for the first Workbuddy literature-intake batch."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import torch_like as tl
from tools.validate_candidate_pool import load_registry_baseline, validate_document


def _config(name):
    cfg = dict(enc_in=3, dec_in=3, c_out=3, seq_len=16, label_len=0, pred_len=5, dropout=0.0)
    if name == "FITS":
        cfg.update(cut_freq=4, individual=True)
    else:
        cfg.update(
            d_model=8,
            e_layers=1,
            bottleneck_channels=4,
            adjacency=[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        )
    return cfg


@pytest.mark.parametrize("name", ["FITS", "STGCN"])
def test_public_factory_contract_and_runtime(name):
    torch.manual_seed(71)
    model = tl.create(name, backend="torch_like", **_config(name)).eval()
    assert type(model) is getattr(tl, name)
    assert name in tl.list_models("forecasting")
    spec = tl.describe(name)
    assert spec.task.kind == tl.TaskKind.TIME_SERIES_FORECASTING
    assert spec.supports("predict")
    runtime = tl.as_runtime(model)
    x = torch.randn(2, 16, 3)
    batch = runtime.make_batch(x_enc=x, x_mark_enc=None, x_dec=torch.zeros(2, 5, 3), x_mark_dec=None)
    actual = runtime.predict(batch)
    assert actual.shape == (2, 5, 3)
    torch.testing.assert_close(actual, model(x, None, None, None))


@pytest.mark.parametrize("name", ["FITS", "STGCN"])
def test_public_save_load_preserves_config_graph_and_prediction(name):
    torch.manual_seed(72)
    model = tl.create_model(name, domain="forecasting", **_config(name)).eval()
    x = torch.randn(2, 16, 3)
    expected = model.predict(model.make_forecast_batch(x))
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / name
        tl.save_model(model, target)
        restored = tl.load_model(target).eval()
        assert restored.config() == model.config()
        torch.testing.assert_close(restored.predict(restored.make_forecast_batch(x)), expected)
        assert set(restored.state_dict()) == set(model.state_dict())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize("name", ["FITS", "STGCN"])
def test_dataclass_configs_and_dtype_roundtrip(name):
    config_type = getattr(tl, name + "Config")
    cfg = config_type(**_config(name))
    model = getattr(tl, name).from_config(cfg).double().eval()
    restored = type(model).from_config(model.config()).double().eval()
    restored.load_state_dict(model.state_dict())
    x = torch.randn(2, 16, 3, dtype=torch.float64)
    expected = model.predict(model.make_forecast_batch(x))
    assert expected.dtype == torch.float64
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(restored.predict(restored.make_forecast_batch(x)), expected)


@pytest.mark.parametrize("name", ["FITS", "STGCN"])
def test_no_future_decoder_target_dependency(name):
    model = tl.create_model(name, **_config(name)).eval()
    x = torch.randn(2, 16, 3)
    empty = model(x, None, None, None)
    poisoned = model(
        x,
        torch.full((2, 16, 4), float("nan")),
        torch.full((2, 5, 3), float("nan")),
        torch.full((2, 5, 4), float("nan")),
    )
    torch.testing.assert_close(empty, poisoned)


def test_intake_registration_is_approved_without_inflating_evidence():
    assert validate_document(Path("docs/MODEL_CANDIDATE_POOL.md")) == []
    baseline = load_registry_baseline()
    assert len(baseline) == 161
    for name in ("FITS", "STGCN"):
        assert name not in baseline
        info = tl.get_model_info(name, domain="forecasting")
        assert info["maturity"] == "unassessed"
        assert info["benchmark_evidence"] == "unassessed"
    assert tl.get_model_info("FITS")["integration_level"] == "paper_aligned"
    assert tl.get_model_info("STGCN")["integration_level"] == "framework_oriented"


def test_new_public_exports_remain_lazy():
    code = """
import sys
import torch_like as tl
assert 'torch_like.models.forecasting.fits' not in sys.modules
assert 'torch_like.models.forecasting.stgcn' not in sys.modules
assert tl.FITS.__name__ == 'FITS'
assert 'torch_like.models.forecasting.stgcn' not in sys.modules
assert 'torch_like.models.vision.vit' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
