"""Source-mechanism tests for learned tau/delta raw-window factor projectors."""

import pytest
import torch

from torch_like.models.forecasting.nonstationary_transformer import _DeStationaryProjector


def test_conv_projects_time_as_channels_with_circular_feature_axis():
    """Catches accidentally interpreting the source Conv1d as a time convolution."""
    projector = _DeStationaryProjector(seq_len=2, channels=3, out_features=1, hidden=3).double()
    with torch.no_grad():
        projector.conv.weight.zero_()
        projector.conv.bias.zero_()
        projector.conv.weight[0, 0, 0] = 1.0
        projector.conv.weight[0, 1, 2] = 2.0
        first, second, final = projector.mlp[0], projector.mlp[2], projector.mlp[4]
        for layer in (first, second, final):
            layer.weight.zero_()
            layer.bias.zero_()
        first.weight[:3, :3] = torch.eye(3, dtype=torch.float64)
        second.weight.copy_(torch.eye(3, dtype=torch.float64))
        final.weight.fill_(1.0)

    raw = torch.tensor([[[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]], dtype=torch.float64)
    stats = torch.zeros(1, 1, 3, dtype=raw.dtype)
    # Circular feature neighborhoods: [30+2*2, 10+2*3, 20+2*1] = [34,16,22].
    torch.testing.assert_close(projector(raw, stats), torch.tensor([[72.0]], dtype=raw.dtype))
    with torch.no_grad():
        first.weight[0, 3] = 1.0
    stats[:, :, 0] = 7.0
    torch.testing.assert_close(projector(raw, stats), torch.tensor([[79.0]], dtype=raw.dtype))


@pytest.mark.parametrize("out_features", [1, 4, 9])
def test_factor_projector_returns_per_sample_scale_or_encoder_key_vector(out_features):
    torch.manual_seed(73)
    model = _DeStationaryProjector(seq_len=9, channels=4, out_features=out_features, hidden=8)
    raw = torch.randn(3, 9, 4)
    mean = raw.mean(dim=1, keepdim=True)
    output = model(raw, mean)
    assert output.shape == (3, out_features)
    assert torch.isfinite(output).all()


def test_projector_blocks_raw_and_stat_input_gradients_but_learns_parameters():
    torch.manual_seed(74)
    model = _DeStationaryProjector(seq_len=6, channels=3, out_features=6, hidden=8).double()
    raw = torch.randn(2, 6, 3, dtype=torch.float64, requires_grad=True)
    stats = torch.randn(2, 1, 3, dtype=torch.float64, requires_grad=True)
    output = model(raw, stats)
    loss = output.square().sum()
    loss.backward()
    assert raw.grad is None and stats.grad is None
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.conv.weight.grad.abs().sum() > 0
    assert model.mlp[0].weight.grad.abs().sum() > 0
    assert model.mlp[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "bad",
    [
        dict(seq_len=0, channels=3, out_features=1, hidden=8),
        dict(seq_len=6, channels=0, out_features=1, hidden=8),
        dict(seq_len=6, channels=3, out_features=0, hidden=8),
        dict(seq_len=6, channels=3, out_features=1, hidden=0),
        dict(seq_len=6, channels=3, out_features=1, hidden=True),
    ],
)
def test_projector_rejects_invalid_dimensions(bad):
    with pytest.raises(ValueError, match="seq_len|channels|out_features|hidden"):
        _DeStationaryProjector(**bad)


def test_projector_rejects_input_and_statistic_shape_mismatch():
    model = _DeStationaryProjector(seq_len=6, channels=3, out_features=1, hidden=8)
    with pytest.raises(ValueError, match="shape|stats"):
        model(torch.randn(2, 5, 3), torch.randn(2, 1, 3))
    with pytest.raises(ValueError, match="shape|stats"):
        model(torch.randn(2, 6, 3), torch.randn(2, 3))
