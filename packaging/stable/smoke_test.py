"""Isolated installed-wheel smoke checks for the algorithm-only distribution.

Execute from a working directory outside the repository, with the standalone
wheel installed and optional full torch_like not needed.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util

import numpy as np
import torch

import torch_like_stable as tl
from torch_like_stable.diagnostics import (
    FaultLabelSchema,
    GaussianHealthIndex,
    MahalanobisHealthIndex,
    SignalBatch,
    SklearnWindowFaultClassifier,
    ZScoreDetector,
)
from torch_like_stable.models import InceptionTime, ROCKET, TCNRUL


def make(values: np.ndarray) -> SignalBatch:
    return SignalBatch(
        values=values,
        sample_rate_hz=100,
        channel_names=("vibration", "temperature"),
        units=("g", "C"),
    )


def main() -> None:
    assert importlib.metadata.version("torch-like-algorithms") == tl.__version__ == "0.1.0"
    assert importlib.util.find_spec("torch_like_stable.automl") is None
    assert importlib.util.find_spec("torch_like_stable.training") is None
    assert importlib.util.find_spec("torch_like_stable.evaluation") is None
    assert importlib.util.find_spec("torch_like_stable.data") is None
    assert importlib.util.find_spec("torch_like_stable.diagnostics.workflow") is None
    assert importlib.util.find_spec("torch_like_stable.diagnostics.selection") is None
    assert importlib.util.find_spec("torch_like_stable.models.forecasting") is None

    baseline = make(np.array([
        [[-1, -1], [0, 0], [1, 1], [0, 0]],
        [[-1, -1], [0, 0], [1, 1], [0, 0]],
    ], dtype=float))
    query = make(np.array([[[0, 0], [0, 0], [0, 0], [0, 0]],
                           [[4, 4], [4, 4], [4, 4], [4, 4]]], dtype=float))
    for model in (GaussianHealthIndex(), MahalanobisHealthIndex()):
        values = model.fit(baseline).score(query).values
        assert values.shape == (2,) and values[0] > values[1]
        assert np.isfinite(values).all()
    out = ZScoreDetector().fit(baseline).score(query)
    assert out.values.shape == (2, 4, 2)

    # Native sklearn classifier and PyTorch network construction, no platform.
    labels = FaultLabelSchema(("normal", "fault"))
    train = make(np.concatenate([
        np.zeros((3, 8, 2)), np.full((3, 8, 2), 5.0),
    ]))
    classifier = SklearnWindowFaultClassifier(n_estimators=8, random_state=3)
    clf_result = classifier.fit(
        train,
        ("normal", "normal", "normal", "fault", "fault", "fault"),
        labels,
    ).predict_proba(train)
    assert clf_result.values.shape == (6, 2)
    assert np.allclose(clf_result.values.sum(axis=1), 1)

    model = InceptionTime(
        input_dim=2, num_classes=2, seq_len=8, d_model=8,
        num_blocks=1, bottleneck_dim=4, kernel_sizes=(3,),
    )
    logits = model(torch.zeros((2, 8, 2)))
    assert logits.shape == (2, 2)

    random_features_model = ROCKET(
        input_dim=2, num_classes=2, seq_len=16, num_kernels=4, kernel_sizes=(3,),
    )
    assert random_features_model(torch.randn(2, 16, 2)).shape == (2, 2)

    rul = TCNRUL(enc_in=2, seq_len=8, d_model=8, e_layers=2)
    assert rul(torch.randn(2, 8, 2)).shape == (2,)
    print("PASS: isolated algorithm wheel contracts, health, sklearn, torch and RUL")


if __name__ == "__main__":
    main()
