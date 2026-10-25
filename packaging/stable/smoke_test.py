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
    AnomalyScoreProjection,
    DecisionMode,
    DiagnosticDecision,
    DiagnosticParameterSpec,
    DiagnosticPipeline,
    DiagnosticRegistry,
    DiagnosticStage,
    FaultLabelSchema,
    GaussianHealthIndex,
    MahalanobisHealthIndex,
    RULRegressor,
    SignalBatch,
    StreamingThreshold,
    SklearnWindowFaultClassifier,
    ZScoreDetector,
    create_rul_regressor,
    list_rul_models,
    project_anomaly_scores,
    register_statistical_detectors,
)
from torch_like_stable.models import ROCKET, TCNRUL, InceptionTime


def make(values: np.ndarray) -> SignalBatch:
    return SignalBatch(
        values=values,
        sample_rate_hz=100,
        channel_names=("vibration", "temperature"),
        units=("g", "C"),
    )


def main() -> None:
    assert importlib.metadata.version("torch-like-algorithms") == tl.__version__ == "0.3.0"
    assert importlib.util.find_spec("torch_like_stable.automl") is None
    assert importlib.util.find_spec("torch_like_stable.training") is None
    assert importlib.util.find_spec("torch_like_stable.evaluation") is None
    assert importlib.util.find_spec("torch_like_stable.data") is None
    assert importlib.util.find_spec("torch_like_stable.diagnostics.workflow") is None
    assert importlib.util.find_spec("torch_like_stable.diagnostics.selection") is None
    assert importlib.util.find_spec("torch_like_stable.models.forecasting") is None

    baseline = make(
        np.array(
            [
                [[-1, -1], [0, 0], [1, 1], [0, 0]],
                [[-1, -1], [0, 0], [1, 1], [0, 0]],
            ],
            dtype=float,
        )
    )
    query = make(np.array([[[0, 0], [0, 0], [0, 0], [0, 0]], [[4, 4], [4, 4], [4, 4], [4, 4]]], dtype=float))
    for model in (GaussianHealthIndex(), MahalanobisHealthIndex()):
        values = model.fit(baseline).score(query).values
        assert values.shape == (2,) and values[0] > values[1]
        assert np.isfinite(values).all()
    detector = ZScoreDetector().fit(baseline)
    out = detector.score(query)
    assert out.values.shape == (2, 4, 2)
    schema = DiagnosticParameterSpec("sigma_threshold", "number", default=3.0, minimum=0.0,
                                     exclusive_minimum=True)
    assert schema.validate(3.0) == 3.0
    registry = DiagnosticRegistry()
    register_statistical_detectors(registry)
    assert registry.create("CUSUMDetector", k=0).k == 0.0
    assert registry.get_spec("CUSUMDetector").validate_parameters({"k": 0})["k"] == 0
    decision = DiagnosticDecision.from_threshold(out, 3.0)
    assert decision.decision_mode == DecisionMode.SCORE_THRESHOLD
    assert decision.labels.shape == out.values.shape
    segment = detector.score(make(query.values[:1]))
    stream = StreamingThreshold(
        algorithm_name="ZScoreDetector", mode="fixed", thresholds=(3.0, 3.0),
        channel_names=("vibration", "temperature"),
    )
    stream_decision = stream.decide_chunk(
        segment, start_index=0, channel_names=("vibration", "temperature"),
    )
    assert stream_decision.labels.shape == segment.values.shape
    assert stream.observations == 4
    snapshot = stream.state_dict()
    resumed = StreamingThreshold(
        algorithm_name="ZScoreDetector", mode="fixed", thresholds=(3.0, 3.0),
        channel_names=("vibration", "temperature"),
    )
    resumed.load_state_dict(snapshot)
    assert resumed.observations == 4
    projected = project_anomaly_scores(
        segment, score_layout="time_channel", sequence_mode="continuous", channel_reduction="max",
    )
    assert isinstance(projected, AnomalyScoreProjection)
    assert projected.resolution == "time" and projected.values.shape == (4,)
    composition = DiagnosticPipeline(
        DiagnosticStage("health", GaussianHealthIndex().fit(baseline)),
        DiagnosticStage("anomaly", detector),
    ).predict(query)
    assert set(composition.results) == {"health", "anomaly"}
    assert "TCNRUL" in list_rul_models()
    assert "THGNN" not in list_rul_models()

    # Native sklearn classifier and PyTorch network construction, no platform.
    labels = FaultLabelSchema(("normal", "fault"))
    train = make(
        np.concatenate(
            [
                np.zeros((3, 8, 2)),
                np.full((3, 8, 2), 5.0),
            ]
        )
    )
    classifier = SklearnWindowFaultClassifier(n_estimators=8, random_state=3)
    clf_result = classifier.fit(
        train,
        ("normal", "normal", "normal", "fault", "fault", "fault"),
        labels,
    ).predict_proba(train)
    assert clf_result.values.shape == (6, 2)
    assert np.allclose(clf_result.values.sum(axis=1), 1)

    model = InceptionTime(
        input_dim=2,
        num_classes=2,
        seq_len=8,
        d_model=8,
        num_blocks=1,
        bottleneck_dim=4,
        kernel_sizes=(3,),
    )
    logits = model(torch.zeros((2, 8, 2)))
    assert logits.shape == (2, 2)

    random_features_model = ROCKET(
        input_dim=2,
        num_classes=2,
        seq_len=16,
        num_kernels=4,
        kernel_sizes=(3,),
    )
    assert random_features_model(torch.randn(2, 16, 2)).shape == (2, 2)

    rul = TCNRUL(enc_in=2, seq_len=8, d_model=8, e_layers=2)
    assert rul(torch.randn(2, 8, 2)).shape == (2,)
    adapter = RULRegressor(rul, unit="cycles", negative_policy="clip", ready=True)
    assert adapter.predict_with_unit(make(np.zeros((2, 8, 2)))).values.shape == (2,)
    another = create_rul_regressor(
        "TCNRUL",
        unit="hours",
        negative_policy="clip",
        enc_in=2,
        seq_len=8,
        d_model=8,
        e_layers=1,
        d_ff=16,
    )
    assert type(another.model).__name__ == "TCNRUL"
    print("PASS: isolated algorithm wheel contracts, health, sklearn, torch and RUL")


if __name__ == "__main__":
    main()
