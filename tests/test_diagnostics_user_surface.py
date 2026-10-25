"""Cross-family algorithm-interface smoke without data-platform integration.

Statistical, sklearn ML, health, fault-classification and native PyTorch RUL
algorithms expose typed DiagnosticResult values. Direct NN training remains
native torch, not an invented common fit(X,y) protocol.
"""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from torch_like.diagnostics import (
    CUSUMDetector,
    DiagnosticPipeline,
    DiagnosticRegistry,
    DiagnosticResult,
    DiagnosticStage,
    FaultLabelSchema,
    GaussianHealthIndex,
    RULRegressor,
    SignalBatch,
    SklearnWindowAnomalyDetector,
    SklearnWindowFaultClassifier,
    ZScoreDetector,
    register_statistical_detectors,
)


class _ReadyRUL(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.constant = torch.nn.Parameter(torch.tensor(5.0))

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        return self.constant.expand(x.shape[0])


def _batch(values: np.ndarray) -> SignalBatch:
    return SignalBatch(
        np.asarray(values, dtype=np.float64),
        sample_rate_hz=100.0,
        channel_names=("wheel", "temperature"),
        units=("rad/s", "C"),
    )


class UserSurfaceTests(unittest.TestCase):
    @staticmethod
    def fixtures() -> tuple[SignalBatch, SignalBatch, SignalBatch]:
        base = np.array([[-1.0, -0.5], [0.0, 0.0], [1.0, 0.5], [0.5, -1.0]])
        reference = np.stack((base, base * 0.8, base * 1.2), axis=0)
        query = np.stack((base, base + 3.0), axis=0)
        labeled = np.concatenate((reference, reference + 4.0), axis=0)
        return _batch(reference), _batch(query), _batch(labeled)

    def test_one_input_batch_and_explicit_typed_result_across_families(self) -> None:
        reference, query, labeled = self.fixtures()

        registry = DiagnosticRegistry()
        register_statistical_detectors(registry)
        # CUSUM >=0 by mathematical definition and class constructor.
        model = registry.create("CUSUMDetector", k=0, h=3.0).fit(reference)
        self.assertIsInstance(model, CUSUMDetector)
        description = registry.get_spec("CUSUMDetector").to_dict()
        self.assertEqual(json.loads(json.dumps(description))["parameters"][0]["minimum"], 0.0)
        anomaly = model.score(query)
        self.assertEqual(anomaly.output_kind.value, "anomaly_score")
        self.assertEqual(anomaly.values.shape, (2, 4, 2))

        sklearn_anomaly = SklearnWindowAnomalyDetector(n_estimators=8, random_state=7).fit(reference).score(query)
        health = GaussianHealthIndex().fit(reference).score(query)
        fault = SklearnWindowFaultClassifier(n_estimators=8, random_state=7).fit(
            labeled,
            ("normal", "normal", "normal", "fault", "fault", "fault"),
            FaultLabelSchema(("normal", "fault")),
        ).predict_proba(query)
        rul = RULRegressor(_ReadyRUL(), unit="hours", ready=True).predict_with_unit(query)
        self.assertEqual(rul.unit, "hours")

        results = (anomaly, sklearn_anomaly, health, fault, rul.result)
        self.assertTrue(all(isinstance(result, DiagnosticResult) for result in results))
        self.assertTrue(all(result.sample_count == 2 for result in results))
        self.assertEqual(
            tuple(result.task.value for result in results),
            ("anomaly_detection", "anomaly_detection", "health_assessment",
             "fault_classification", "rul_regression"),
        )
        self.assertEqual(health.values.shape, (2,))
        self.assertEqual(fault.values.shape, (2, 2))
        self.assertEqual(rul.values.shape, (2,))
        np.testing.assert_allclose(fault.values.sum(axis=1), np.ones(2))
        self.assertTrue(np.isfinite(np.concatenate([x.values.reshape(-1) for x in results])).all())

    def test_pipeline_preserves_scoring_semantics_without_silent_fusion(self) -> None:
        reference, query, _ = self.fixtures()
        pipeline = DiagnosticPipeline(
            DiagnosticStage("anomaly", ZScoreDetector().fit(reference)),
            DiagnosticStage("health", GaussianHealthIndex().fit(reference)),
        )
        results = pipeline.predict(query)
        self.assertEqual(set(results.results), {"anomaly", "health"})
        self.assertEqual(results["anomaly"].values.shape, (2, 4, 2))
        self.assertEqual(results["health"].values.shape, (2,))
        standalone = ZScoreDetector().fit(reference).score(query)
        np.testing.assert_array_equal(results["anomaly"].values, standalone.values)

    def test_no_implicit_rul_ready_or_negative_policy(self) -> None:
        _, query, _ = self.fixtures()
        regressor = RULRegressor(_ReadyRUL(), unit="cycles")
        with self.assertRaisesRegex(RuntimeError, "fitted|ready"):
            regressor.predict(query)
        regressor.mark_ready()
        forecast = regressor.predict_with_unit(query)
        self.assertEqual(forecast.unit, "cycles")
        np.testing.assert_allclose(forecast.values, [5.0, 5.0])


if __name__ == "__main__":
    unittest.main()
