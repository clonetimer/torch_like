"""0.7 data/batch adapter and dataset-level evaluation regression tests."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from sklearn.metrics import f1_score

import torch_like as tl
from torch_like.data import DataLoaderConfig, TensorDataAdapter
from torch_like.evaluation import (
    ClassificationMetric,
    ForecastingMetric,
    FunctionalMetric,
    MetricEvaluator,
    RegressionMetric,
    forecasting_metrics,
)
from torch_like.training import ForecastingBatchAdapter, Trainer, TrainerConfig


class DataAdapterTests(unittest.TestCase):
    def test_tensor_data_adapter_preserves_dtype_indices_and_seed(self):
        x = np.arange(24, dtype=np.float32).reshape(12, 2)
        y = np.arange(12, dtype=np.int64)
        indices = np.array([1, 3, 5, 7, 9, 11])
        adapter = TensorDataAdapter()
        config = DataLoaderConfig(batch_size=2, shuffle=True, seed=123)
        loader_a = adapter.loader(x, y, indices=indices, config=config)
        loader_b = adapter.loader(x, y, indices=indices, config=config)
        order_a = torch.cat([batch_y for _, batch_y in loader_a])
        order_b = torch.cat([batch_y for _, batch_y in loader_b])
        torch.testing.assert_close(order_a, order_b)
        self.assertEqual(order_a.dtype, torch.int64)
        self.assertEqual(set(order_a.tolist()), set(y[indices].tolist()))

    def test_tensor_data_adapter_accepts_explicit_distributed_sampler(self):
        x = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        y = torch.arange(8)
        adapter = TensorDataAdapter()
        dataset = adapter.dataset(x, y)
        context = tl.training.DistributedContext(rank=1, world_size=2)
        sampler = tl.training.create_distributed_sampler(
            dataset, context, tl.training.DistributedSamplerConfig(shuffle=False)
        )
        loader = adapter.loader_from_dataset(
            dataset,
            config=DataLoaderConfig(batch_size=2, shuffle=False),
            sampler=sampler,
        )
        observed = torch.cat([batch_y for _, batch_y in loader]).tolist()
        self.assertEqual(observed, [1, 3, 5, 7])
        with self.assertRaisesRegex(ValueError, "shuffle"):
            adapter.loader_from_dataset(
                dataset,
                config=DataLoaderConfig(batch_size=2, shuffle=True),
                sampler=sampler,
            )

    def test_forecasting_adapter_keeps_target_separate_and_prevents_future_leakage(self):
        model = tl.DLinear(enc_in=1, dec_in=1, c_out=1, seq_len=4, label_len=2, pred_len=2, moving_avg=3)
        x_enc = torch.randn(3, 4, 1)
        decoder_target = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)
        raw = {
            "x_enc": x_enc,
            "x_dec_target": decoder_target,
            "x_mark_enc": torch.zeros(3, 4, 4),
            "x_mark_dec": torch.zeros(3, 4, 4),
            "x_target_enc": None,
            "x_exog_enc": None,
            "x_target_dec": None,
            "x_exog_dec": None,
        }
        adapted = ForecastingBatchAdapter().adapt(model, raw)
        torch.testing.assert_close(adapted.model_batch.x_dec[:, :2], decoder_target[:, :2])
        torch.testing.assert_close(adapted.model_batch.x_dec[:, 2:], torch.zeros_like(decoder_target[:, 2:]))
        torch.testing.assert_close(adapted.targets, decoder_target[:, -2:])
        self.assertEqual(adapted.resolved_sample_count(), 3)


    def test_forecasting_adapter_accepts_legacy_four_tensor_batches(self):
        model = tl.DLinear(enc_in=1, dec_in=1, c_out=1, seq_len=4, label_len=2, pred_len=2, moving_avg=3)
        x_enc = torch.randn(2, 4, 1)
        decoder_target = torch.randn(2, 4, 1)
        x_mark_enc = torch.zeros(2, 4, 4)
        x_mark_dec = torch.zeros(2, 4, 4)
        adapted = ForecastingBatchAdapter().adapt(
            model,
            (x_enc, decoder_target, x_mark_enc, x_mark_dec),
        )
        torch.testing.assert_close(adapted.model_batch.x_mark_enc, x_mark_enc)
        torch.testing.assert_close(adapted.model_batch.x_mark_dec, x_mark_dec)
        torch.testing.assert_close(adapted.targets, decoder_target[:, -2:])

class EvaluationTests(unittest.TestCase):
    def test_metric_evaluator_computes_global_f1_not_mean_of_batch_f1(self):
        evaluator = MetricEvaluator({"f1": lambda y, p: f1_score(y, p, zero_division=0)})
        target_a = np.array([1, 1])
        pred_a = np.array([1, 0])
        target_b = np.array([0, 0])
        pred_b = np.array([1, 0])
        evaluator.update(pred_a, target_a)
        evaluator.update(pred_b, target_b)
        expected = f1_score(np.concatenate([target_a, target_b]), np.concatenate([pred_a, pred_b]), zero_division=0)
        batch_mean = (f1_score(target_a, pred_a, zero_division=0) + f1_score(target_b, pred_b, zero_division=0)) / 2
        self.assertAlmostEqual(evaluator.compute()["f1"], expected)
        self.assertNotAlmostEqual(expected, batch_mean)

    def test_forecasting_metrics_match_expected_shapes(self):
        truth = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
        prediction = truth + 1.0
        metrics = forecasting_metrics(truth, prediction)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertIn("horizon_3_mae", metrics)

    def test_streaming_regression_matches_exact_and_merges_shards(self):
        truth = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 9.0], [8.0, 10.0]])
        prediction = truth + np.array([[0.5, -1.0], [1.0, 0.0], [-0.5, 2.0], [0.0, -1.0]])
        for name in ("mae", "rmse", "r2"):
            whole = MetricEvaluator({name: RegressionMetric(name)})
            left = MetricEvaluator({name: RegressionMetric(name)})
            right = MetricEvaluator({name: RegressionMetric(name)})
            whole.update(prediction, truth)
            left.update(prediction[:2], truth[:2])
            right.update(prediction[2:], truth[2:])
            left.merge_state(right.state_dict())
            self.assertTrue(whole.fully_streaming)
            self.assertAlmostEqual(left.compute()[name], whole.compute()[name], places=12)

    def test_streaming_forecasting_matches_exact_metrics_across_batches(self):
        truth = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
        prediction = truth + np.array([0.5, -1.0])[None, None, :]
        evaluator = MetricEvaluator(
            {
                "mae": ForecastingMetric("mae"),
                "rmse": ForecastingMetric("rmse"),
                "smape": ForecastingMetric("smape"),
                "horizon_2_mae": ForecastingMetric("mae", horizon=1),
            }
        )
        evaluator.update(prediction[:1], truth[:1])
        evaluator.update(prediction[1:], truth[1:])
        exact = forecasting_metrics(truth, prediction)
        actual = evaluator.compute()
        self.assertTrue(evaluator.fully_streaming)
        self.assertAlmostEqual(actual["mae"], exact["mae"], places=12)
        self.assertAlmostEqual(actual["rmse"], exact["rmse"], places=12)
        self.assertAlmostEqual(actual["smape"], exact["smape"], places=12)
        self.assertAlmostEqual(actual["horizon_2_mae"], exact["horizon_2_mae"], places=12)

    def test_streaming_classification_matches_sklearn_and_merges(self):
        truth = np.array([0, 1, 2, 2, 1, 0, 2, 1])
        prediction = np.array([0, 2, 2, 1, 1, 0, 0, 1])
        metrics = {
            name: ClassificationMetric(name)
            for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        }
        evaluator = MetricEvaluator(metrics)
        evaluator.update(prediction[:3], truth[:3])
        shard = MetricEvaluator(
            {name: ClassificationMetric(name) for name in metrics}
        )
        shard.update(prediction[3:], truth[3:])
        evaluator.merge(shard)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        expected = {
            "accuracy": accuracy_score(truth, prediction),
            "precision_macro": precision_score(truth, prediction, average="macro", zero_division=0),
            "recall_macro": recall_score(truth, prediction, average="macro", zero_division=0),
            "f1_macro": f1_score(truth, prediction, average="macro", zero_division=0),
        }
        for name, value in evaluator.compute().items():
            self.assertAlmostEqual(value, expected[name], places=12)

    def test_functional_metric_remains_exact_buffered_fallback(self):
        metric = FunctionalMetric(lambda y, p: float(np.max(np.abs(y - p))))
        evaluator = MetricEvaluator({"max_error": metric})
        evaluator.update(np.array([1.0, 5.0]), np.array([0.0, 3.0]))
        evaluator.update(np.array([2.0]), np.array([10.0]))
        self.assertFalse(evaluator.fully_streaming)
        self.assertAlmostEqual(evaluator.compute()["max_error"], 8.0)
        state = evaluator.state_dict()
        restored = MetricEvaluator({"max_error": FunctionalMetric(lambda y, p: float(np.max(np.abs(y - p))))})
        restored.load_state_dict(state)
        self.assertAlmostEqual(restored.compute()["max_error"], 8.0)

    def test_trainer_auto_adapts_supervised_tabular_batches_and_uses_dataset_metrics(self):
        torch.manual_seed(5)
        model = tl.FTTransformer(
            input_dim=2,
            output_dim=2,
            d_model=4,
            n_heads=1,
            n_layers=1,
            d_ff=8,
            dropout=0.0,
        )
        x = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        y = torch.tensor([0, 0, 1, 1])
        loader = TensorDataAdapter().loader(x, y, config=DataLoaderConfig(batch_size=2, shuffle=False))
        trainer = Trainer(
            model,
            TrainerConfig(epochs=1, device="cpu", learning_rate=1e-2),
            metrics={"accuracy": lambda truth, prediction: np.mean(truth == prediction)},
            prediction_fn=lambda current, batch: current.predict(batch).argmax(dim=-1),
        )
        result = trainer.fit(loader, loader)
        accuracy = result.history[-1].validation_metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


    def test_forecasting_adapter_can_drive_trainer_with_explicit_step_fn(self):
        torch.manual_seed(7)
        model = tl.DLinear(enc_in=1, dec_in=1, c_out=1, seq_len=4, label_len=2, pred_len=2, moving_avg=3)
        raw = {
            "x_enc": torch.randn(4, 4, 1),
            "x_dec_target": torch.randn(4, 4, 1),
            "x_mark_enc": torch.zeros(4, 4, 4),
            "x_mark_dec": torch.zeros(4, 4, 4),
            "x_target_enc": None,
            "x_exog_enc": None,
            "x_target_dec": None,
            "x_exog_dec": None,
        }
        trainer = Trainer(
            model,
            TrainerConfig(epochs=1, device="cpu"),
            batch_adapter=ForecastingBatchAdapter(),
            step_fn=lambda current, batch: torch.nn.functional.mse_loss(
                current.predict(batch.model_batch), batch.targets
            ),
            metrics={
                "mae": lambda truth, prediction: float(np.mean(np.abs(truth - prediction))),
            },
        )
        result = trainer.fit([raw], [raw])
        self.assertEqual(result.epochs_ran, 1)
        self.assertIn("mae", result.history[0].validation_metrics)

if __name__ == "__main__":
    unittest.main()
