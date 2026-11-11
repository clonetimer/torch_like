"""Knowledge-data PINN RUL model and adapter contract tests."""

import unittest

import numpy as np
import torch

from torch_like.diagnostics import SignalBatch
from torch_like.diagnostics.rul_interface import RULRegressor
from torch_like.models.knowledge_data_pinn import KnowledgeDataPINNRUL


def make_signals(batch=6, *, steps=10, channels=3):
    rng = np.random.default_rng(31)
    values = rng.normal(size=(batch, steps, channels)).astype(np.float32)
    time = np.linspace(0.0, 1.0, steps, dtype=np.float32)[None, :, None]
    values += 0.2 * time
    return SignalBatch(
        values,
        sample_rate_hz=1.0,
        channel_names=tuple(f"feature_{index}" for index in range(channels)),
        units=tuple("normalized" for _ in range(channels)),
    )


def make_times(batch, steps):
    times = np.linspace(0.0, 1.0, steps, dtype=np.float32)[None, :, None]
    return np.repeat(times, batch, axis=0)


def small_model(**kwargs):
    settings = {
        "enc_in": 3,
        "seq_len": 10,
        "hidden_dim": 8,
        "hidden_layers": 2,
        "physics_weight": 0.2,
        "monotonicity_weight": 0.3,
    }
    settings.update(kwargs)
    return KnowledgeDataPINNRUL(**settings)


class KnowledgeDataPINNRULTests(unittest.TestCase):
    def test_forward_and_trajectory_shapes_are_finite_and_nonnegative(self):
        model = small_model()
        model.eval()
        signals = make_signals()
        x = torch.as_tensor(signals.values)
        times = torch.as_tensor(make_times(6, 10))

        prediction = model(x, times)
        trajectory = model.predict_trajectory(x, times)

        self.assertEqual(prediction.shape, (6,))
        self.assertEqual(trajectory.shape, (6, 10, 1))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(trajectory).all())
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(trajectory >= 0))

    def test_physics_residual_and_all_loss_components_are_finite(self):
        model = small_model()
        x = torch.as_tensor(make_signals().values)
        times = torch.as_tensor(make_times(6, 10))
        targets = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

        residual = model.physics_residual(x, times)
        components = model.loss_components(x, targets, x_mark=times)

        self.assertEqual(residual.shape, (6, 10, 1))
        self.assertTrue(torch.isfinite(residual).all())
        self.assertEqual(set(components), {"total", "data", "physics", "monotonicity"})
        for value in components.values():
            self.assertEqual(value.ndim, 0)
            self.assertTrue(torch.isfinite(value))

    def test_training_loss_backpropagates_through_state_and_rate_networks(self):
        model = small_model()
        model.train()
        x = torch.as_tensor(make_signals().values)
        times = torch.as_tensor(make_times(6, 10))
        targets = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

        objective = model.training_loss(x, targets, x_mark=times, loss="mse")
        self.assertEqual(objective.ndim, 0)
        self.assertTrue(torch.isfinite(objective))
        objective.backward()

        state_grad = model.state_net[0].weight.grad
        rate_grad = model.rate_net[0].weight.grad
        self.assertIsNotNone(state_grad)
        self.assertIsNotNone(rate_grad)
        self.assertTrue(torch.isfinite(state_grad).all())
        self.assertTrue(torch.isfinite(rate_grad).all())
        self.assertGreater(float(rate_grad.abs().sum()), 0.0)

    def test_rul_regressor_trains_and_predicts_with_explicit_time_coordinates(self):
        signals = make_signals()
        times = make_times(6, 10)
        targets = np.array([90.0, 80.0, 70.0, 60.0, 50.0, 40.0], dtype=np.float32)
        model = small_model()
        regressor = RULRegressor(model, unit="cycles")

        regressor.fit(
            signals,
            targets,
            x_mark=times,
            epochs=2,
            batch_size=3,
            learning_rate=1e-3,
            seed=17,
        )
        self.assertTrue(regressor.is_ready)
        result = regressor.predict(signals, x_mark=times)
        self.assertEqual(result.values.shape, (6,))
        self.assertTrue(np.isfinite(result.values).all())
        self.assertTrue(np.all(result.values >= 0))

    def test_default_time_grid_and_loss_functions(self):
        model = small_model()
        x = torch.as_tensor(make_signals().values)
        targets = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

        for loss in ("mse", "mae", "huber"):
            with self.subTest(loss=loss):
                objective = model.training_loss(x, targets, loss=loss)
                self.assertTrue(torch.isfinite(objective))

    def test_rejects_invalid_time_shapes_order_and_input_metadata(self):
        model = small_model()
        x = torch.as_tensor(make_signals().values)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            model(x, torch.zeros((6, 10, 1)))
        with self.assertRaisesRegex(ValueError, "shape"):
            model(x, torch.zeros((6, 10, 2)))
        with self.assertRaisesRegex(ValueError, "features"):
            model(torch.ones((6, 10, 4)))
        with self.assertRaisesRegex(ValueError, "at least two"):
            KnowledgeDataPINNRUL(enc_in=3, seq_len=1)

    def test_rejects_invalid_weights_and_nonfinite_values(self):
        for kwargs in (
            {"physics_weight": -1.0},
            {"monotonicity_weight": -0.1},
            {"min_rate": 0.0},
            {"dropout": 1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                small_model(**kwargs)

        model = small_model()
        x = torch.as_tensor(make_signals().values)
        x[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model(x)


if __name__ == "__main__":
    unittest.main()
