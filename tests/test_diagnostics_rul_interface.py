"""R5c unified RUL interface for available native model families."""

import unittest

import numpy as np
import torch

from torch_like.diagnostics import (
    DiagnosticPipeline,
    DiagnosticStage,
    RULRegressor,
    SignalBatch,
    create_rul_regressor,
    list_rul_models,
)
from torch_like.diagnostics.rul_interface import RULForecast
from torch_like.models.rul_backbones import TCNRUL


def signals(x, *, units=("g", "C"), rate=100.0, names=("vib", "temp")):
    return SignalBatch(np.asarray(x, dtype=float), sample_rate_hz=rate, channel_names=names, units=units)


class TinyRUL(torch.nn.Module):
    def __init__(self, channels=2, length=8, bias=2.0):
        super().__init__()
        self.enc_in = channels
        self.seq_len = length
        self.bias = torch.nn.Parameter(torch.tensor(float(bias)))

    def forward(self, x, x_mark=None):
        return x.mean(dim=(1, 2)) + self.bias


class RULInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.train = signals(np.arange(48, dtype=float).reshape(3, 8, 2) / 40)
        self.input = signals(np.zeros((2, 8, 2), dtype=float))

    def test_direct_native_wrapper_training_preserves_model_identity_and_units(self):
        native = TinyRUL()
        model = RULRegressor(native, unit="cycles")
        self.assertIs(model.model, native)
        with self.assertRaisesRegex(RuntimeError, "fitted or explicitly"):
            model.predict(self.input)
        self.assertIs(model.fit(self.train, np.array([5.0, 3.0, 1.0]), epochs=2, batch_size=2), model)
        self.assertTrue(model.is_ready)
        prediction = model.predict(self.input)
        self.assertEqual(prediction.task.value, "rul_regression")
        self.assertEqual(prediction.output_kind.value, "rul_value")
        self.assertEqual(prediction.algorithm_name, "TinyRUL")
        self.assertEqual(prediction.values.shape, (2,))
        self.assertTrue(np.all(prediction.values >= 0))
        with_unit = model.predict_with_unit(self.input)
        self.assertIsInstance(with_unit, RULForecast)
        self.assertEqual(with_unit.unit, "cycles")
        np.testing.assert_array_equal(with_unit.values, prediction.values)
        self.assertFalse(with_unit.values.flags.writeable)
        self.assertEqual(
            DiagnosticPipeline(DiagnosticStage("rul", model, method="predict")).predict(self.input)["rul"].task.value,
            "rul_regression",
        )

    def test_readiness_for_pretrained_weights_is_explicit(self):
        m = RULRegressor(TinyRUL(), unit="hours", ready=True)
        np.testing.assert_allclose(m.predict_with_unit(self.input).values, [2, 2])
        self.assertEqual(m.unit, "hours")
        self.assertTrue(RULRegressor(TinyRUL(), unit="seconds").mark_ready().is_ready)

    def test_negative_estimates_rejected_or_clipped_by_explicit_policy(self):
        native = TinyRUL(bias=-2.0)
        with self.assertRaisesRegex(ValueError, "negative RUL"):
            RULRegressor(native, unit="hours", ready=True).predict(self.input)
        clipped = RULRegressor(native, unit="hours", negative_policy="clip", ready=True).predict(self.input)
        np.testing.assert_array_equal(clipped.values, [0, 0])

    def test_factory_exposes_all_installed_native_models_without_rewriting_class(self):
        names = list_rul_models()
        self.assertTrue({"TCNRUL", "PatchTSTRUL", "TransformerRUL"}.issubset(names))
        self.assertTrue({"THGNN", "InformerRUL", "AutoformerRUL", "LogTransRUL"}.issubset(names))
        adapter = create_rul_regressor(
            "TCNRUL",
            unit="cycles",
            negative_policy="clip",
            enc_in=2,
            seq_len=8,
            d_model=8,
            e_layers=1,
            d_ff=16,
            kernel_size=3,
        )
        self.assertIsInstance(adapter.model, TCNRUL)
        self.assertFalse(adapter.is_ready)
        adapter.fit(self.train, np.array([2.0, 2.0, 2.0]), epochs=1, batch_size=2)
        self.assertEqual(adapter.predict(self.input).values.shape, (2,))
        with self.assertRaisesRegex(KeyError, "unknown RUL"):
            create_rul_regressor("MadeUpRUL", unit="cycles")

    def test_other_native_backbones_are_accepted_with_no_forked_model_api(self):
        for name, kwargs in (
            ("PatchTSTRUL", {"patch_len": 4, "patch_stride": 2}),
            ("TransformerRUL", {}),
            ("THGNN", {"graph_hidden_dim": 4, "graph_depth": 1}),
            ("InformerRUL", {"factor": 1}),
        ):
            with self.subTest(name=name):
                config = dict(enc_in=2, seq_len=8, d_model=8, n_heads=2, e_layers=1, d_ff=16, dropout=0.0)
                config.update(kwargs)
                adapter = create_rul_regressor(name, unit="cycles", negative_policy="clip", **config)
                self.assertIsInstance(adapter.model, torch.nn.Module)
                adapter.mark_ready()
                result = adapter.predict(self.input)
                self.assertEqual(result.values.shape, (2,))

    def test_invalid_inputs_targets_and_metadata_fail_closed_without_data_audit(self):
        for kwargs in (
            {"unit": "unknown"},
            {"unit": ""},
            {"unit": "hours", "negative_policy": "silently_ignore"},
            {"unit": "hours", "ready": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                RULRegressor(TinyRUL(), **kwargs)
        with self.assertRaisesRegex(TypeError, "native torch"):
            RULRegressor(object(), unit="cycles")
        m = RULRegressor(TinyRUL(), unit="cycles")
        for target in ([1, 2], [-1, 0, 3], [1, 2, float("nan")]):
            with self.subTest(targets=target), self.assertRaisesRegex(ValueError, "RUL targets"):
                m.fit(self.train, target)
        for kwargs in (
            {"epochs": 0},
            {"batch_size": True},
            {"learning_rate": 0},
            {"loss": "not-a-loss"},
            {"seed": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                m.fit(self.train, [1, 2, 3], **kwargs)
        m.fit(self.train, [1, 2, 3], epochs=1)
        with self.assertRaisesRegex(ValueError, "channel names"):
            m.predict(signals(self.input.values, names=("temp", "vib")))
        with self.assertRaisesRegex(ValueError, "sampling rate"):
            m.predict(signals(self.input.values, rate=42))
        with self.assertRaisesRegex(ValueError, "channel count"):
            m.predict(SignalBatch(np.zeros((2, 8, 3)), sample_rate_hz=100))
        with self.assertRaisesRegex(ValueError, "window length"):
            m.predict(signals(np.zeros((2, 7, 2))))
        with self.assertRaisesRegex(ValueError, "x_mark"):
            m.predict(self.input, x_mark=np.zeros((2, 5, 4)))
        with self.assertRaisesRegex(ValueError, "x_mark"):
            m.predict(self.input, x_mark=np.full((2, 8, 4), float("nan")))

    def test_passed_time_markers_are_preserved_and_fit_is_never_implicit(self):
        class MarkerModel(TinyRUL):
            def forward(self, x, x_mark=None):
                if x_mark is None:
                    raise ValueError("explicit time marker required")
                return x.mean((1, 2)) + x_mark.mean((1, 2)) + self.bias

        m = RULRegressor(MarkerModel(), unit="seconds", ready=True)
        marks = np.ones((2, 8, 4))
        np.testing.assert_allclose(m.predict(self.input, x_mark=marks).values, [3, 3])
        with self.assertRaisesRegex(ValueError, "explicit time marker"):
            m.predict(self.input)


if __name__ == "__main__":
    unittest.main()
