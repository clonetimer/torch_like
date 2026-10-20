"""R4d: native PyTorch models share leak-safe R4a/R4c classification protocols."""

import unittest

import numpy as np
import torch

from torch_like.diagnostics.classification import FaultLabelSchema
from torch_like.diagnostics.classification_workflow import (
    FaultClassificationPartition,
    FaultClassificationSplit,
    FaultClassificationWorkflow,
)
from torch_like.diagnostics.generalization import (
    FaultDomainPartition,
    FaultGeneralizationSplit,
    FaultGeneralizationWorkflow,
)
from torch_like.diagnostics.registry import DiagnosticRegistry
from torch_like.diagnostics.torch_classifier import TorchWindowFaultClassifier, register_torch_fault_classifiers
from torch_like.diagnostics.types import SignalBatch

SCHEMA = FaultLabelSchema(("normal", "bearing", "imbalance"))


def partition(*, prefix, device, condition, values=(0, 1, 3, 0, 1, 3)):
    data = np.array([np.full((8, 1), float(v)) for v in values])
    signal = SignalBatch(
        data,
        sample_rate_hz=100,
        channel_names=("vibration",),
        units=("g",),
        sample_ids=tuple(f"{prefix}-{i}" for i in range(len(values))),
    )
    labels = tuple(("normal", "bearing", "imbalance")[i % 3] for i in range(len(values)))
    return FaultDomainPartition(
        FaultClassificationPartition(signal, (prefix,) * len(values), labels),
        (device,) * len(values),
        (condition,) * len(values),
    )


def split():
    train = partition(prefix="train", device="A", condition="cold")
    val = partition(prefix="validation", device="B", condition="warm", values=(0, 1, 3))
    test = partition(prefix="test", device="C", condition="hot", values=(0, 1, 3))
    return FaultGeneralizationSplit(SCHEMA, train, test, val, holdout="both")


class TrackedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 3)
        self.modes = []

    def forward(self, x):
        self.modes.append((self.training, torch.is_grad_enabled()))
        return self.linear(x.reshape(x.shape[0], -1))


class TorchFaultClassifierTests(unittest.TestCase):
    def test_native_inceptiontime_end_to_end(self):
        run = split()
        clf = TorchWindowFaultClassifier(
            "InceptionTime",
            model_kwargs={"d_model": 8, "num_blocks": 1, "bottleneck_dim": 4, "kernel_sizes": (3,), "dropout": 0},
            epochs=1,
            batch_size=3,
            random_state=9,
        )
        result = FaultClassificationWorkflow(clf).run(run.as_classification_split())
        from torch_like.models.diagnosis import InceptionTime

        self.assertIsInstance(clf.model, InceptionTime)
        self.assertEqual(result.probabilities.shape, (3, 3))
        self.assertEqual(result.class_labels, SCHEMA.labels)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1, atol=1e-6)

    def test_native_rocket_integrates_with_unseen_device_and_condition_holdout(self):
        clf = TorchWindowFaultClassifier(
            "ROCKET",
            model_kwargs={"num_kernels": 4, "kernel_sizes": (3,), "dropout": 0},
            epochs=1,
            batch_size=3,
        )
        result = FaultGeneralizationWorkflow(clf).run(split())
        from torch_like.models.diagnosis import ROCKET

        self.assertIsInstance(clf.model, ROCKET)
        self.assertEqual(result.overall.confusion.sum(), 3)
        self.assertEqual(len(result.slices), 4)
        self.assertEqual(result.holdout, "both")

    def test_inference_is_grad_free_label_free_and_never_updates_parameters(self):
        module = TrackedLinear()
        clf = TorchWindowFaultClassifier(model=module, epochs=2, batch_size=3)
        data = split()
        clf.fit(data.train.classification.signals, data.train.classification.labels, SCHEMA)
        original = [param.detach().clone() for param in module.parameters()]
        result = clf.predict_proba(data.test.classification.signals)
        self.assertEqual(result.label_names, SCHEMA.labels)
        self.assertEqual(result.algorithm_name, "TrackedLinear")
        self.assertTrue(any(training and gradients for training, gradients in module.modes))
        self.assertEqual(module.modes[-1], (False, False))
        for before, after in zip(original, module.parameters(), strict=True):
            torch.testing.assert_close(before, after)
        predicted = clf.predict(data.test.classification.signals)
        self.assertEqual(predicted.values.shape, (3,))
        with self.assertRaisesRegex(RuntimeError, "second time"):
            clf.fit(data.train.classification.signals, data.train.classification.labels, SCHEMA)

    def test_training_does_not_advance_global_torch_cpu_rng(self):
        data = split()
        module = TrackedLinear()
        torch.manual_seed(7)
        expected = torch.rand(2)
        torch.manual_seed(7)
        clf = TorchWindowFaultClassifier(model=module, epochs=1, batch_size=3)
        clf.fit(data.train.classification.signals, data.train.classification.labels, SCHEMA)
        torch.testing.assert_close(expected, torch.rand(2))

    def test_metadata_and_missing_values_rejected(self):
        data = split()
        clf = TorchWindowFaultClassifier(model=TrackedLinear(), epochs=1)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            clf.predict_proba(data.test.classification.signals)
        clf.fit(data.train.classification.signals, data.train.classification.labels, SCHEMA)
        for kwargs in ({"sample_rate_hz": 200}, {"channel_names": ("other",)}, {"units": ("m/s2",)}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "must match"):
                batch = SignalBatch(
                    data.test.classification.signals.values,
                    sample_rate_hz=kwargs.get("sample_rate_hz", 100),
                    channel_names=kwargs.get("channel_names", ("vibration",)),
                    units=kwargs.get("units", ("g",)),
                )
                clf.predict_proba(batch)
        with self.assertRaisesRegex(ValueError, "must match"):
            clf.predict_proba(SignalBatch(np.ones((2, 9, 1)), sample_rate_hz=100, units=("g",),
                                          channel_names=("vibration",)))
        missing = SignalBatch(
            np.full((1, 8, 1), np.nan),
            sample_rate_hz=100,
            units=("g",),
            channel_names=("vibration",),
            allow_missing=True,
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            clf.predict_proba(missing)

    def test_unexpected_native_logits_fail_closed(self):
        class WrongOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(2))

            def forward(self, x):
                return self.bias.expand(x.shape[0], -1)

        data = split()
        with self.assertRaisesRegex(ValueError, "declared_classes"):
            TorchWindowFaultClassifier(model=WrongOutput(), epochs=1).fit(
                data.train.classification.signals, data.train.classification.labels, SCHEMA
            )

        class NonfiniteOutput(WrongOutput):
            def forward(self, x):
                return torch.full((len(x), 3), float("nan")) + self.bias.sum() * 0

        with self.assertRaisesRegex(ValueError, "nonfinite logits"):
            TorchWindowFaultClassifier(model=NonfiniteOutput(), epochs=1).fit(
                data.train.classification.signals, data.train.classification.labels, SCHEMA
            )

    def test_train_classes_required_and_one_shot_prevents_test_reuse(self):
        data = split()
        with self.assertRaisesRegex(ValueError, "all declared"):
            TorchWindowFaultClassifier(model=TrackedLinear(), epochs=1).fit(
                data.train.classification.signals, ("normal",) * 6, SCHEMA
            )
        workflow = FaultClassificationWorkflow(TorchWindowFaultClassifier(model=TrackedLinear(), epochs=1))
        workflow.run(data.as_classification_split())
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            workflow.run(data.as_classification_split())

    def test_hyperparameter_validation_and_explicit_registration(self):
        for options in ({"epochs": 0}, {"epochs": True}, {"batch_size": 0},
                        {"learning_rate": 0}, {"learning_rate": float("inf")}, {"random_state": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TorchWindowFaultClassifier(**options)
        with self.assertRaisesRegex(ValueError, "model_name"):
            TorchWindowFaultClassifier("unknown")
        with self.assertRaisesRegex(TypeError, "native torch.nn.Module"):
            TorchWindowFaultClassifier(model=object())
        with self.assertRaisesRegex(ValueError, "model_kwargs"):
            TorchWindowFaultClassifier(model=TrackedLinear(), model_kwargs={"dropout": 0})
        registry = DiagnosticRegistry()
        register_torch_fault_classifiers(registry)
        spec = registry.get_spec("TorchWindowFaultClassifier")
        self.assertEqual(spec.backend, "torch_like")
        self.assertIsInstance(registry.create("TorchWindowFaultClassifier", epochs=1), TorchWindowFaultClassifier)


if __name__ == "__main__":
    unittest.main()
