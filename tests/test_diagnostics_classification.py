"""R4a: class vocabulary, native sklearn parity and closed-world fault outputs."""
import unittest

import numpy as np

from torch_like.diagnostics.classification import (
    FaultLabelSchema,
    SklearnWindowFaultClassifier,
    register_sklearn_fault_classifiers,
)
from torch_like.diagnostics.registry import DiagnosticRegistry
from torch_like.diagnostics.types import SignalBatch


SCHEMA = FaultLabelSchema(("normal", "bearing", "imbalance"))


def batch(values, *, rate=100, names=("vibration",), units=("g",), prefix="item"):
    data = np.asarray(values, dtype=float)
    if data.ndim == 2:
        data = data[:, :, None]
    return SignalBatch(
        data,
        sample_rate_hz=rate,
        channel_names=names,
        units=units,
        sample_ids=tuple(f"{prefix}-{index}" for index in range(data.shape[0])),
    )


def training():
    rows = [[0] * 8, [0.1] * 8, [1] * 8, [1.1] * 8, [3] * 8, [3.1] * 8]
    return batch(rows, prefix="train"), ("normal", "normal", "bearing", "bearing", "imbalance", "imbalance")


class ReversedNativeEstimator:
    def __init__(self):
        self.fit_calls = 0
        self.prediction_calls = 0

    def fit(self, x, y):
        self.fit_calls += 1
        self.training_matrix = np.array(x, copy=True)
        self.training_labels = np.array(y, copy=True)
        self.classes_ = np.array([2, 1, 0])
        return self

    def predict_proba(self, x):
        self.prediction_calls += 1
        return np.tile([0.7, 0.2, 0.1], (len(x), 1))


class ClassifierTests(unittest.TestCase):
    def test_schema_validates_order_and_unknown_labels(self):
        self.assertEqual(SCHEMA.encode(["imbalance", "normal"]).tolist(), [2, 0])
        with self.assertRaisesRegex(ValueError, "normal_label"):
            FaultLabelSchema(("fault_a", "fault_b"))
        for names in (("normal",), ("normal", "normal"), ("normal", "")):
            with self.subTest(labels=names), self.assertRaisesRegex(ValueError, "labels"):
                FaultLabelSchema(names)
        with self.assertRaisesRegex(ValueError, "predefined"):
            SCHEMA.encode(["unknown"])

    def test_native_reversed_class_probability_mapping_and_no_refit(self):
        estimator = ReversedNativeEstimator()
        classifier = SklearnWindowFaultClassifier(estimator, features=("mean",))
        reference, labels = training()
        classifier.fit(reference, labels, SCHEMA)
        query = batch([[0] * 8, [3] * 8], prefix="query")
        probability = classifier.predict_proba(query)
        np.testing.assert_allclose(probability.values, [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
        self.assertEqual(probability.label_names, SCHEMA.labels)
        self.assertEqual(probability.output_kind, "class_probability")
        predicted = classifier.predict(query)
        self.assertEqual(predicted.values.tolist(), ["imbalance", "imbalance"])
        self.assertEqual(predicted.output_kind, "fault_label")
        self.assertEqual(estimator.fit_calls, 1)
        self.assertEqual(estimator.prediction_calls, 2)
        self.assertEqual(classifier.feature_names_, ("vibration__mean",))
        np.testing.assert_array_equal(estimator.training_labels, [0, 0, 1, 1, 2, 2])

    def test_real_random_forest_native_and_svc_probability_models(self):
        reference, labels = training()
        query = batch([[0.0] * 8, [1.0] * 8, [3.0] * 8], prefix="query")
        rf = SklearnWindowFaultClassifier(n_estimators=14, random_state=12).fit(reference, labels, SCHEMA)
        self.assertEqual(rf.estimator.__class__.__name__, "RandomForestClassifier")
        values = rf.predict_proba(query).values
        self.assertEqual(values.shape, (3, 3))
        np.testing.assert_allclose(values.sum(axis=1), [1, 1, 1])
        self.assertEqual(rf.schema_.labels, SCHEMA.labels)
        svc = SklearnWindowFaultClassifier(estimator_name="SVC", kernel="linear", random_state=1)
        svc.fit(reference, labels, SCHEMA)
        self.assertEqual(svc.estimator.__class__.__name__, "SVC")
        self.assertEqual(svc.predict_proba(query).values.shape, (3, 3))

    def test_requires_training_before_scoring_and_all_train_classes(self):
        reference, labels = training()
        clf = SklearnWindowFaultClassifier(ReversedNativeEstimator())
        with self.assertRaisesRegex(RuntimeError, "fit"):
            clf.predict_proba(reference)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            _ = clf.schema_
        with self.assertRaisesRegex(RuntimeError, "fit"):
            _ = clf.feature_names_
        with self.assertRaisesRegex(ValueError, "all declared"):
            clf.fit(reference, ("normal",) * 6, SCHEMA)
        with self.assertRaisesRegex(ValueError, "align"):
            clf.fit(reference, labels[:3], SCHEMA)
        with self.assertRaisesRegex(ValueError, "predefined"):
            clf.fit(reference, ("normal",) * 5 + ("unknown",), SCHEMA)

    def test_inference_metadata_rejects_unit_channel_rate_and_length_changes(self):
        reference, labels = training()
        clf = SklearnWindowFaultClassifier(ReversedNativeEstimator()).fit(reference, labels, SCHEMA)
        for kwargs, match in (
            ({"names": ("different",)}, "channel_names"),
            ({"units": ("m/s2",)}, "units"),
            ({"rate": 20}, "sample_rate_hz"),
        ):
            with self.subTest(metadata=kwargs), self.assertRaisesRegex(ValueError, match):
                clf.predict_proba(batch([[1] * 8], prefix="query", **kwargs))
        with self.assertRaisesRegex(ValueError, "window length"):
            clf.predict_proba(batch([[1] * 9], prefix="query"))

    def test_invalid_estimator_contracts_and_classes(self):
        reference, labels = training()
        with self.assertRaisesRegex(TypeError, "predict_proba"):
            SklearnWindowFaultClassifier(object())
        with self.assertRaisesRegex(ValueError, "combined"):
            SklearnWindowFaultClassifier(ReversedNativeEstimator(), random_state=1)
        with self.assertRaisesRegex(ValueError, "supported native"):
            SklearnWindowFaultClassifier(estimator_name="not_allowed")
        with self.assertRaisesRegex(ValueError, "probability=True"):
            SklearnWindowFaultClassifier(estimator_name="SVC", probability=False)
        class BrokenNative(ReversedNativeEstimator):
            def fit(self, x, y):
                self.classes_ = np.array([0, 1])
                return self
        with self.assertRaisesRegex(ValueError, "classes_"):
            SklearnWindowFaultClassifier(BrokenNative()).fit(reference, labels, SCHEMA)

    def test_malformed_probabilities_fail_closed(self):
        reference, labels = training()
        class BrokenOutput(ReversedNativeEstimator):
            output = [[0.2, 0.2, 0.2]]

            def predict_proba(self, x):
                return np.tile(self.output, (len(x), 1))
        estimator = BrokenOutput()
        clf = SklearnWindowFaultClassifier(estimator).fit(reference, labels, SCHEMA)
        with self.assertRaisesRegex(ValueError, "normalized"):
            clf.predict_proba(batch([[1] * 8], prefix="query"))
        estimator.output = [[0.2, float("nan"), 0.8]]
        with self.assertRaisesRegex(ValueError, "finite"):
            clf.predict_proba(batch([[1] * 8], prefix="query"))
        estimator.output = [[-0.2, 0.2, 1.0]]
        with self.assertRaisesRegex(ValueError, "invalid"):
            clf.predict_proba(batch([[1] * 8], prefix="query"))

    def test_registry_opt_in_and_duplicate_protection(self):
        registry = DiagnosticRegistry()
        self.assertFalse(registry.list_specs())
        register_sklearn_fault_classifiers(registry)
        spec = registry.get_spec("SklearnWindowFaultClassifier")
        self.assertEqual((spec.backend, spec.task, spec.output_kind), (
            "sklearn", "fault_classification", "class_probability"
        ))
        self.assertIsInstance(registry.create("SklearnWindowFaultClassifier", n_estimators=4), SklearnWindowFaultClassifier)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_sklearn_fault_classifiers(registry)


if __name__ == "__main__":
    unittest.main()
