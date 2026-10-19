"""Native sklearn window-level fault classification with an explicit class schema.

This classifier identifies predeclared fault types. It is NOT a fault-source
isolation or root-cause inference system. No labels enter inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from torch_like.diagnostics.sklearn_adapter import DEFAULT_WINDOW_FEATURES, WindowFeatureTransformer
from torch_like.diagnostics.types import DiagnosticAlgorithmSpec, DiagnosticResult, SignalBatch


@dataclass(frozen=True)
class FaultLabelSchema:
    """Explicit ordered single-label vocabulary, including a healthy class."""

    labels: tuple[str, ...]
    normal_label: str = "normal"

    def __post_init__(self) -> None:
        names = tuple(self.labels)
        if (
            len(names) < 2
            or len(set(names)) != len(names)
            or any(not isinstance(x, str) or not x.strip() for x in names)
        ):
            raise ValueError("labels must contain at least two unique nonempty class names")
        if self.normal_label not in names:
            raise ValueError("normal_label must belong to the declared class labels")
        object.__setattr__(self, "labels", names)

    def encode(self, labels: tuple[str, ...] | list[str]) -> np.ndarray:
        mapping = {name: index for index, name in enumerate(self.labels)}
        values = tuple(labels)
        if not values or any(name not in mapping for name in values):
            raise ValueError("fault labels must all belong to the predefined nonempty schema")
        return np.asarray([mapping[name] for name in values], dtype=np.int64)


class SklearnWindowFaultClassifier:
    """Feature transform plus native RandomForest or probability-enabled SVC."""

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        estimator_name: str = "RandomForestClassifier",
        features: tuple[str, ...] = DEFAULT_WINDOW_FEATURES,
        **estimator_kwargs: Any,
    ) -> None:
        if estimator is not None and estimator_kwargs:
            raise ValueError("custom estimator cannot be combined with estimator keyword arguments")
        if estimator is None:
            if estimator_name not in {"RandomForestClassifier", "SVC"}:
                raise ValueError("supported native estimators: RandomForestClassifier and SVC")
            from torch_like.classical import create_estimator

            options = (
                {"n_estimators": 100, "random_state": 0}
                if estimator_name == "RandomForestClassifier"
                else {"probability": True, "random_state": 0}
            )
            options.update(estimator_kwargs)
            if estimator_name == "SVC" and not options["probability"]:
                raise ValueError("SVC requires probability=True for class-probability output")
            estimator = create_estimator(estimator_name, **options)
        if not callable(getattr(estimator, "fit", None)) or not callable(getattr(estimator, "predict_proba", None)):
            raise TypeError("estimator must provide native fit and predict_proba")
        self.estimator = estimator
        self.transformer = WindowFeatureTransformer(features=features)
        self._schema: FaultLabelSchema | None = None
        self._channel_names: tuple[str, ...] | None = None
        self._units: tuple[str, ...] | None = None
        self._rate: float | None = None
        self._window_length: int | None = None
        self._class_order: np.ndarray | None = None

    @property
    def schema_(self) -> FaultLabelSchema:
        if self._schema is None:
            raise RuntimeError("fit the classifier on development labels before inference")
        return self._schema

    @property
    def feature_names_(self) -> tuple[str, ...]:
        if self._channel_names is None:
            raise RuntimeError("fit the classifier before requesting feature names")
        return self.transformer.column_names(self._channel_names)

    def fit(
        self,
        reference: SignalBatch,
        labels: tuple[str, ...] | list[str],
        schema: FaultLabelSchema,
    ) -> "SklearnWindowFaultClassifier":
        """Fit on caller-provided development data; no sealed-test labels."""
        if not isinstance(reference, SignalBatch) or not isinstance(schema, FaultLabelSchema):
            raise TypeError("fit requires SignalBatch and FaultLabelSchema")
        names = tuple(labels)
        if len(names) != reference.shape[0]:
            raise ValueError("training labels must align with input windows")
        encoded = schema.encode(names)
        if set(encoded.tolist()) != set(range(len(schema.labels))):
            raise ValueError("training windows must represent all declared fault classes")
        matrix = self.transformer.transform(reference)
        self.estimator.fit(matrix, encoded)
        native = np.asarray(getattr(self.estimator, "classes_", []))
        if native.shape != (len(schema.labels),) or native.dtype.kind not in "iu":
            raise ValueError("native estimator classes_ must contain all integer-encoded classes")
        if set(native.tolist()) != set(range(len(schema.labels))):
            raise ValueError("native estimator returned unexpected class indices")
        order = np.asarray([int(np.flatnonzero(native == i)[0]) for i in range(len(schema.labels))])
        order.setflags(write=False)
        self._schema = schema
        self._class_order = order
        self._channel_names = reference.channel_names
        self._units = reference.units
        self._rate = reference.sample_rate_hz
        self._window_length = reference.shape[1]
        return self

    def _checked_features(self, batch: SignalBatch) -> np.ndarray:
        _ = self.schema_
        if not isinstance(batch, SignalBatch):
            raise TypeError("inference input must be SignalBatch")
        if batch.channel_names != self._channel_names:
            raise ValueError("channel_names and ordering must match training")
        if batch.units != self._units:
            raise ValueError("physical units must match training")
        if batch.sample_rate_hz != self._rate:
            raise ValueError("sample_rate_hz must match training")
        if batch.shape[1] != self._window_length:
            raise ValueError("window length must match training")
        return self.transformer.transform(batch)

    def predict_proba(self, batch: SignalBatch) -> DiagnosticResult:
        """Return per-window probabilities ordered by the fixed fault schema."""
        matrix = self._checked_features(batch)
        values = np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)
        schema = self.schema_
        if values.shape != (batch.shape[0], len(schema.labels)) or not np.isfinite(values).all():
            raise ValueError("native estimator must emit finite (windows, classes) probabilities")
        if self._class_order is None:
            raise RuntimeError("native estimator class ordering has not been fitted")
        ordered = values[:, self._class_order]
        if (
            np.any(ordered < 0)
            or np.any(ordered > 1)
            or not np.allclose(ordered.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
        ):
            raise ValueError("native estimator produced invalid normalized class probabilities")
        return DiagnosticResult(
            task="fault_classification",
            output_kind="class_probability",
            values=ordered,
            algorithm_name="SklearnWindowFaultClassifier",
            label_names=schema.labels,
        )

    def predict(self, batch: SignalBatch) -> DiagnosticResult:
        """Return categorical predicted labels, never inference-time ground truth."""
        result = self.predict_proba(batch)
        names = np.asarray(self.schema_.labels)
        return DiagnosticResult(
            task="fault_classification",
            output_kind="fault_label",
            values=names[np.argmax(result.values, axis=1)],
            algorithm_name=result.algorithm_name,
            label_names=self.schema_.labels,
        )


def register_sklearn_fault_classifiers(registry=None) -> None:
    """Opt-in factory registration without modifying native sklearn objects."""
    from torch_like.diagnostics.registry import diagnostic_registry

    target = diagnostic_registry if registry is None else registry
    target.register(
        DiagnosticAlgorithmSpec(
            name="SklearnWindowFaultClassifier",
            backend="sklearn",
            task="fault_classification",
            output_kind="class_probability",
            operations=("fit", "predict"),
        ),
        SklearnWindowFaultClassifier,
    )
