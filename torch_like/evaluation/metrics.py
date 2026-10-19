"""Dataset-level metric primitives shared by Trainer, runners, and AutoML.

The 0.8 metric contract distinguishes two execution modes:

* :class:`FunctionalMetric` is an exact buffered fallback for arbitrary
  ``(y_true, y_pred)`` callables. It intentionally stores all batches because
  an arbitrary nonlinear function cannot in general be merged safely.
* Builtin streaming metrics keep sufficient statistics only and expose
  ``state_dict`` / ``merge_state``. Their state can be combined across data
  shards without averaging already-computed metric values.

This module deliberately does not import scikit-learn at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

MetricFunction = Callable[[np.ndarray, np.ndarray], float]
MetricState = Mapping[str, Any]


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _copy_state(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_state(item) for item in value)
    return deepcopy(value)


def _validate_pair(predictions: Any, targets: Any) -> tuple[np.ndarray, np.ndarray]:
    prediction = _numpy(predictions)
    target = _numpy(targets)
    if prediction.ndim == 0 or target.ndim == 0:
        raise ValueError("metric predictions and targets must include a leading sample dimension")
    if prediction.shape[0] != target.shape[0]:
        raise ValueError(
            "metric predictions and targets must have matching leading sample dimensions: "
            f"{prediction.shape} != {target.shape}"
        )
    return prediction, target


@runtime_checkable
class Metric(Protocol):
    """Stateful dataset metric interface kept compatible with 0.7 custom metrics."""

    def reset(self) -> None: ...

    def update(self, predictions: Any, targets: Any) -> None: ...

    def compute(self) -> float: ...


@runtime_checkable
class MergeableMetric(Metric, Protocol):
    """Metric whose sufficient state can be serialized and merged exactly."""

    @property
    def is_streaming(self) -> bool: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: MetricState) -> None: ...

    def merge_state(self, state: MetricState) -> None: ...


class FunctionalMetric:
    """Exact dataset metric backed by a ``(y_true, y_pred)`` function.

    Batches are accumulated on CPU and the function is called once at epoch end,
    avoiding mathematically incorrect averaging of nonlinear batch metrics such
    as F1 and R². For large validation sets prefer one of the builtin streaming
    metrics when its semantics match the desired metric.
    """

    is_streaming = False

    def __init__(self, function: MetricFunction):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function
        self.reset()

    def reset(self) -> None:
        self._predictions: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, predictions: Any, targets: Any) -> None:
        prediction, target = _validate_pair(predictions, targets)
        self._predictions.append(np.array(prediction, copy=True))
        self._targets.append(np.array(target, copy=True))

    def compute(self) -> float:
        if not self._predictions:
            raise ValueError("metric received no batches")
        predictions = np.concatenate(self._predictions, axis=0)
        targets = np.concatenate(self._targets, axis=0)
        value = float(self.function(targets, predictions))
        if not np.isfinite(value):
            raise FloatingPointError("metric produced NaN or infinite value")
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "functional",
            "predictions": [value.copy() for value in self._predictions],
            "targets": [value.copy() for value in self._targets],
        }

    def load_state_dict(self, state: MetricState) -> None:
        if state.get("kind") != "functional":
            raise ValueError("incompatible FunctionalMetric state")
        predictions = state.get("predictions", [])
        targets = state.get("targets", [])
        if len(predictions) != len(targets):
            raise ValueError("FunctionalMetric state has mismatched prediction/target batches")
        self._predictions = [np.asarray(value).copy() for value in predictions]
        self._targets = [np.asarray(value).copy() for value in targets]

    def merge_state(self, state: MetricState) -> None:
        if state.get("kind") != "functional":
            raise ValueError("incompatible FunctionalMetric state")
        predictions = state.get("predictions", [])
        targets = state.get("targets", [])
        if len(predictions) != len(targets):
            raise ValueError("FunctionalMetric state has mismatched prediction/target batches")
        self._predictions.extend(np.asarray(value).copy() for value in predictions)
        self._targets.extend(np.asarray(value).copy() for value in targets)


class RegressionMetric:
    """Streaming MAE, MSE, RMSE, or uniform-average R².

    Non-leading dimensions are treated as independent outputs. This matches the
    semantics used by :func:`regression_metrics`: MAE/MSE are uniformly averaged
    over output elements, while R² is computed per output then uniformly
    averaged. Constant-target outputs follow scikit-learn's finite convention:
    perfect prediction -> 1.0, otherwise -> 0.0.
    """

    is_streaming = True
    _SUPPORTED = {"mae", "mse", "rmse", "r2"}

    def __init__(self, metric: str):
        if metric not in self._SUPPORTED:
            raise ValueError(f"unsupported regression metric {metric!r}; expected one of {sorted(self._SUPPORTED)}")
        self.metric = metric
        self.reset()

    def reset(self) -> None:
        self._samples = 0
        self._outputs: int | None = None
        self._sum_abs_error: np.ndarray | None = None
        self._sum_sq_error: np.ndarray | None = None
        self._sum_target: np.ndarray | None = None
        self._sum_target_sq: np.ndarray | None = None

    @staticmethod
    def _matrix(value: np.ndarray) -> np.ndarray:
        if value.ndim == 1:
            return value.reshape(-1, 1).astype(np.float64, copy=False)
        return value.reshape(value.shape[0], -1).astype(np.float64, copy=False)

    def update(self, predictions: Any, targets: Any) -> None:
        prediction, target = _validate_pair(predictions, targets)
        if prediction.shape != target.shape:
            raise ValueError(
                f"regression predictions and targets must have identical shapes: {prediction.shape} != {target.shape}"
            )
        pred = self._matrix(prediction)
        truth = self._matrix(target)
        outputs = pred.shape[1]
        if self._outputs is None:
            self._outputs = outputs
            self._sum_abs_error = np.zeros(outputs, dtype=np.float64)
            self._sum_sq_error = np.zeros(outputs, dtype=np.float64)
            self._sum_target = np.zeros(outputs, dtype=np.float64)
            self._sum_target_sq = np.zeros(outputs, dtype=np.float64)
        elif outputs != self._outputs:
            raise ValueError(f"regression output width changed across batches: {outputs} != {self._outputs}")
        error = pred - truth
        self._sum_abs_error += np.sum(np.abs(error), axis=0)
        self._sum_sq_error += np.sum(np.square(error), axis=0)
        self._sum_target += np.sum(truth, axis=0)
        self._sum_target_sq += np.sum(np.square(truth), axis=0)
        self._samples += int(pred.shape[0])

    def _require_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._samples <= 0 or self._outputs is None:
            raise ValueError("metric received no batches")
        assert self._sum_abs_error is not None
        assert self._sum_sq_error is not None
        assert self._sum_target is not None
        assert self._sum_target_sq is not None
        return self._sum_abs_error, self._sum_sq_error, self._sum_target, self._sum_target_sq

    def compute(self) -> float:
        sum_abs, sum_sq, sum_target, sum_target_sq = self._require_state()
        if self.metric == "mae":
            value = np.mean(sum_abs / self._samples)
        elif self.metric == "mse":
            value = np.mean(sum_sq / self._samples)
        elif self.metric == "rmse":
            value = np.sqrt(np.mean(sum_sq / self._samples))
        else:
            denominator = sum_target_sq - np.square(sum_target) / self._samples
            scores = np.empty_like(denominator)
            non_constant = denominator > np.finfo(np.float64).eps
            scores[non_constant] = 1.0 - sum_sq[non_constant] / denominator[non_constant]
            perfect_constant = (~non_constant) & (sum_sq <= np.finfo(np.float64).eps)
            scores[perfect_constant] = 1.0
            scores[(~non_constant) & (~perfect_constant)] = 0.0
            value = np.mean(scores)
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("metric produced NaN or infinite value")
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "regression",
            "metric": self.metric,
            "samples": self._samples,
            "outputs": self._outputs,
            "sum_abs_error": None if self._sum_abs_error is None else self._sum_abs_error.copy(),
            "sum_sq_error": None if self._sum_sq_error is None else self._sum_sq_error.copy(),
            "sum_target": None if self._sum_target is None else self._sum_target.copy(),
            "sum_target_sq": None if self._sum_target_sq is None else self._sum_target_sq.copy(),
        }

    def load_state_dict(self, state: MetricState) -> None:
        self.reset()
        self._load_or_merge(state, merge=False)

    def merge_state(self, state: MetricState) -> None:
        self._load_or_merge(state, merge=True)

    def _load_or_merge(self, state: MetricState, *, merge: bool) -> None:
        if state.get("kind") != "regression" or state.get("metric") != self.metric:
            raise ValueError("incompatible RegressionMetric state")
        samples = int(state.get("samples", 0))
        outputs = state.get("outputs")
        if samples == 0:
            return
        if outputs is None or int(outputs) <= 0:
            raise ValueError("RegressionMetric state is missing a valid output width")
        arrays = [
            np.asarray(state[name], dtype=np.float64)
            for name in ("sum_abs_error", "sum_sq_error", "sum_target", "sum_target_sq")
        ]
        if any(array.shape != (int(outputs),) for array in arrays):
            raise ValueError("RegressionMetric state arrays do not match output width")
        if not merge or self._samples == 0:
            self._samples = samples
            self._outputs = int(outputs)
            self._sum_abs_error, self._sum_sq_error, self._sum_target, self._sum_target_sq = [
                array.copy() for array in arrays
            ]
            return
        if self._outputs != int(outputs):
            raise ValueError(f"cannot merge regression states with output widths {self._outputs} and {outputs}")
        self._samples += samples
        self._sum_abs_error += arrays[0]
        self._sum_sq_error += arrays[1]
        self._sum_target += arrays[2]
        self._sum_target_sq += arrays[3]


class ForecastingMetric:
    """Streaming forecasting MAE/RMSE/SMAPE, optionally for one horizon."""

    is_streaming = True
    _SUPPORTED = {"mae", "rmse", "smape"}

    def __init__(self, metric: str, *, horizon: int | None = None):
        if metric not in self._SUPPORTED:
            raise ValueError(f"unsupported forecasting metric {metric!r}; expected one of {sorted(self._SUPPORTED)}")
        if horizon is not None and horizon < 0:
            raise ValueError("horizon must be a zero-based non-negative index")
        if metric == "smape" and horizon is not None:
            # This is mathematically valid, but keeping the contract small avoids
            # an accidental second naming convention for horizon SMAPE.
            raise ValueError("horizon-specific SMAPE is not part of the public metric contract")
        self.metric = metric
        self.horizon = horizon
        self.reset()

    def reset(self) -> None:
        self._element_count = 0
        self._sum_abs_error = 0.0
        self._sum_sq_error = 0.0
        self._sum_smape = 0.0

    def update(self, predictions: Any, targets: Any) -> None:
        prediction, target = _validate_pair(predictions, targets)
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("forecasting targets and predictions must have shape (samples, horizon, targets)")
        if self.horizon is not None:
            if self.horizon >= prediction.shape[1]:
                raise ValueError(f"horizon index {self.horizon} is outside prediction horizon {prediction.shape[1]}")
            prediction = prediction[:, self.horizon, :]
            target = target[:, self.horizon, :]
        pred = prediction.astype(np.float64, copy=False)
        truth = target.astype(np.float64, copy=False)
        error = pred - truth
        self._sum_abs_error += float(np.sum(np.abs(error)))
        self._sum_sq_error += float(np.sum(np.square(error)))
        if self.metric == "smape":
            denominator = np.maximum(np.abs(truth) + np.abs(pred), np.finfo(np.float64).eps)
            self._sum_smape += float(np.sum(2.0 * np.abs(error) / denominator))
        self._element_count += int(error.size)

    def compute(self) -> float:
        if self._element_count <= 0:
            raise ValueError("metric received no batches")
        if self.metric == "mae":
            value = self._sum_abs_error / self._element_count
        elif self.metric == "rmse":
            value = np.sqrt(self._sum_sq_error / self._element_count)
        else:
            value = self._sum_smape / self._element_count
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("metric produced NaN or infinite value")
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "forecasting",
            "metric": self.metric,
            "horizon": self.horizon,
            "element_count": self._element_count,
            "sum_abs_error": self._sum_abs_error,
            "sum_sq_error": self._sum_sq_error,
            "sum_smape": self._sum_smape,
        }

    def load_state_dict(self, state: MetricState) -> None:
        self.reset()
        self._load_or_merge(state, merge=False)

    def merge_state(self, state: MetricState) -> None:
        self._load_or_merge(state, merge=True)

    def _load_or_merge(self, state: MetricState, *, merge: bool) -> None:
        if (
            state.get("kind") != "forecasting"
            or state.get("metric") != self.metric
            or state.get("horizon") != self.horizon
        ):
            raise ValueError("incompatible ForecastingMetric state")
        count = int(state.get("element_count", 0))
        values = (
            float(state.get("sum_abs_error", 0.0)),
            float(state.get("sum_sq_error", 0.0)),
            float(state.get("sum_smape", 0.0)),
        )
        if not merge:
            self._element_count = count
            self._sum_abs_error, self._sum_sq_error, self._sum_smape = values
            return
        self._element_count += count
        self._sum_abs_error += values[0]
        self._sum_sq_error += values[1]
        self._sum_smape += values[2]


class ClassificationMetric:
    """Streaming accuracy/precision/recall/F1 for single-label or multilabel data."""

    is_streaming = True
    _SUPPORTED = {"accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_micro"}

    def __init__(self, metric: str, *, multilabel: bool = False):
        if metric not in self._SUPPORTED:
            raise ValueError(f"unsupported classification metric {metric!r}; expected one of {sorted(self._SUPPORTED)}")
        self.metric = metric
        self.multilabel = bool(multilabel)
        self.reset()

    def reset(self) -> None:
        self._samples = 0
        self._correct_samples = 0
        self._counts: dict[Any, np.ndarray] = {}

    def _bucket(self, label: Any) -> np.ndarray:
        key = label.item() if isinstance(label, np.generic) else label
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError(f"classification label must be hashable, got {type(key).__name__}") from exc
        return self._counts.setdefault(key, np.zeros(3, dtype=np.int64))  # tp, fp, fn

    def update(self, predictions: Any, targets: Any) -> None:
        prediction, target = _validate_pair(predictions, targets)
        if prediction.shape != target.shape:
            raise ValueError(
                "classification predictions and targets must have identical shapes: "
                f"{prediction.shape} != {target.shape}"
            )
        if self.multilabel:
            if prediction.ndim != 2:
                raise ValueError("multilabel classification expects shape (samples, labels)")
            pred_bool = prediction.astype(bool)
            truth_bool = target.astype(bool)
            self._correct_samples += int(np.sum(np.all(pred_bool == truth_bool, axis=1)))
            self._samples += int(prediction.shape[0])
            for index in range(prediction.shape[1]):
                pred_col = pred_bool[:, index]
                truth_col = truth_bool[:, index]
                bucket = self._bucket(index)
                bucket[0] += int(np.sum(pred_col & truth_col))
                bucket[1] += int(np.sum(pred_col & ~truth_col))
                bucket[2] += int(np.sum(~pred_col & truth_col))
            return

        pred_flat = prediction.reshape(-1)
        truth_flat = target.reshape(-1)
        if len(pred_flat) != prediction.shape[0]:
            raise ValueError("single-label classification expects one label per sample")
        self._samples += int(len(pred_flat))
        self._correct_samples += int(np.sum(pred_flat == truth_flat))
        for label in np.unique(np.concatenate([truth_flat, pred_flat])):
            bucket = self._bucket(label)
            bucket[0] += int(np.sum((pred_flat == label) & (truth_flat == label)))
            bucket[1] += int(np.sum((pred_flat == label) & (truth_flat != label)))
            bucket[2] += int(np.sum((pred_flat != label) & (truth_flat == label)))

    @staticmethod
    def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.zeros_like(numerator, dtype=np.float64)
        valid = denominator != 0
        result[valid] = numerator[valid] / denominator[valid]
        return result

    def compute(self) -> float:
        if self._samples <= 0:
            raise ValueError("metric received no batches")
        if self.metric == "accuracy":
            return float(self._correct_samples / self._samples)
        if not self._counts:
            raise ValueError("metric received no class observations")
        counts = np.stack(list(self._counts.values()), axis=0).astype(np.float64)
        tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
        if self.metric == "precision_macro":
            value = np.mean(self._safe_divide(tp, tp + fp))
        elif self.metric == "recall_macro":
            value = np.mean(self._safe_divide(tp, tp + fn))
        elif self.metric == "f1_macro":
            value = np.mean(self._safe_divide(2.0 * tp, 2.0 * tp + fp + fn))
        else:
            value = self._safe_divide(
                np.asarray([2.0 * tp.sum()]),
                np.asarray([2.0 * tp.sum() + fp.sum() + fn.sum()]),
            )[0]
        return float(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "classification",
            "metric": self.metric,
            "multilabel": self.multilabel,
            "samples": self._samples,
            "correct_samples": self._correct_samples,
            "counts": {key: value.copy() for key, value in self._counts.items()},
        }

    def load_state_dict(self, state: MetricState) -> None:
        self.reset()
        self._load_or_merge(state, merge=False)

    def merge_state(self, state: MetricState) -> None:
        self._load_or_merge(state, merge=True)

    def _load_or_merge(self, state: MetricState, *, merge: bool) -> None:
        if (
            state.get("kind") != "classification"
            or state.get("metric") != self.metric
            or bool(state.get("multilabel")) != self.multilabel
        ):
            raise ValueError("incompatible ClassificationMetric state")
        if not merge:
            self.reset()
        self._samples += int(state.get("samples", 0))
        self._correct_samples += int(state.get("correct_samples", 0))
        for key, value in dict(state.get("counts", {})).items():
            array = np.asarray(value, dtype=np.int64)
            if array.shape != (3,):
                raise ValueError("ClassificationMetric state count arrays must have shape (3,)")
            self._bucket(key)[:] += array


class MeanMetric:
    """Streaming arithmetic mean over prediction values, ignoring targets."""

    is_streaming = True

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._sum = 0.0

    def update(self, predictions: Any, targets: Any = None) -> None:
        del targets
        values = _numpy(predictions).astype(np.float64, copy=False)
        self._count += int(values.size)
        self._sum += float(values.sum())

    def compute(self) -> float:
        if self._count <= 0:
            raise ValueError("metric received no values")
        return float(self._sum / self._count)

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "mean", "count": self._count, "sum": self._sum}

    def load_state_dict(self, state: MetricState) -> None:
        if state.get("kind") != "mean":
            raise ValueError("incompatible MeanMetric state")
        self._count = int(state.get("count", 0))
        self._sum = float(state.get("sum", 0.0))

    def merge_state(self, state: MetricState) -> None:
        if state.get("kind") != "mean":
            raise ValueError("incompatible MeanMetric state")
        self._count += int(state.get("count", 0))
        self._sum += float(state.get("sum", 0.0))


class StandardDeviationMetric:
    """Streaming population standard deviation over prediction values."""

    is_streaming = True

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, predictions: Any, targets: Any = None) -> None:
        del targets
        values = _numpy(predictions).astype(np.float64, copy=False)
        self._count += int(values.size)
        self._sum += float(values.sum())
        self._sum_sq += float(np.square(values).sum())

    def compute(self) -> float:
        if self._count <= 0:
            raise ValueError("metric received no values")
        mean = self._sum / self._count
        variance = max(self._sum_sq / self._count - mean * mean, 0.0)
        return float(np.sqrt(variance))

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "std", "count": self._count, "sum": self._sum, "sum_sq": self._sum_sq}

    def load_state_dict(self, state: MetricState) -> None:
        if state.get("kind") != "std":
            raise ValueError("incompatible StandardDeviationMetric state")
        self._count = int(state.get("count", 0))
        self._sum = float(state.get("sum", 0.0))
        self._sum_sq = float(state.get("sum_sq", 0.0))

    def merge_state(self, state: MetricState) -> None:
        if state.get("kind") != "std":
            raise ValueError("incompatible StandardDeviationMetric state")
        self._count += int(state.get("count", 0))
        self._sum += float(state.get("sum", 0.0))
        self._sum_sq += float(state.get("sum_sq", 0.0))


class MetricEvaluator:
    """Manage a named collection of exact dataset-level metrics.

    ``MetricEvaluator`` never averages final metric values across shards. Use
    :meth:`state_dict` and :meth:`merge_state` to combine metric sufficient
    statistics, then call :meth:`compute` once. This is the contract future
    DDP/FSDP integrations can use without changing metric semantics.
    """

    def __init__(self, metrics: Mapping[str, Metric | MetricFunction]):
        if not metrics:
            raise ValueError("metrics must contain at least one metric")
        self.metrics: dict[str, Metric] = {}
        for name, metric in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric names must be non-empty strings")
            self.metrics[name] = metric if isinstance(metric, Metric) else FunctionalMetric(metric)

    @property
    def fully_streaming(self) -> bool:
        """Whether every configured metric has bounded sufficient state."""

        return all(bool(getattr(metric, "is_streaming", False)) for metric in self.metrics.values())

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def update(self, predictions: Any, targets: Any) -> None:
        for metric in self.metrics.values():
            metric.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def state_dict(self) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for name, metric in self.metrics.items():
            if not isinstance(metric, MergeableMetric):
                raise TypeError(
                    f"metric {name!r} does not expose mergeable state; implement state_dict/load_state_dict/merge_state"
                )
            states[name] = _copy_state(metric.state_dict())
        return {"schema_version": 1, "metrics": states}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        states = self._validated_states(state)
        for name, metric in self.metrics.items():
            if not isinstance(metric, MergeableMetric):
                raise TypeError(f"metric {name!r} does not expose mergeable state")
            metric.load_state_dict(states[name])

    def merge_state(self, state: Mapping[str, Any]) -> None:
        states = self._validated_states(state)
        for name, metric in self.metrics.items():
            if not isinstance(metric, MergeableMetric):
                raise TypeError(f"metric {name!r} does not expose mergeable state")
            metric.merge_state(states[name])

    def merge(self, other: "MetricEvaluator") -> None:
        if not isinstance(other, MetricEvaluator):
            raise TypeError("other must be a MetricEvaluator")
        self.merge_state(other.state_dict())

    def _validated_states(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("unsupported MetricEvaluator state schema")
        states = state.get("metrics")
        if not isinstance(states, Mapping):
            raise ValueError("MetricEvaluator state is missing metrics mapping")
        expected = set(self.metrics)
        actual = set(states)
        if actual != expected:
            raise ValueError(
                f"MetricEvaluator state names do not match evaluator: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        return states


def classification_metric_set(*, multilabel: bool = False) -> dict[str, ClassificationMetric]:
    """Return streaming metrics equivalent to :func:`classification_metrics`."""

    names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    if multilabel:
        names.append("f1_micro")
    return {name: ClassificationMetric(name, multilabel=multilabel) for name in names}


def regression_metric_set() -> dict[str, RegressionMetric]:
    """Return streaming MAE/RMSE/R² metrics."""

    return {name: RegressionMetric(name) for name in ("mae", "rmse", "r2")}


def forecasting_metric_set(horizon: int | None = None) -> dict[str, Metric]:
    """Return streaming forecasting metrics.

    With ``horizon=None`` the collection contains overall MAE/RMSE/SMAPE.
    With a positive horizon length, per-horizon MAE/RMSE metrics are added using
    the same one-based names as :func:`forecasting_metrics`.
    """

    metrics: dict[str, Metric] = {
        "mae": ForecastingMetric("mae"),
        "rmse": ForecastingMetric("rmse"),
        "smape": ForecastingMetric("smape"),
    }
    if horizon is not None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        for index in range(horizon):
            metrics[f"horizon_{index + 1}_mae"] = ForecastingMetric("mae", horizon=index)
            metrics[f"horizon_{index + 1}_rmse"] = ForecastingMetric("rmse", horizon=index)
    return metrics


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, multilabel: bool = False) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if multilabel:
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_2d = y_true[:, None] if y_true.ndim == 1 else y_true
    y_pred_2d = y_pred[:, None] if y_pred.ndim == 1 else y_pred
    return {
        "mae": float(mean_absolute_error(y_true_2d, y_pred_2d)),
        "rmse": float(mean_squared_error(y_true_2d, y_pred_2d) ** 0.5),
        "r2": float(r2_score(y_true_2d, y_pred_2d, multioutput="uniform_average")),
    }


def forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise ValueError("Forecasting targets and predictions must have shape (samples, horizon, targets)")
    metrics = regression_metrics(y_true.reshape(len(y_true), -1), y_pred.reshape(len(y_pred), -1))
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), np.finfo(np.float64).eps)
    metrics["smape"] = float(np.mean(2.0 * np.abs(y_pred - y_true) / denominator))
    for horizon in range(y_true.shape[1]):
        metrics[f"horizon_{horizon + 1}_mae"] = float(mean_absolute_error(y_true[:, horizon], y_pred[:, horizon]))
        metrics[f"horizon_{horizon + 1}_rmse"] = float(
            mean_squared_error(y_true[:, horizon], y_pred[:, horizon]) ** 0.5
        )
    return metrics


def multi_target_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_weights: np.ndarray | None = None,
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_2d = y_true[:, None] if y_true.ndim == 1 else y_true
    y_pred_2d = y_pred[:, None] if y_pred.ndim == 1 else y_pred
    metrics = regression_metrics(y_true_2d, y_pred_2d)
    target_ranges = np.ptp(y_true_2d, axis=0)
    target_scales = np.where(target_ranges > 0, target_ranges, np.maximum(np.std(y_true_2d, axis=0), 1.0))
    target_rmse = np.sqrt(np.mean(np.square(y_true_2d - y_pred_2d), axis=0))
    normalized_rmse = target_rmse / target_scales
    metrics["macro_nrmse"] = float(np.mean(normalized_rmse))
    if target_weights is not None:
        weights = np.asarray(target_weights, dtype=np.float64)
        if weights.shape != normalized_rmse.shape or weights.sum() <= 0:
            raise ValueError("target_weights must match target count and have a positive sum")
        metrics["weighted_nrmse"] = float(normalized_rmse @ (weights / weights.sum()))
    for index in range(y_true_2d.shape[1]):
        metrics[f"target_{index}_mae"] = float(mean_absolute_error(y_true_2d[:, index], y_pred_2d[:, index]))
        metrics[f"target_{index}_rmse"] = float(mean_squared_error(y_true_2d[:, index], y_pred_2d[:, index]) ** 0.5)
    return metrics


def metric_direction(metric: str) -> str:
    return (
        "minimize"
        if metric in {"mae", "rmse", "smape", "macro_nrmse", "weighted_nrmse", "false_alarm_rate"}
        or metric.endswith(("_mae", "_rmse"))
        else "maximize"
    )


__all__ = [
    "ClassificationMetric",
    "ForecastingMetric",
    "FunctionalMetric",
    "MeanMetric",
    "MergeableMetric",
    "Metric",
    "MetricEvaluator",
    "MetricFunction",
    "MetricState",
    "RegressionMetric",
    "StandardDeviationMetric",
    "classification_metric_set",
    "classification_metrics",
    "forecasting_metric_set",
    "forecasting_metrics",
    "metric_direction",
    "multi_target_regression_metrics",
    "regression_metric_set",
    "regression_metrics",
]
