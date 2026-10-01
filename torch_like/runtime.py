"""Optional orchestration adapter over native torch_like/sklearn objects."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from torch_like.contracts import BackendKind, ModelSpec, TaskKind, get_model_spec
from torch_like.serialization import save_model


class UnsupportedOperationError(RuntimeError):
    """Raised when a model contract does not advertise an operation."""


class ModelRuntime:
    """A thin orchestration adapter that preserves the wrapped native object.

    The adapter intentionally does not replace ``nn.Module`` or sklearn APIs.
    It exists for code that needs to dispatch across heterogeneous algorithms
    while checking capabilities before invocation.
    """

    def __init__(self, model: Any):
        self.model = model
        self.spec: ModelSpec = get_model_spec(model)

    def supports(self, operation: str) -> bool:
        return self.spec.supports(operation)

    def _require(self, operation: str) -> None:
        if not self.supports(operation):
            raise UnsupportedOperationError(
                f"{self.spec.name} ({self.spec.backend.value}/{self.spec.domain}) does not advertise "
                f"operation {operation!r}; supported operations: {list(self.spec.operations)}"
            )

    def make_batch(self, **values: Any) -> Any:
        """Construct the task dataclass declared by the contract."""

        batch_type = self.spec.task.input.batch_type
        if batch_type is None:
            raise UnsupportedOperationError(
                f"{self.spec.name} uses its native {self.spec.backend.value} input API and has no torch_like batch type"
            )
        module_name, _, class_name = batch_type.rpartition(".")
        cls = getattr(import_module(module_name), class_name)
        return cls(**values)

    def predict(self, batch: Any) -> Any:
        self._require("predict")
        if self.spec.backend == BackendKind.SKLEARN:
            return self.model.predict(batch)
        if self.spec.task.kind == TaskKind.CAUSAL_LANGUAGE_MODELING:
            return self.model.next_token_logits(batch)
        predictor = getattr(self.model, "predict", None)
        if not callable(predictor):
            raise UnsupportedOperationError(f"{self.spec.name} has no callable predict(batch) implementation")
        return predictor(batch)

    def loss(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        self._require("loss")
        loss_step = getattr(self.model, "loss_step", None)
        if callable(loss_step):
            return loss_step(batch, *args, **kwargs)
        fit_step = getattr(self.model, "fit_step", None)
        if callable(fit_step):
            return fit_step(batch, *args, **kwargs)
        raise UnsupportedOperationError(f"{self.spec.name} advertises loss but no loss_step/fit_step was found")

    def score(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        self._require("score")
        return self.model.score_batch(batch, *args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self._require("generate")
        return self.model.generate(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        self._require("sample")
        return self.model.sample(*args, **kwargs)

    def distribution(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        self._require("distribution")
        return self.model.predict_distribution(batch, *args, **kwargs)

    def act(self, *args: Any, **kwargs: Any) -> Any:
        self._require("act")
        for name in ("act", "greedy_action", "predict_action", "sample_action"):
            method = getattr(self.model, name, None)
            if callable(method):
                return method(*args, **kwargs)
        raise UnsupportedOperationError(f"{self.spec.name} advertises act but no supported action method was found")

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Any:
        self._require("fit")
        if y is None:
            return self.model.fit(X, **kwargs)
        return self.model.fit(X, y, **kwargs)

    def transform(self, X: Any, **kwargs: Any) -> Any:
        self._require("transform")
        return self.model.transform(X, **kwargs)

    def fit_predict(self, X: Any, y: Any = None, **kwargs: Any) -> Any:
        self._require("fit_predict")
        if y is None:
            return self.model.fit_predict(X, **kwargs)
        return self.model.fit_predict(X, y, **kwargs)

    def save(self, path: str, *, overwrite: bool = False):
        return save_model(self.model, path, overwrite=overwrite)


def as_runtime(model: Any) -> ModelRuntime:
    """Return a capability-checked orchestration adapter for *model*."""

    if isinstance(model, ModelRuntime):
        return model
    return ModelRuntime(model)


__all__ = ["UnsupportedOperationError", "ModelRuntime", "as_runtime"]
