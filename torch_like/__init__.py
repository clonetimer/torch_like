"""PyTorch-native multi-domain model library with sklearn interoperability.

Recommended usage keeps backend boundaries explicit::

    import torch_like as tl

    model = tl.create("PatchTST", domain="forecasting", ...)
    baseline = tl.create("RandomForestRegressor", backend="sklearn", ...)

Legacy flattened model shortcuts such as ``tl.PatchTST`` remain supported.
"""

from __future__ import annotations

from importlib import import_module

import torch_like.classical as classical
import torch_like.contracts as contracts
import torch_like.models as models
from torch_like.classical import (
    create_estimator,
    get_estimator_class,
    get_estimator_info,
    list_estimators,
)
from torch_like.contracts import (
    BackendKind,
    ContractStatus,
    FieldSpec,
    InputSpec,
    ModelSpec,
    OutputSpec,
    TaskKind,
    TaskSpec,
    describe,
    get_model_spec,
)
from torch_like.factory import create

model_zoo = models.model_zoo
list_domains = models.list_domains
list_model_names = models.list_model_names
list_models = models.list_models
list_models_by_support = models.list_models_by_support
list_models_by_family = models.list_models_by_family
list_models_by_fidelity = models.list_models_by_fidelity
list_models_by_maturity = models.list_models_by_maturity
get_evidence_record = models.get_evidence_record
list_evidence_records = models.list_evidence_records
get_model_class = models.get_model_class
get_model_info = models.get_model_info
find_model_domain = models.find_model_domain
create_model = models.create_model

__version__ = "0.9.0"

_REGISTRY_EXPORTS = (
    "model_zoo",
    "list_domains",
    "list_model_names",
    "list_models",
    "list_models_by_support",
    "list_models_by_family",
    "list_models_by_fidelity",
    "list_models_by_maturity",
    "get_evidence_record",
    "list_evidence_records",
    "get_model_class",
    "get_model_info",
    "find_model_domain",
    "create_model",
)

_CONTRACT_EXPORTS = (
    "BackendKind",
    "ContractStatus",
    "TaskKind",
    "FieldSpec",
    "InputSpec",
    "OutputSpec",
    "TaskSpec",
    "ModelSpec",
    "get_model_spec",
    "describe",
)
_RUNTIME_EXPORTS = ("as_runtime", "save_model", "load_model")
_TRAINING_EXPORTS = (
    "Trainer",
    "TrainerConfig",
    "TrainingResult",
    "EpochResult",
    "train_model",
    "load_training_checkpoint",
    "save_training_checkpoint",
)

_CLASSICAL_EXPORTS = (
    "create_estimator",
    "get_estimator_class",
    "get_estimator_info",
    "list_estimators",
)
_NAMESPACE_EXPORTS = ("models", "classical", "contracts", "evaluation", "training", "nn", "data", "automl")
__all__ = [
    "__version__",
    "create",
    *_CONTRACT_EXPORTS,
    *_RUNTIME_EXPORTS,
    *_TRAINING_EXPORTS,
    *_REGISTRY_EXPORTS,
    *_CLASSICAL_EXPORTS,
    *_NAMESPACE_EXPORTS,
    *models.MODEL_EXPORTS,
]


def as_runtime(model):
    """Return a capability-checked runtime adapter without changing the native model object."""

    from torch_like.runtime import as_runtime as _as_runtime

    return _as_runtime(model)


def save_model(model, path, *, overwrite: bool = False):
    """Persist a torch_like or curated sklearn model into a self-describing directory."""

    from torch_like.serialization import save_model as _save_model

    return _save_model(model, path, overwrite=overwrite)


def load_model(path, *, map_location="cpu", strict: bool = True):
    """Restore a model written by :func:`save_model`."""

    from torch_like.serialization import load_model as _load_model

    return _load_model(path, map_location=map_location, strict=strict)


def __getattr__(name: str):
    if name in models.MODEL_EXPORTS:
        value = getattr(models, name)
        globals()[name] = value
        return value
    if name in _TRAINING_EXPORTS:
        training_module = import_module("torch_like.training")
        value = getattr(training_module, name)
        globals()[name] = value
        return value
    if name in {"evaluation", "training", "nn", "data", "automl"}:
        value = import_module(f"torch_like.{name}")
        globals()[name] = value
        return value
    raise AttributeError(f"module 'torch_like' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
