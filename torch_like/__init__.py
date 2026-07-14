"""PyTorch-native model library with lazy, domain-oriented model exports.

Typical usage::

    import torch_like as tl
    model = tl.Informer(...)

    model = tl.create_model("Autoformer", domain="forecasting", ...)
"""

from __future__ import annotations

import torch_like.models as _models

model_zoo = _models.model_zoo
list_domains = _models.list_domains
list_models = _models.list_models
list_models_by_support = _models.list_models_by_support
list_models_by_family = _models.list_models_by_family
get_model_info = _models.get_model_info
create_model = _models.create_model

__version__ = "0.2.0"

_REGISTRY_EXPORTS = (
    "model_zoo",
    "list_domains",
    "list_models",
    "list_models_by_support",
    "list_models_by_family",
    "get_model_info",
    "create_model",
)
__all__ = ["__version__", *_REGISTRY_EXPORTS, *_models.MODEL_EXPORTS]


def __getattr__(name: str):
    if name in _models.MODEL_EXPORTS:
        return getattr(_models, name)
    raise AttributeError(f"module 'torch_like' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
