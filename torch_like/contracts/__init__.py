"""Public model/task contract API."""

from torch_like.contracts.inspect import describe, get_model_spec
from torch_like.contracts.types import (
    BackendKind,
    ContractStatus,
    FieldSpec,
    InputSpec,
    ModelSpec,
    OutputSpec,
    TaskKind,
    TaskSpec,
)

__all__ = [
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
]
