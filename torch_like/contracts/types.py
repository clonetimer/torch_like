"""Stable, machine-readable contracts for torch_like algorithms.

The contracts describe orchestration-facing behavior without replacing the
native PyTorch or scikit-learn object models.  They are intentionally small,
JSON-serializable descriptors that can be consumed by AutoML, CLIs, services,
and project-specific runners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class BackendKind(StringEnum):
    TORCH_LIKE = "torch_like"
    SKLEARN = "sklearn"


class ContractStatus(StringEnum):
    STANDARDIZED = "standardized"
    PARTIAL = "partial"


class TaskKind(StringEnum):
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    PROBABILISTIC_FORECASTING = "probabilistic_forecasting"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    GRAPH_PREDICTION = "graph_prediction"
    CAUSAL_LANGUAGE_MODELING = "causal_language_modeling"
    SEQ2SEQ = "seq2seq"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    GENERATIVE = "generative"
    TABULAR_PREDICTION = "tabular_prediction"
    RECOMMENDATION = "recommendation"
    AUDIO = "audio"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    RUL_REGRESSION = "rul_regression"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DECOMPOSITION = "decomposition"
    CUSTOM = "custom"


@dataclass(frozen=True)
class FieldSpec:
    """One input field in a task batch."""

    name: str
    required: bool
    semantic: str
    dtype: str | None = None
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InputSpec:
    """Batch-level input contract used by one model family."""

    batch_type: str | None
    fields: tuple[FieldSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_type": self.batch_type,
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True)
class OutputSpec:
    """Semantic output contract.

    ``shape`` is descriptive rather than an executable shape checker because
    several domains (graph, vision detection, multimodal, RL) intentionally
    support more than one concrete output structure.
    """

    kind: str
    semantic: str
    shape: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSpec:
    """Task-level contract shared by compatible model implementations."""

    kind: TaskKind
    input: InputSpec
    output: OutputSpec
    target_field: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "input": self.input.to_dict(),
            "output": self.output.to_dict(),
            "target_field": self.target_field,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ModelSpec:
    """Resolved orchestration contract for one concrete algorithm."""

    name: str
    backend: BackendKind
    domain: str
    class_path: str
    task: TaskSpec
    operations: tuple[str, ...]
    contract_status: ContractStatus = ContractStatus.STANDARDIZED
    contract_version: str = "1.0"

    def supports(self, operation: str) -> bool:
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        return operation in self.operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend.value,
            "domain": self.domain,
            "class_path": self.class_path,
            "task": self.task.to_dict(),
            "operations": list(self.operations),
            "contract_status": self.contract_status.value,
            "contract_version": self.contract_version,
        }
