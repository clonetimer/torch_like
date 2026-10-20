"""Explicit inference-only composition of independently fitted diagnostic algorithms.

No automatic score fusion, data-source auditing, fit, label handling or
evaluation workflow. Downstream stages consume the preceding typed result.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from torch_like.diagnostics.types import DiagnosticResult, SignalBatch


@dataclass(frozen=True)
class DiagnosticStage:
    """One named algorithm with an explicit method and typed upstream source."""

    name: str
    algorithm: Any
    method: str = "score"
    source: str = "signals"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or self.name == "signals":
            raise ValueError("stage name must be nonempty and cannot be signals")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be signals or a preceding stage name")
        if self.method not in {"score", "predict", "predict_proba"}:
            raise ValueError("method must be score, predict or predict_proba")
        if not callable(getattr(self.algorithm, self.method, None)):
            raise TypeError("algorithm must implement its declared inference method")


@dataclass(frozen=True)
class DiagnosticBundleResult:
    """Immutable mapping of unmodified per-stage diagnostic outputs."""

    results: Mapping[str, DiagnosticResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))

    def __getitem__(self, name: str) -> DiagnosticResult:
        return self.results[name]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """JSON-compatible typed output without inventing probability semantics."""
        return {
            name: {
                "task": result.task.value,
                "output_kind": result.output_kind.value,
                "algorithm_name": result.algorithm_name,
                "values": result.values.tolist(),
                "threshold": result.threshold,
                "label_names": None if result.label_names is None else list(result.label_names),
            }
            for name, result in self.results.items()
        }


class DiagnosticPipeline:
    """Run named native algorithms without changing their training semantics.

    Independent stages consume the original SignalBatch (source="signals").
    Explicit dependent stages consume exactly one prior DiagnosticResult, e.g.
    a classification probability result passed to a fault-source mapper.

    There is deliberately no fit(), automatic thresholding, calibration,
    label processing, score averaging, or implicit unit conversion.
    """

    def __init__(self, *stages: DiagnosticStage) -> None:
        if not stages:
            raise ValueError("pipeline requires at least one explicit stage")
        available = {"signals"}
        for stage in stages:
            if not isinstance(stage, DiagnosticStage):
                raise TypeError("pipeline entries must be DiagnosticStage")
            if stage.name in available:
                raise ValueError(f"duplicate stage name: {stage.name}")
            if stage.source not in available:
                raise ValueError(f"stage {stage.name} refers to unknown or later source: {stage.source}")
            available.add(stage.name)
        self.stages = tuple(stages)

    def predict(self, batch: SignalBatch) -> DiagnosticBundleResult:
        if not isinstance(batch, SignalBatch):
            raise TypeError("pipeline requires SignalBatch input")
        outputs: dict[str, DiagnosticResult] = {}
        for stage in self.stages:
            value = batch if stage.source == "signals" else outputs[stage.source]
            result = getattr(stage.algorithm, stage.method)(value)
            if not isinstance(result, DiagnosticResult):
                raise TypeError(f"stage {stage.name} must return DiagnosticResult")
            if result.sample_count != batch.shape[0]:
                raise ValueError(f"stage {stage.name} changed the sample axis")
            outputs[stage.name] = result
        return DiagnosticBundleResult(outputs)

    def score(self, batch: SignalBatch) -> DiagnosticBundleResult:
        """Convenience synonym for predict: all stages use declared methods."""
        return self.predict(batch)


__all__ = ["DiagnosticStage", "DiagnosticBundleResult", "DiagnosticPipeline"]
