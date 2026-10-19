"""Transparent candidate fault-source *support*, not causal root-cause inference.

This module maps known fault type probabilities through a caller-reviewed
fault-to-component association table. It performs no model fitting, assigns
no causal priors, and deliberately DOES NOT normalize candidate scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from torch_like.diagnostics.types import DiagnosticAlgorithmSpec, DiagnosticOutputKind, DiagnosticResult, DiagnosticTask


@dataclass(frozen=True)
class FaultSourceRulebook:
    """Versioned expert association: each fault class -> possible physical sources.

    The normal class must map to zero sources. Each other fault class must
    map to >=1 sources. All named sources must have at least one association.
    """

    class_labels: tuple[str, ...]
    source_names: tuple[str, ...]
    fault_to_sources: Mapping[str, tuple[str, ...]]
    normal_label: str = "normal"
    version: str = "unversioned"

    def __post_init__(self) -> None:
        classes = tuple(self.class_labels)
        sources = tuple(self.source_names)
        if (
            len(classes) < 2
            or len(set(classes)) != len(classes)
            or any(not isinstance(name, str) or not name.strip() for name in classes)
        ):
            raise ValueError("class_labels must contain unique nonempty classes")
        if (
            not sources
            or len(set(sources)) != len(sources)
            or any(not isinstance(name, str) or not name.strip() for name in sources)
        ):
            raise ValueError("source_names must contain unique nonempty physical source identifiers")
        if self.normal_label not in classes:
            raise ValueError("normal_label must be a declared class")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must identify the engineering rulebook")
        rules = dict(self.fault_to_sources)
        if set(rules) != set(classes):
            raise ValueError("fault_to_sources must provide exactly one mapping for every declared class")
        resolved: dict[str, tuple[str, ...]] = {}
        associated: set[str] = set()
        for fault in classes:
            candidates = tuple(rules[fault])
            if len(set(candidates)) != len(candidates) or any(name not in sources for name in candidates):
                raise ValueError("fault_to_sources contains duplicate or unknown physical source identifiers")
            if fault == self.normal_label and candidates:
                raise ValueError("normal_label must map to no fault sources")
            if fault != self.normal_label and not candidates:
                raise ValueError("every fault label requires an explicit candidate source")
            associated.update(candidates)
            resolved[fault] = candidates
        if associated != set(sources):
            raise ValueError("every declared physical source must have an associated fault class")
        object.__setattr__(self, "class_labels", classes)
        object.__setattr__(self, "source_names", sources)
        object.__setattr__(self, "fault_to_sources", MappingProxyType(resolved))


class FaultSourceEvidenceMapper:
    """Propagate class support to candidate physical sources without causality."""

    def __init__(self, rulebook: FaultSourceRulebook) -> None:
        if not isinstance(rulebook, FaultSourceRulebook):
            raise TypeError("rulebook must be a reviewed FaultSourceRulebook")
        self.rulebook = rulebook

    def predict(self, classification: DiagnosticResult) -> DiagnosticResult:
        """Return per-source max associated class probability as *support*.

        Multiple sources may receive equal scores from one fault class; scores
        are neither mutually exclusive probabilities nor root-cause priors.
        """
        if not isinstance(classification, DiagnosticResult):
            raise TypeError("classification must be DiagnosticResult")
        if (
            classification.task != DiagnosticTask.FAULT_CLASSIFICATION
            or classification.output_kind != DiagnosticOutputKind.CLASS_PROBABILITY
        ):
            raise ValueError("source mapping requires fault_classification/class_probability input")
        if classification.label_names != self.rulebook.class_labels:
            raise ValueError("classification label ordering differs from the reviewed rulebook")
        probabilities = classification.values
        if probabilities.shape[1] != len(self.rulebook.class_labels) or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5
        ):
            raise ValueError("classification probabilities must be normalized in the declared class order")
        weights = np.zeros((len(self.rulebook.class_labels), len(self.rulebook.source_names)), dtype=bool)
        for class_index, name in enumerate(self.rulebook.class_labels):
            for source in self.rulebook.fault_to_sources[name]:
                weights[class_index, self.rulebook.source_names.index(source)] = True
        score = np.zeros((probabilities.shape[0], len(self.rulebook.source_names)), dtype=float)
        for index in range(score.shape[1]):
            related = weights[:, index]
            score[:, index] = np.max(probabilities[:, related], axis=1)
        return DiagnosticResult(
            task="fault_isolation",
            output_kind="fault_source_score",
            values=score,
            algorithm_name="FaultSourceEvidenceMapper",
            label_names=self.rulebook.source_names,
        )


def register_fault_source_mapper(registry=None) -> None:
    """Opt-in registration; creation requires a caller-provided reviewed rulebook."""
    from torch_like.diagnostics.registry import diagnostic_registry

    target = diagnostic_registry if registry is None else registry
    target.register(
        DiagnosticAlgorithmSpec(
            name="FaultSourceEvidenceMapper",
            backend="statistical",
            task="fault_isolation",
            output_kind="fault_source_score",
            operations=("predict",),
        ),
        FaultSourceEvidenceMapper,
    )
