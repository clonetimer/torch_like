"""Development-only multi-seed fault-classifier selection before one sealed test.

No candidate receives sealed test signals, predictions, or labels during ranking.
Native sklearn estimators and PyTorch modules remain backend-owned objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from torch_like.diagnostics.classification_workflow import (
    FaultClassificationEvaluation,
    FaultClassificationSplit,
    FaultClassificationWorkflow,
)
from torch_like.diagnostics.generalization import (
    FaultGeneralizationEvaluation,
    FaultGeneralizationSplit,
    FaultGeneralizationWorkflow,
)


def _immutable_parameters(items: tuple[tuple[str, Any], ...], field: str) -> tuple[tuple[str, Any], ...]:
    """Accept JSON-like constructor atoms, not mutable objects or nested dicts."""
    try:
        pairs = tuple(items)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of key/value pairs") from exc
    names = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{field} must contain (name, value) pairs")
        key, value = pair
        if not isinstance(key, str) or not key.isidentifier() or key.startswith("_"):
            raise ValueError(f"{field} keys must be public identifiers")
        if key in {"random_state", "model", "model_name", "estimator", "estimator_name"}:
            raise ValueError("seed and model identity must be set by the candidate specification")
        members = value if isinstance(value, tuple) else (value,)
        if any(type(member) not in {str, bool, int, float} for member in members):
            raise ValueError(f"{field} values must be immutable scalar atoms or flat tuples")
        if any(type(member) is float and not np.isfinite(member) for member in members):
            raise ValueError(f"{field} values must be finite")
        names.append(key)
    if len(names) != len(set(names)):
        raise ValueError(f"{field} keys must be unique")
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class FaultCandidateSpec:
    """Declared backend, model, seeds and nominal budget, frozen before selection."""

    identifier: str
    backend: str
    algorithm: str
    seeds: tuple[int, ...]
    model_parameters: tuple[tuple[str, Any], ...] = ()
    training_parameters: tuple[tuple[str, Any], ...] = ()
    budget_unit: str = "unspecified"
    budget_value: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("candidate identifier must be nonempty")
        allowed = {
            "sklearn": {"RandomForestClassifier", "SVC"},
            "torch_like": {"InceptionTime", "ROCKET"},
        }
        if self.backend not in allowed or self.algorithm not in allowed[self.backend]:
            raise ValueError("unsupported native backend or classifier model")
        try:
            seeds = tuple(self.seeds)
        except TypeError as exc:
            raise ValueError("seeds must be a sequence") from exc
        if len(seeds) < 2 or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("seeds must contain at least two distinct nonnegative integers")
        model_args = _immutable_parameters(self.model_parameters, "model_parameters")
        train_args = _immutable_parameters(self.training_parameters, "training_parameters")
        if self.backend == "sklearn" and train_args:
            raise ValueError("sklearn candidates cannot declare training_parameters")
        expected_unit = (
            "trees" if self.algorithm == "RandomForestClassifier" else "fits" if self.algorithm == "SVC" else "epochs"
        )
        if self.budget_unit != expected_unit or type(self.budget_value) is not int or self.budget_value < 1:
            raise ValueError(f"{self.algorithm} must declare a positive {expected_unit} budget")
        model_kwargs = dict(model_args)
        train_kwargs = dict(train_args)
        configured = (
            model_kwargs.get("n_estimators", 100)
            if expected_unit == "trees"
            else train_kwargs.get("epochs", 3)
            if expected_unit == "epochs"
            else 1
        )
        if configured != self.budget_value:
            raise ValueError("declared budget_value must equal configured estimator trees or training epochs")
        if self.algorithm == "SVC" and model_kwargs.get("probability", True) is not True:
            raise ValueError("SVC must retain probability=True")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "model_parameters", model_args)
        object.__setattr__(self, "training_parameters", train_args)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "seeds": list(self.seeds),
            "model_parameters": dict(self.model_parameters),
            "training_parameters": dict(self.training_parameters),
            "budget": {"unit": self.budget_unit, "value": self.budget_value},
        }


@dataclass(frozen=True)
class FaultSeedResult:
    """Validation-only metric or explicit candidate/seed failure."""

    candidate: str
    seed: int
    accuracy: float | None
    macro_f1: float | None
    failure_type: str | None = None
    failure_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "seed": self.seed,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


@dataclass(frozen=True)
class FaultCandidateRanking:
    identifier: str
    mean_validation_score: float | None
    min_validation_score: float | None
    score_std: float | None
    eligible: bool
    successful_seeds: int
    required_seeds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.identifier,
            "mean_validation_score": self.mean_validation_score,
            "min_validation_score": self.min_validation_score,
            "score_std": self.score_std,
            "eligible": self.eligible,
            "successful_seeds": self.successful_seeds,
            "required_seeds": self.required_seeds,
        }


@dataclass(frozen=True)
class FaultSelectionReport:
    """No sealed test predictions, labels, scores or probabilities are retained."""

    metric: str
    class_labels: tuple[str, ...]
    validation_support: tuple[int, ...]
    validation_recordings: int
    validation_domains: tuple[tuple[str, str, int, int], ...]
    candidates: tuple[FaultCandidateSpec, ...]
    seed_results: tuple[FaultSeedResult, ...]
    rankings: tuple[FaultCandidateRanking, ...]
    selected_candidate: str
    selected_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "development_validation_only_multiseed",
            "metric": self.metric,
            "class_labels": list(self.class_labels),
            "validation_support": list(self.validation_support),
            "validation_recordings": self.validation_recordings,
            "validation_domains": [
                {"axis": axis, "identity": identity, "windows": windows, "recordings": recordings}
                for axis, identity, windows, recordings in self.validation_domains
            ],
            "candidate_specs": [spec.to_dict() for spec in self.candidates],
            "seed_results": [result.to_dict() for result in self.seed_results],
            "rankings": [ranking.to_dict() for ranking in self.rankings],
            "selected_candidate": self.selected_candidate,
            "selected_seed": self.selected_seed,
            "selection_rule": "maximum mean validation metric, then lower std, then identifier",
            "seed_rule": "first predeclared seed (not maximum validation seed)",
            "budget_comparability": "nominal counts only; training compute across algorithms is not comparable",
        }


@dataclass(frozen=True)
class FaultSelectionEvaluation:
    selection: FaultSelectionReport
    sealed: FaultClassificationEvaluation | FaultGeneralizationEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {"selection": self.selection.to_dict(), "sealed_test": self.sealed.to_dict()}


class FaultSelectionWorkflow:
    """Select on validation, then fit a fresh chosen model and score test once."""

    def __init__(
        self,
        candidates: tuple[FaultCandidateSpec, ...],
        *,
        metric: str = "macro_f1",
        min_class_windows: int = 1,
        min_validation_recordings: int = 1,
        min_domain_recordings: int = 1,
        factory: Callable[[FaultCandidateSpec, int], Any] | None = None,
    ) -> None:
        specs = tuple(candidates)
        if not specs or any(not isinstance(spec, FaultCandidateSpec) for spec in specs):
            raise ValueError("candidates must be a nonempty tuple of FaultCandidateSpec")
        if len({spec.identifier for spec in specs}) != len(specs):
            raise ValueError("candidate identifiers must be unique")
        if metric not in {"macro_f1", "accuracy"}:
            raise ValueError("metric must be macro_f1 or accuracy")
        for value in (min_class_windows, min_validation_recordings, min_domain_recordings):
            if type(value) is not int or value < 1:
                raise ValueError("minimum support requirements must be positive integers")
        if factory is not None and not callable(factory):
            raise TypeError("factory must be callable")
        self.candidates = specs
        self.metric = metric
        self.min_class_windows = min_class_windows
        self.min_validation_recordings = min_validation_recordings
        self.min_domain_recordings = min_domain_recordings
        self.factory = factory
        self._consumed = False

    def _new_classifier(self, spec: FaultCandidateSpec, seed: int) -> Any:
        if self.factory is not None:
            return self.factory(spec, seed)
        if spec.backend == "sklearn":
            from torch_like.diagnostics.classification import SklearnWindowFaultClassifier

            return SklearnWindowFaultClassifier(
                estimator_name=spec.algorithm, random_state=seed, **dict(spec.model_parameters)
            )
        from torch_like.diagnostics.torch_classifier import TorchWindowFaultClassifier

        return TorchWindowFaultClassifier(
            model_name=spec.algorithm,
            model_kwargs=dict(spec.model_parameters),
            random_state=seed,
            **dict(spec.training_parameters),
        )

    def _development(
        self, split: FaultClassificationSplit | FaultGeneralizationSplit
    ) -> tuple[FaultClassificationSplit, tuple[tuple[str, str, int, int], ...]]:
        if split.validation is None:
            raise ValueError("model selection requires an independent validation partition")
        if isinstance(split, FaultGeneralizationSplit):
            train = split.train.classification
            validation = split.validation.classification
            axes = (
                ("device", split.validation.device_ids),
                ("condition", split.validation.condition_ids),
                ("pair", tuple("|".join(pair) for pair in split.validation.context_pairs)),
            )
            domains = []
            for axis, identities in axes:
                for identity in sorted(set(identities)):
                    positions = [i for i, key in enumerate(identities) if key == identity]
                    recordings = len({validation.recording_ids[i] for i in positions})
                    if recordings < self.min_domain_recordings:
                        raise ValueError(f"insufficient independent validation recordings for {axis}:{identity}")
                    domains.append((axis, identity, len(positions), recordings))
        else:
            train, validation = split.train, split.validation
            domains = []
        support = np.bincount(split.schema.encode(validation.labels), minlength=len(split.schema.labels))
        if (support < self.min_class_windows).any():
            raise ValueError("insufficient validation class support for all declared classes")
        if len(set(validation.recording_ids)) < self.min_validation_recordings:
            raise ValueError("insufficient independent validation recordings")
        return FaultClassificationSplit(split.schema, train, validation), tuple(domains)

    def run(self, split: FaultClassificationSplit | FaultGeneralizationSplit) -> FaultSelectionEvaluation:
        if self._consumed:
            raise RuntimeError("selection workflow has already consumed its sealed test")
        if not isinstance(split, (FaultClassificationSplit, FaultGeneralizationSplit)):
            raise TypeError("run requires a validated classification or generalization split")
        self._consumed = True
        development, domains = self._development(split)
        validation = development.test
        records = []
        rankings = []
        for spec in self.candidates:
            successful = []
            for seed in spec.seeds:
                try:
                    classifier = self._new_classifier(spec, seed)
                    outcome = FaultClassificationWorkflow(classifier).run(development)
                    score = float(getattr(outcome, self.metric))
                    if not np.isfinite(score):
                        raise ValueError("validation metric must be finite")
                    successful.append(score)
                    records.append(FaultSeedResult(spec.identifier, seed, outcome.accuracy, outcome.macro_f1))
                except Exception as exc:
                    records.append(
                        FaultSeedResult(
                            spec.identifier,
                            seed,
                            None,
                            None,
                            type(exc).__name__,
                            str(exc)[:256],
                        )
                    )
            if len(successful) == len(spec.seeds):
                rankings.append(
                    FaultCandidateRanking(
                        spec.identifier,
                        float(np.mean(successful)),
                        float(np.min(successful)),
                        float(np.std(successful)),
                        True,
                        len(successful),
                        len(spec.seeds),
                    )
                )
            else:
                rankings.append(
                    FaultCandidateRanking(spec.identifier, None, None, None, False, len(successful), len(spec.seeds))
                )
        eligible = [rank for rank in rankings if rank.eligible]
        if not eligible:
            raise RuntimeError("no fully validated candidate; sealed test was not evaluated")
        ordered = sorted(
            eligible,
            key=lambda rank: (
                -float(rank.mean_validation_score),
                float(rank.score_std),
                rank.identifier,
            ),
        )
        winner = next(spec for spec in self.candidates if spec.identifier == ordered[0].identifier)
        report = FaultSelectionReport(
            metric=self.metric,
            class_labels=split.schema.labels,
            validation_support=tuple(
                int(x) for x in np.bincount(split.schema.encode(validation.labels), minlength=len(split.schema.labels))
            ),
            validation_recordings=len(set(validation.recording_ids)),
            validation_domains=domains,
            candidates=self.candidates,
            seed_results=tuple(records),
            rankings=tuple(rankings),
            selected_candidate=winner.identifier,
            selected_seed=winner.seeds[0],
        )
        final_classifier = self._new_classifier(winner, winner.seeds[0])
        if isinstance(split, FaultGeneralizationSplit):
            sealed = FaultGeneralizationWorkflow(final_classifier).run(split)
        else:
            sealed = FaultClassificationWorkflow(final_classifier).run(split)
        return FaultSelectionEvaluation(selection=report, sealed=sealed)
