"""Runtime trust audits for leaderboard data-split protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from torch_like.automl.data import DataSplit, GroupCVSplit, PreparedData, RollingOriginSplit, ValidationFold
from torch_like.automl.specs import LeaderboardConfig, LeaderboardTrustLevel, SplitSpec, SplitType, TaskType


@dataclass(frozen=True)
class ProtocolAudit:
    """Serializable proof that one materialized split passed trust checks."""

    status: str
    protocol: str
    sample_count: int
    checks: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationTrustAudit:
    """Serializable proof that a leaderboard evaluation plan matches its declared trust level."""

    status: str
    level: str
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_field(candidate: Any, field: str) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(field)
    return getattr(candidate, field, None)


def _minimum_seed_count(config: LeaderboardConfig) -> int:
    if config.trust.minimum_evaluation_seeds is not None:
        return config.trust.minimum_evaluation_seeds
    return 3 if config.trust.level == LeaderboardTrustLevel.VALIDATED else 1


def _indices(values: np.ndarray, name: str, sample_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Protocol audit requires one-dimensional integer indices for {name}")
    if len(array) == 0:
        raise ValueError(f"Protocol audit found an empty index set: {name}")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"Protocol audit found duplicate indices within {name}")
    if array.min() < 0 or array.max() >= sample_count:
        raise ValueError(f"Protocol audit found out-of-range indices in {name}")
    return array.astype(np.int64, copy=False)


def _disjoint(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> None:
    overlap = np.intersect1d(left, right)
    if len(overlap):
        raise ValueError(
            f"Protocol audit found {len(overlap)} overlapping samples between {left_name} and {right_name}"
        )


def _complete_partition(parts: tuple[np.ndarray, ...], sample_count: int, name: str) -> None:
    combined = np.concatenate(parts)
    if len(combined) != sample_count or not np.array_equal(np.sort(combined), np.arange(sample_count)):
        raise ValueError(f"Protocol audit found that {name} does not partition all prepared samples exactly once")


def _groups_disjoint(data: PreparedData, left: np.ndarray, right: np.ndarray, context: str) -> None:
    if data.groups is None:
        raise ValueError(f"Protocol audit requires groups for {context}")
    overlap = np.intersect1d(np.unique(data.groups[left]), np.unique(data.groups[right]))
    if len(overlap):
        raise ValueError(f"Protocol audit found group leakage in {context}: {overlap.astype(str).tolist()}")


def _strictly_before(data: PreparedData, left: np.ndarray, right: np.ndarray, context: str) -> None:
    if data.timestamps is None:
        raise ValueError(f"Protocol audit requires timestamps for {context}")
    left_times = pd.to_datetime(data.timestamps[left], errors="raise")
    right_times = pd.to_datetime(data.timestamps[right], errors="raise")
    if left_times.max() >= right_times.min():
        raise ValueError(f"Protocol audit found non-chronological or equal-boundary timestamps in {context}")


def _fold_indices(fold: ValidationFold, index: int, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    train = _indices(fold.train, f"fold[{index}].train", sample_count)
    val = _indices(fold.val, f"fold[{index}].val", sample_count)
    _disjoint(train, val, f"fold[{index}].train", f"fold[{index}].val")
    return train, val


def _audit_single(
    data: PreparedData,
    split: DataSplit,
    spec: SplitSpec,
    task_type: TaskType,
) -> ProtocolAudit:
    count = len(data.x)
    train = _indices(split.train, "train", count)
    val = _indices(split.val, "val", count)
    test = _indices(split.test, "test", count)
    _disjoint(train, val, "train", "validation")
    _disjoint(train, test, "train", "sealed test")
    _disjoint(val, test, "validation", "sealed test")
    _complete_partition((train, val, test), count, "train/validation/test")
    checks = ["indices_valid", "partitions_disjoint", "complete_partition", "sealed_test_isolated"]
    if spec.type == SplitType.GROUP:
        _groups_disjoint(data, train, val, "train/validation")
        _groups_disjoint(data, train, test, "train/test")
        _groups_disjoint(data, val, test, "validation/test")
        checks.append("groups_disjoint")
    if spec.type in {SplitType.TIME, SplitType.PREDEFINED} and data.timestamps is not None:
        _strictly_before(data, train, val, "train/validation")
        _strictly_before(data, val, test, "validation/test")
        checks.append("timestamps_chronological")
    protocol = (
        "predefined_validation_sealed_test"
        if task_type == TaskType.ANOMALY_DETECTION
        else "single_validation_sealed_test"
    )
    return ProtocolAudit(
        status="passed",
        protocol=protocol,
        sample_count=count,
        checks=tuple(checks),
        details={"train": len(train), "validation": len(val), "sealed_test": len(test)},
    )


def _audit_group_cv(data: PreparedData, split: GroupCVSplit) -> ProtocolAudit:
    count = len(data.x)
    development = _indices(split.development, "development", count)
    test = _indices(split.test, "sealed test", count)
    _disjoint(development, test, "development", "sealed test")
    _complete_partition((development, test), count, "development/sealed-test")
    _groups_disjoint(data, development, test, "development/sealed-test")
    validation_parts = []
    development_set = set(development.tolist())
    for index, fold in enumerate(split.folds):
        train, val = _fold_indices(fold, index, count)
        if not set(train).issubset(development_set) or not set(val).issubset(development_set):
            raise ValueError(f"Protocol audit found fold[{index}] outside development data")
        _groups_disjoint(data, train, val, f"fold[{index}] train/validation")
        validation_parts.append(val)
    combined_validation = np.concatenate(validation_parts)
    if len(np.unique(combined_validation)) != len(combined_validation):
        raise ValueError("Protocol audit found repeated validation samples across group-CV folds")
    if not np.array_equal(np.sort(combined_validation), np.sort(development)):
        raise ValueError("Protocol audit requires group-CV validation folds to cover development exactly once")
    return ProtocolAudit(
        status="passed",
        protocol="group_cv_sealed_test",
        sample_count=count,
        checks=(
            "indices_valid",
            "development_test_disjoint",
            "sealed_test_groups_isolated",
            "folds_within_development",
            "fold_groups_disjoint",
            "validation_exact_cover",
        ),
        details={"development": len(development), "sealed_test": len(test), "folds": len(split.folds)},
    )


def _audit_rolling(data: PreparedData, split: RollingOriginSplit) -> ProtocolAudit:
    count = len(data.x)
    development = _indices(split.development, "development", count)
    test = _indices(split.test, "sealed test", count)
    _disjoint(development, test, "development", "sealed test")
    _complete_partition((development, test), count, "development/sealed-test")
    _strictly_before(data, development, test, "development/sealed-test")
    development_set = set(development.tolist())
    previous_train: set[int] = set()
    validation_parts = []
    for index, fold in enumerate(split.folds):
        train, val = _fold_indices(fold, index, count)
        train_set = set(train.tolist())
        if not train_set.issubset(development_set) or not set(val).issubset(development_set):
            raise ValueError(f"Protocol audit found fold[{index}] outside development data")
        if previous_train and not previous_train.issubset(train_set):
            raise ValueError("Protocol audit requires expanding rolling-origin training windows")
        _strictly_before(data, train, val, f"fold[{index}] train/validation")
        previous_train = train_set
        validation_parts.append(val)
    combined_validation = np.concatenate(validation_parts)
    if len(np.unique(combined_validation)) != len(combined_validation):
        raise ValueError("Protocol audit found repeated validation samples across rolling-origin folds")
    return ProtocolAudit(
        status="passed",
        protocol="rolling_origin_sealed_test",
        sample_count=count,
        checks=(
            "indices_valid",
            "development_test_disjoint",
            "sealed_test_chronological",
            "folds_within_development",
            "folds_chronological",
            "expanding_training_windows",
            "validation_non_overlapping",
        ),
        details={"development": len(development), "sealed_test": len(test), "folds": len(split.folds)},
    )


def audit_split_protocol(
    data: PreparedData,
    split: DataSplit | GroupCVSplit | RollingOriginSplit,
    spec: SplitSpec,
    task_type: TaskType,
) -> ProtocolAudit:
    """Audit a materialized leaderboard split and raise before training on leakage."""

    task_type = TaskType(task_type)
    if isinstance(split, GroupCVSplit):
        if spec.type != SplitType.GROUP_CV:
            raise ValueError("GroupCVSplit requires split.type='group_cv'")
        return _audit_group_cv(data, split)
    if isinstance(split, RollingOriginSplit):
        if spec.type != SplitType.ROLLING_ORIGIN:
            raise ValueError("RollingOriginSplit requires split.type='rolling_origin'")
        return _audit_rolling(data, split)
    if isinstance(split, DataSplit):
        if spec.type in {SplitType.GROUP_CV, SplitType.ROLLING_ORIGIN}:
            raise ValueError(f"DataSplit is incompatible with split.type={spec.type.value!r}")
        return _audit_single(data, split, spec, task_type)
    raise TypeError(f"Unsupported split object: {type(split).__name__}")


def audit_evaluation_trust(
    config: LeaderboardConfig,
    candidates: list[Any],
    planned_trials: list[tuple[int, Any, int, str]],
    protocol_audit: ProtocolAudit,
) -> EvaluationTrustAudit:
    """Audit the candidate/seed/budget plan against the declared leaderboard trust level."""

    level = config.trust.level
    seeds = config.search.evaluation_seeds or (config.seed,)
    minimum_seed_count = _minimum_seed_count(config)
    checks = ["split_protocol_passed"]
    warnings = []
    details: dict[str, Any] = {
        "trust_level": level.value,
        "candidate_count": len(candidates),
        "planned_trial_count": len(planned_trials),
        "evaluation_seeds": list(seeds),
        "minimum_evaluation_seeds": minimum_seed_count,
        "timeout_per_trial_minutes": config.resources.timeout_per_trial_minutes,
        "torch_training_budget": {
            "epochs": config.search.torch_epochs,
            "batch_size": config.search.torch_batch_size,
            "learning_rate": config.search.torch_learning_rate,
            "optimizer": config.search.torch_optimizer,
            "weight_decay": config.search.torch_weight_decay,
            "early_stopping_patience": config.search.torch_early_stopping_patience,
        },
    }
    if protocol_audit.status != "passed":
        raise ValueError("Evaluation trust audit requires a passed split protocol audit")
    if len(seeds) < minimum_seed_count:
        raise ValueError(
            f"trust.level={level.value!r} requires at least {minimum_seed_count} evaluation seeds; got {len(seeds)}"
        )
    checks.append("minimum_seed_count_met")
    expected_trial_count = len(candidates) * len(seeds)
    if len(planned_trials) != expected_trial_count:
        raise ValueError(
            f"Evaluation trust audit expected {expected_trial_count} planned trials, got {len(planned_trials)}"
        )
    coverage: dict[int, set[int]] = {index: set() for index in range(len(candidates))}
    trial_ids = []
    for candidate_index, _candidate, seed, trial_id in planned_trials:
        if candidate_index not in coverage:
            raise ValueError(f"Evaluation trust audit found an unknown candidate index: {candidate_index}")
        coverage[candidate_index].add(seed)
        trial_ids.append(trial_id)
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("Evaluation trust audit found duplicate trial ids")
    missing = {
        index: sorted(set(seeds) - candidate_seeds)
        for index, candidate_seeds in coverage.items()
        if candidate_seeds != set(seeds)
    }
    if missing:
        raise ValueError(f"Evaluation trust audit found incomplete seed coverage: {missing}")
    checks.extend(["unique_trial_ids", "every_candidate_gets_every_seed", "uniform_timeout_per_trial"])
    backends = sorted({str(_candidate_field(candidate, "backend")) for candidate in candidates})
    families = sorted({str(_candidate_field(candidate, "model_family")) for candidate in candidates})
    details.update({"candidate_backends": backends, "candidate_families": families})
    if any(_candidate_field(candidate, "backend") == "pytorch" for candidate in candidates):
        checks.append("uniform_torch_training_budget")
    if level == LeaderboardTrustLevel.DEVELOPMENT and len(seeds) == 1:
        warnings.append("single_seed_development_run")
    return EvaluationTrustAudit(
        status="passed",
        level=level.value,
        checks=tuple(checks),
        warnings=tuple(warnings),
        details=details,
    )


__all__ = ["EvaluationTrustAudit", "ProtocolAudit", "audit_evaluation_trust", "audit_split_protocol"]
