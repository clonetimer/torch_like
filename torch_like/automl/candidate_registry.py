"""Task-aware candidate registry for AutoML model discovery."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from torch_like.automl.specs import TaskType


@dataclass(frozen=True)
class CandidateSpec:
    """Describe one executable candidate family and its audited search defaults."""

    name: str
    task_types: tuple[TaskType, ...]
    backend: str
    parameter_grid: tuple[dict[str, Any], ...]
    requires_sequence: bool = False
    requires_equal_lengths: bool = False
    optional_dependency: str | None = None
    automl_default: bool = True

    def supports(self, task_type: TaskType, *, is_sequence: bool, equal_sequence_lengths: bool) -> bool:
        return (
            task_type in self.task_types
            and (not self.requires_sequence or is_sequence)
            and (not self.requires_equal_lengths or equal_sequence_lengths)
        )


_LOCK = RLock()
_CANDIDATES: dict[str, CandidateSpec] = {}


def register_candidate(spec: CandidateSpec) -> CandidateSpec:
    """Register an AutoML adapter without changing candidate planning code."""

    if not spec.name:
        raise ValueError("Candidate name must be non-empty")
    if not spec.task_types or not spec.parameter_grid:
        raise ValueError(f"Candidate {spec.name!r} requires task types and a non-empty parameter grid")
    with _LOCK:
        existing = _CANDIDATES.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f"AutoML candidate {spec.name!r} is already registered")
        _CANDIDATES[spec.name] = spec
    return spec


def list_candidate_specs(
    task_type: TaskType,
    *,
    is_sequence: bool,
    equal_sequence_lengths: bool,
    include_experimental: bool = False,
) -> tuple[CandidateSpec, ...]:
    """Return compatible candidates in stable registration order."""

    with _LOCK:
        specs = tuple(_CANDIDATES.values())
    return tuple(
        spec
        for spec in specs
        if spec.supports(
            task_type,
            is_sequence=is_sequence,
            equal_sequence_lengths=equal_sequence_lengths,
        )
        and (spec.automl_default or include_experimental)
    )


_CLASSIFICATION = (
    TaskType.TABULAR_CLASSIFICATION,
    TaskType.TIME_SERIES_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
)
_REGRESSION = (
    TaskType.TABULAR_REGRESSION,
    TaskType.RUL_REGRESSION,
    TaskType.MULTI_TARGET_REGRESSION,
)


def _register_builtins() -> None:
    definitions = (
        CandidateSpec("LogisticRegression", _CLASSIFICATION, "sklearn", tuple({"C": v} for v in (0.1, 1.0, 10.0))),
        CandidateSpec(
            "RandomForestClassifier",
            _CLASSIFICATION,
            "sklearn",
            tuple({"n_estimators": 100, "max_depth": depth} for depth in (None, 12)),
        ),
        CandidateSpec("ExtraTreesClassifier", _CLASSIFICATION, "sklearn", ({"n_estimators": 100, "max_depth": None},)),
        CandidateSpec(
            "HistGradientBoostingClassifier",
            _CLASSIFICATION,
            "sklearn",
            tuple({"learning_rate": value} for value in (0.05, 0.1)),
        ),
        CandidateSpec(
            "LightGBMClassifier",
            _CLASSIFICATION,
            "sklearn",
            (
                {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
                {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15},
            ),
            optional_dependency="lightgbm",
        ),
        CandidateSpec(
            "FTTransformer",
            (TaskType.TABULAR_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION),
            "pytorch",
            ({"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64},),
        ),
        CandidateSpec(
            "TabNet",
            (TaskType.TABULAR_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION),
            "pytorch",
            ({"d_model": 32, "n_steps": 3},),
        ),
        CandidateSpec(
            "NODE",
            (TaskType.TABULAR_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION),
            "pytorch",
            ({"d_model": 32, "num_trees": 16, "tree_depth": 4},),
        ),
        CandidateSpec(
            "SAINT",
            (TaskType.TABULAR_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION),
            "pytorch",
            ({"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "use_row_attention": False},),
        ),
        CandidateSpec(
            "Ridge",
            _REGRESSION + (TaskType.TIME_SERIES_FORECASTING,),
            "sklearn",
            tuple({"alpha": v} for v in (0.1, 1.0, 10.0)),
        ),
        CandidateSpec(
            "RandomForestRegressor",
            _REGRESSION,
            "sklearn",
            tuple({"n_estimators": 100, "max_depth": depth} for depth in (None, 12)),
        ),
        CandidateSpec("ExtraTreesRegressor", _REGRESSION, "sklearn", ({"n_estimators": 100, "max_depth": None},)),
        CandidateSpec(
            "HistGradientBoostingRegressor",
            _REGRESSION,
            "sklearn",
            tuple({"learning_rate": value} for value in (0.05, 0.1)),
        ),
        CandidateSpec(
            "LightGBMRegressor",
            _REGRESSION,
            "sklearn",
            (
                {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
                {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15},
            ),
            optional_dependency="lightgbm",
        ),
        CandidateSpec(
            "InceptionTime",
            (TaskType.TIME_SERIES_CLASSIFICATION,),
            "pytorch",
            ({"d_model": 32, "num_blocks": 3},),
            True,
            True,
        ),
        CandidateSpec("ROCKET", (TaskType.TIME_SERIES_CLASSIFICATION,), "pytorch", ({"num_kernels": 128},), True, True),
        CandidateSpec("THGNN", (TaskType.RUL_REGRESSION,), "pytorch", ({"d_model": 32, "graph_depth": 1},), True, True),
        CandidateSpec(
            "InformerRUL",
            (TaskType.RUL_REGRESSION,),
            "pytorch",
            ({"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64},),
            True,
            True,
        ),
        CandidateSpec(
            "AutoformerRUL",
            (TaskType.RUL_REGRESSION,),
            "pytorch",
            ({"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64},),
            True,
            True,
        ),
        CandidateSpec(
            "LogTransRUL",
            (TaskType.RUL_REGRESSION,),
            "pytorch",
            ({"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64},),
            True,
            True,
        ),
        CandidateSpec(
            "TCNRUL",
            (TaskType.RUL_REGRESSION,),
            "pytorch",
            ({"d_model": 32, "e_layers": 2, "d_ff": 64, "kernel_size": 5},),
            True,
            True,
        ),
        CandidateSpec(
            "PatchTSTRUL",
            (TaskType.RUL_REGRESSION,),
            "pytorch",
            ({"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64, "patch_len": 16, "patch_stride": 8},),
            True,
            True,
        ),
        CandidateSpec(
            "TransformerRUL",
            (TaskType.RUL_REGRESSION,),
            "pytorch",
            ({"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64},),
            True,
            True,
        ),
        CandidateSpec("LastValue", (TaskType.TIME_SERIES_FORECASTING,), "baseline", ({},), True, True),
        CandidateSpec(
            "DLinear",
            (TaskType.TIME_SERIES_FORECASTING,),
            "pytorch",
            ({"moving_avg": 25, "individual": True, "use_norm": True},),
            True,
            True,
        ),
        CandidateSpec(
            "PatchTST",
            (TaskType.TIME_SERIES_FORECASTING,),
            "pytorch",
            ({"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64, "patch_len": 16},),
            True,
            True,
        ),
        CandidateSpec(
            "TSMixer", (TaskType.TIME_SERIES_FORECASTING,), "pytorch", ({"e_layers": 2, "d_ff": 64},), True, True
        ),
    )
    for definition in definitions:
        register_candidate(definition)


_register_builtins()


__all__ = ["CandidateSpec", "list_candidate_specs", "register_candidate"]
