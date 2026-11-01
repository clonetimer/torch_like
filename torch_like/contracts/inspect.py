"""Resolve machine-readable contracts for torch_like and sklearn algorithms."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from torch_like.classical import get_estimator_info, has_estimator
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
from torch_like.models.registry import find_model_domain, get_model_class


@dataclass(frozen=True)
class _TaskTemplate:
    task: TaskSpec
    operations: tuple[str, ...]
    status: ContractStatus = ContractStatus.STANDARDIZED


def _f(name: str, required: bool, semantic: str, dtype: str | None = "tensor", rank: int | None = None) -> FieldSpec:
    return FieldSpec(name=name, required=required, semantic=semantic, dtype=dtype, rank=rank)


_TEMPLATES: dict[str, _TaskTemplate] = {
    "DeepState": _TaskTemplate(
        TaskSpec(
            TaskKind.PROBABILISTIC_FORECASTING,
            InputSpec(
                "torch_like.models.forecasting.DeepStateBatch",
                (
                    _f("x_enc", True, "historical univariate values [batch,seq_len,1]", rank=3),
                    _f("x_mark_enc", False, "required when time_feature_dim>0: known context covariates", rank=3),
                    _f("x_dec", False, "optional decoder placeholder; all values ignored", rank=3),
                    _f("x_mark_dec", False, "known future covariates; use final pred_len rows", rank=3),
                    _f(
                        "observed_enc",
                        False,
                        "explicit observed-history mask; omitted means all observed",
                        dtype="bool tensor",
                        rank=3,
                    ),
                    _f("observed_dec", False, "observed future training-label mask", dtype="bool tensor", rank=3),
                    _f(
                        "seasonal_indices_enc",
                        False,
                        "required for configured seasonal periods: explicit phase indices",
                        dtype="integer tensor",
                        rank=3,
                    ),
                    _f(
                        "seasonal_indices_dec",
                        False,
                        "required future phase indices; use final pred_len rows",
                        dtype="integer tensor",
                        rank=3,
                    ),
                    _f(
                        "feat_static_cat",
                        False,
                        "required for configured cardinalities: static category IDs",
                        dtype="integer tensor",
                        rank=2,
                    ),
                    _f("y_true", False, "future labels used only for conditional training likelihood", rank=3),
                ),
            ),
            OutputSpec("tensor", "conditional marginal mean", "(batch,pred_len,1)"),
            target_field="y_true",
            notes=(
                "Each tensor position advances one state step; caller supplies all calendar semantics.",
                "Distribution scale is marginal standard deviation; sample retains temporal dependence.",
            ),
        ),
        ("predict",),
    ),
    "ForecastingModel": _TaskTemplate(
        TaskSpec(
            TaskKind.TIME_SERIES_FORECASTING,
            InputSpec(
                "torch_like.models.forecasting.ForecastingBatch",
                (
                    _f("x_enc", True, "historical input sequence", rank=3),
                    _f("x_mark_enc", False, "encoder time features", rank=3),
                    _f("x_dec", True, "decoder context/future placeholder", rank=3),
                    _f("x_mark_dec", False, "decoder time features", rank=3),
                    _f("x_target_enc", False, "historical target channels", rank=3),
                    _f("x_exog_enc", False, "historical exogenous channels", rank=3),
                    _f("x_target_dec", False, "decoder target channels", rank=3),
                    _f("x_exog_dec", False, "decoder exogenous channels", rank=3),
                ),
            ),
            OutputSpec("tensor", "point forecast", "(batch, pred_len, channels)"),
        ),
        ("predict",),
    ),
    "ProbabilisticForecastingModel": _TaskTemplate(
        TaskSpec(
            TaskKind.PROBABILISTIC_FORECASTING,
            InputSpec(
                "torch_like.models.forecasting.ProbabilisticForecastingBatch",
                (
                    _f("x_enc", True, "historical input sequence", rank=3),
                    _f("x_mark_enc", False, "encoder time features", rank=3),
                    _f("x_dec", True, "decoder context/future placeholder", rank=3),
                    _f("x_mark_dec", False, "decoder time features", rank=3),
                    _f("y_true", False, "forecast target used for training", rank=3),
                ),
            ),
            OutputSpec("tensor", "deterministic summary forecast", "(batch, pred_len, channels)"),
            target_field="y_true",
        ),
        ("predict",),
    ),
    "DiagnosisModel": _TaskTemplate(
        TaskSpec(
            TaskKind.TIME_SERIES_CLASSIFICATION,
            InputSpec(
                "torch_like.models.diagnosis.DiagnosisBatch",
                (
                    _f("x", True, "sensor/time-series input", rank=3),
                    _f("x_mark", False, "time features", rank=3),
                    _f("labels", False, "class labels", dtype="integer tensor", rank=1),
                ),
            ),
            OutputSpec("tensor", "class logits", "(batch, num_classes)"),
            target_field="labels",
        ),
        ("predict", "loss"),
    ),
    "DetectionModel": _TaskTemplate(
        TaskSpec(
            TaskKind.ANOMALY_DETECTION,
            InputSpec(
                "torch_like.models.detection.DetectionBatch",
                (
                    _f("x", True, "input windows", rank=3),
                    _f("x_mark", False, "time features", rank=3),
                ),
            ),
            OutputSpec("tensor", "detector prediction/reconstruction", None),
        ),
        ("predict", "loss", "score"),
    ),
    "GraphModel": _TaskTemplate(
        TaskSpec(
            TaskKind.GRAPH_PREDICTION,
            InputSpec(
                "torch_like.models.graph.GraphBatch",
                (
                    _f("x", True, "node features", rank=3),
                    _f("adjacency", False, "dense adjacency", rank=None),
                    _f("edge_index", False, "sparse edge indices", dtype="integer tensor", rank=2),
                    _f("edge_attr", False, "edge features", rank=None),
                    _f("batch_index", False, "graph membership ids", dtype="integer tensor", rank=None),
                    _f("node_mask", False, "valid-node mask", rank=2),
                ),
            ),
            OutputSpec("tensor", "node-level or graph-level prediction", None),
        ),
        ("predict",),
    ),
    "SequenceModel": _TaskTemplate(
        TaskSpec(
            TaskKind.CAUSAL_LANGUAGE_MODELING,
            InputSpec(
                "torch_like.models.sequence.SequenceBatch",
                (
                    _f("input_ids", True, "token ids", dtype="integer tensor", rank=2),
                    _f("attention_mask", False, "valid-token mask", rank=2),
                    _f("labels", False, "next-token targets", dtype="integer tensor", rank=2),
                    _f("position_ids", False, "token positions", dtype="integer tensor", rank=2),
                    _f("causal_mask", False, "causal attention mask", rank=None),
                ),
            ),
            OutputSpec("tensor", "next-token logits", "(batch, seq_len, vocab_size)"),
            target_field="labels",
        ),
        ("predict",),
    ),
    "Seq2SeqModel": _TaskTemplate(
        TaskSpec(
            TaskKind.SEQ2SEQ,
            InputSpec(
                "torch_like.models.sequence.Seq2SeqBatch",
                (
                    _f("input_ids", True, "source token ids", dtype="integer tensor", rank=2),
                    _f("decoder_input_ids", True, "decoder token ids", dtype="integer tensor", rank=2),
                    _f("attention_mask", False, "source mask", rank=2),
                    _f("decoder_attention_mask", False, "decoder mask", rank=2),
                    _f("labels", False, "target token ids", dtype="integer tensor", rank=2),
                ),
            ),
            OutputSpec("tensor", "decoder logits", "(batch, target_len, vocab_size)"),
            target_field="labels",
        ),
        ("predict",),
    ),
    "VisionModel": _TaskTemplate(
        TaskSpec(
            TaskKind.VISION,
            InputSpec(
                "torch_like.models.vision.VisionBatch",
                (
                    _f("x", True, "image tensor", rank=4),
                    _f("y", False, "task target", rank=None),
                ),
            ),
            OutputSpec("model_output", "classification, dense prediction, or detection output", None),
            target_field="y",
            notes=("Output structure is model-specific within the vision domain.",),
        ),
        ("predict",),
    ),
    "MultimodalModel": _TaskTemplate(
        TaskSpec(
            TaskKind.MULTIMODAL,
            InputSpec(
                "torch_like.models.multimodal.MultimodalBatch",
                (
                    _f("image", True, "image input", rank=4),
                    _f("input_ids", True, "text token ids", dtype="integer tensor", rank=2),
                    _f("attention_mask", False, "text attention mask", rank=2),
                    _f("labels", False, "task target", rank=None),
                ),
            ),
            OutputSpec("model_output", "multimodal model output", None),
            target_field="labels",
        ),
        ("predict",),
    ),
    "GenerativeModel": _TaskTemplate(
        TaskSpec(
            TaskKind.GENERATIVE,
            InputSpec(
                "torch_like.models.generative.GenerativeBatch",
                (
                    _f("x_start", True, "training/inference sample", rank=None),
                    _f("timesteps", True, "diffusion/time index; ignored by models that do not require it", rank=1),
                    _f("noise", False, "optional training noise", rank=None),
                ),
            ),
            OutputSpec(
                "model_output",
                "model-specific reconstruction, score, noise, or generated representation",
                None,
            ),
            notes=("Conditional generators may require additional model-specific keyword arguments.",),
        ),
        ("predict",),
        ContractStatus.PARTIAL,
    ),
    "TabularModel": _TaskTemplate(
        TaskSpec(
            TaskKind.TABULAR_PREDICTION,
            InputSpec(
                "torch_like.models.tabular.TabularBatch",
                (
                    _f("x_cont", True, "continuous features", rank=2),
                    _f("x_cat", False, "categorical feature ids", dtype="integer tensor", rank=2),
                    _f("y", False, "training target", rank=None),
                ),
            ),
            OutputSpec("tensor", "tabular prediction/logits", None),
            target_field="y",
        ),
        ("predict", "loss"),
    ),
    "RecommendationModel": _TaskTemplate(
        TaskSpec(
            TaskKind.RECOMMENDATION,
            InputSpec(
                "torch_like.models.recommendation.RecommendationBatch",
                (
                    _f("user_ids", False, "user ids", dtype="integer tensor", rank=None),
                    _f("item_ids", False, "item ids", dtype="integer tensor", rank=None),
                    _f("x_cat", False, "categorical/context features", rank=None),
                    _f("x_cont", False, "continuous/context features", rank=None),
                    _f("y", False, "training target", rank=None),
                ),
            ),
            OutputSpec("tensor", "ranking/CTR score or logits", None),
            target_field="y",
        ),
        ("predict", "loss"),
    ),
    "AudioModel": _TaskTemplate(
        TaskSpec(
            TaskKind.AUDIO,
            InputSpec(
                "torch_like.models.audio.AudioBatch",
                (
                    _f("x", True, "waveform or acoustic features", rank=None),
                    _f("lengths", False, "valid input lengths", dtype="integer tensor", rank=1),
                    _f("y", False, "training target", rank=None),
                ),
            ),
            OutputSpec("tensor", "audio model prediction/representation", None),
            target_field="y",
        ),
        ("predict", "loss"),
    ),
    "ReinforcementModel": _TaskTemplate(
        TaskSpec(
            TaskKind.REINFORCEMENT_LEARNING,
            InputSpec(
                "torch_like.models.reinforcement.ReinforcementBatch",
                (
                    _f("states", True, "environment states", rank=None),
                    _f("actions", False, "actions", rank=None),
                    _f("rewards", False, "rewards", rank=None),
                    _f("next_states", False, "next states", rank=None),
                    _f("dones", False, "terminal flags", rank=None),
                    _f("returns", False, "returns", rank=None),
                    _f("returns_to_go", False, "trajectory returns-to-go", rank=None),
                ),
            ),
            OutputSpec("model_output", "policy, value, Q-value, or action output", None),
            notes=("Action semantics vary by RL algorithm; use runtime.act() only when advertised.",),
        ),
        (),
        ContractStatus.PARTIAL,
    ),
    "RULModel": _TaskTemplate(
        TaskSpec(
            TaskKind.RUL_REGRESSION,
            InputSpec(
                "torch_like.models.RULBatch",
                (
                    _f("x", True, "sensor sequence", rank=3),
                    _f("x_mark", False, "time features", rank=3),
                    _f("targets", False, "remaining-useful-life target", rank=None),
                ),
            ),
            OutputSpec("tensor", "remaining useful life prediction", None),
            target_field="targets",
        ),
        ("predict", "loss"),
    ),
}

_PLACEHOLDER_METHODS = {
    ("ProbabilisticForecastingModel", "predict_distribution"),
    ("ProbabilisticForecastingModel", "sample"),
    ("ProbabilisticForecastingModel", "fit_step"),
    ("SequenceModel", "loss_step"),
    ("SequenceModel", "generate"),
    ("Seq2SeqModel", "loss_step"),
    ("Seq2SeqModel", "generate"),
}


def _owner_name(cls: type[Any], method_name: str) -> str | None:
    for base in cls.__mro__:
        if method_name in base.__dict__:
            return base.__name__
    return None


def _has_concrete_method(cls: type[Any], method_name: str) -> bool:
    owner = _owner_name(cls, method_name)
    if owner is None:
        return False
    return (owner, method_name) not in _PLACEHOLDER_METHODS


def _template_for_class(cls: type[Any]) -> _TaskTemplate:
    # Order matters because ProbabilisticForecastingModel and Seq2SeqModel need
    # to win before broader domain assumptions.
    for base in cls.__mro__:
        template = _TEMPLATES.get(base.__name__)
        if template is not None:
            return template
    return _TaskTemplate(
        TaskSpec(
            TaskKind.CUSTOM,
            InputSpec(None, ()),
            OutputSpec("model_output", "custom model output", None),
            notes=("No built-in task contract was resolved for this class.",),
        ),
        (),
        ContractStatus.PARTIAL,
    )


def _operations_for_class(cls: type[Any], template: _TaskTemplate) -> tuple[str, ...]:
    operations = list(template.operations)
    # GenerativeModel's generic predict(batch) can only forward x_start and
    # timesteps. A subclass such as ControlNet may add another required forward
    # argument; in that case the shared predict contract must fail closed.
    if "predict" in operations and _owner_name(cls, "predict") == "GenerativeModel":
        parameters = [
            parameter for name, parameter in inspect.signature(cls.forward).parameters.items() if name != "self"
        ]
        required = [
            parameter
            for parameter in parameters
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        if len(required) > 2:
            operations.remove("predict")
    if _has_concrete_method(cls, "loss_step") or _has_concrete_method(cls, "fit_step"):
        operations.append("loss")
    if _has_concrete_method(cls, "generate"):
        operations.append("generate")
    if _has_concrete_method(cls, "sample"):
        operations.append("sample")
    if _has_concrete_method(cls, "predict_distribution"):
        operations.append("distribution")
    if _has_concrete_method(cls, "score_batch"):
        operations.append("score")
    if any(_has_concrete_method(cls, name) for name in ("act", "greedy_action", "predict_action", "sample_action")):
        operations.append("act")
    return tuple(dict.fromkeys(operations))


def _torch_model_spec(name: str, domain: str | None = None) -> ModelSpec:
    resolved_domain = domain or find_model_domain(name)
    cls = get_model_class(name, domain=resolved_domain)
    template = _template_for_class(cls)
    return ModelSpec(
        name=name,
        backend=BackendKind.TORCH_LIKE,
        domain=resolved_domain,
        class_path=f"{cls.__module__}.{cls.__qualname__}",
        task=template.task,
        operations=_operations_for_class(cls, template),
        contract_status=template.status,
    )


def _sklearn_model_spec(name: str) -> ModelSpec:
    info = get_estimator_info(name)
    kind = info["kind"]
    task_kind = {
        "classification": TaskKind.CLASSIFICATION,
        "regression": TaskKind.REGRESSION,
        "anomaly_detection": TaskKind.ANOMALY_DETECTION,
        "clustering": TaskKind.CLUSTERING,
        "decomposition": TaskKind.DECOMPOSITION,
    }.get(kind, TaskKind.CUSTOM)
    if kind in {"clustering", "decomposition"}:
        target = None
    else:
        target = "y"
    operations = ["fit"]
    if kind == "decomposition":
        operations.append("transform")
    elif kind == "clustering":
        operations.append("fit_predict")
        if name != "DBSCAN":
            operations.append("predict")
        if name == "KMeans":
            operations.append("transform")
    else:
        operations.append("predict")
    return ModelSpec(
        name=name,
        backend=BackendKind.SKLEARN,
        domain="classical",
        class_path=info["class_path"],
        task=TaskSpec(
            task_kind,
            InputSpec(
                None,
                (
                    FieldSpec("X", True, "feature matrix", "array-like", 2),
                    FieldSpec("y", target is not None, "target values", "array-like", None),
                ),
            ),
            OutputSpec("array", f"scikit-learn {kind} output", None),
            target_field=target,
            notes=("Native scikit-learn fit/predict conventions are preserved.",),
        ),
        operations=tuple(operations),
    )


def get_model_spec(
    target: str | Any,
    *,
    backend: str = "auto",
    domain: str | None = None,
) -> ModelSpec:
    """Resolve the orchestration contract for a model name or model instance.

    ``backend='auto'`` follows :func:`torch_like.create`: built-in torch_like
    names win, then the curated sklearn catalog is checked.
    """

    if isinstance(target, str):
        name = target
        normalized = backend.strip().lower()
        if normalized not in {"auto", "torch_like", "torch", "pytorch", "sklearn", "classical"}:
            raise ValueError("backend must be one of: auto, torch_like, torch, pytorch, sklearn, classical")
        if normalized in {"sklearn", "classical"}:
            if domain is not None:
                raise ValueError("domain is only valid for torch_like/PyTorch models")
            return _sklearn_model_spec(name)
        if normalized in {"torch_like", "torch", "pytorch"} or domain is not None:
            return _torch_model_spec(name, domain)
        try:
            return _torch_model_spec(name)
        except KeyError:
            if has_estimator(name):
                return _sklearn_model_spec(name)
            raise KeyError(
                f"Unknown algorithm '{name}' in torch_like models and sklearn compatibility catalog"
            ) from None

    cls = target if isinstance(target, type) else type(target)
    class_module = getattr(cls, "__module__", "")
    if class_module.startswith("sklearn."):
        name = cls.__name__
        if has_estimator(name):
            return _sklearn_model_spec(name)
        return ModelSpec(
            name=name,
            backend=BackendKind.SKLEARN,
            domain="classical",
            class_path=f"{cls.__module__}.{cls.__qualname__}",
            task=TaskSpec(
                TaskKind.CUSTOM,
                InputSpec(None, (FieldSpec("X", True, "feature matrix", "array-like", 2),)),
                OutputSpec("array", "native sklearn estimator output", None),
                notes=("Estimator is not in torch_like's curated sklearn catalog.",),
            ),
            operations=tuple(
                name for name in ("fit", "predict", "transform", "fit_predict") if callable(getattr(cls, name, None))
            ),
            contract_status=ContractStatus.PARTIAL,
        )

    name = cls.__name__
    try:
        resolved_domain = domain or find_model_domain(name)
    except KeyError:
        resolved_domain = domain or "custom"
    template = _template_for_class(cls)
    return ModelSpec(
        name=name,
        backend=BackendKind.TORCH_LIKE,
        domain=resolved_domain,
        class_path=f"{cls.__module__}.{cls.__qualname__}",
        task=template.task,
        operations=_operations_for_class(cls, template),
        contract_status=template.status,
    )


# More discoverable alias for external orchestration code.
describe = get_model_spec


__all__ = ["get_model_spec", "describe"]
