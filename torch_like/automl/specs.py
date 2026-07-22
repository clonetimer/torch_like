"""Strongly typed configuration contracts for leaderboard tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TaskType(StringEnum):
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    RUL_REGRESSION = "rul_regression"
    ANOMALY_DETECTION = "anomaly_detection"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    MULTI_TARGET_REGRESSION = "multi_target_regression"


class SampleMode(StringEnum):
    ROW = "row"
    FULL_SEQUENCE = "full_sequence"
    SLIDING_WINDOW = "sliding_window"


class DatasetFormat(StringEnum):
    CSV = "csv"
    XJTU_SY = "xjtu_sy"
    ESA_ADB = "esa_adb"


class LabelGranularity(StringEnum):
    SAMPLE = "sample"
    SEQUENCE = "sequence"


class SplitType(StringEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    GROUP_CV = "group_cv"
    ROLLING_ORIGIN = "rolling_origin"
    TIME = "time"
    PREDEFINED = "predefined"


class DatasetScale(StringEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    USER_DEFINED = "user_defined"


class SearchStrategy(StringEnum):
    NONE = "none"
    GRID = "grid"
    RANDOM = "random"


class CandidatePolicy(StringEnum):
    PRODUCTION = "production"
    TORCH_LIKE_ONLY = "torch_like_only"
    BASELINE_ONLY = "baseline_only"
    PREFER_TORCH_LIKE = "prefer_torch_like"


class LeaderboardTrustLevel(StringEnum):
    DEVELOPMENT = "development"
    VALIDATED = "validated"


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    label_columns: tuple[str, ...]
    format: DatasetFormat = DatasetFormat.CSV
    labels_path: str | None = None
    label_join_key: str = "sample_id"
    feature_columns: tuple[str, ...] | None = None
    entity_column: str | None = None
    time_column: str | None = None
    split_column: str | None = None
    sample_mode: SampleMode = SampleMode.ROW
    label_granularity: LabelGranularity = LabelGranularity.SAMPLE
    sequence_length: int | None = None
    prediction_length: int | None = None
    stride: int | None = None
    condition: str | None = None
    bearing_ids: tuple[str, ...] | None = None
    max_files_per_entity: int | None = None
    max_windows_per_file: int | None = None
    resample_seconds: int = 30
    split_time_ranges: dict[str, tuple[str, str]] | None = None
    anomaly_categories: tuple[str, ...] = ("Anomaly", "Rare Event")
    cache_dir: str | None = None
    scale: DatasetScale = DatasetScale.USER_DEFINED

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", DatasetFormat(self.format))
        object.__setattr__(self, "sample_mode", SampleMode(self.sample_mode))
        object.__setattr__(self, "label_granularity", LabelGranularity(self.label_granularity))
        object.__setattr__(self, "scale", DatasetScale(self.scale))
        object.__setattr__(self, "label_columns", tuple(self.label_columns))
        if self.feature_columns is not None:
            object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        if self.bearing_ids is not None:
            object.__setattr__(self, "bearing_ids", tuple(self.bearing_ids))
        object.__setattr__(self, "anomaly_categories", tuple(self.anomaly_categories))
        if self.split_time_ranges is not None:
            object.__setattr__(
                self,
                "split_time_ranges",
                {name: tuple(bounds) for name, bounds in self.split_time_ranges.items()},
            )
        if self.format == DatasetFormat.CSV and not self.path.lower().endswith(".csv"):
            raise ValueError("CSV datasets must use a .csv path")
        if not self.label_columns:
            raise ValueError("label_columns must contain at least one column")
        if len(set(self.label_columns)) != len(self.label_columns):
            raise ValueError("label_columns must not contain duplicates")
        if (
            self.format == DatasetFormat.CSV
            and self.sample_mode != SampleMode.ROW
            and not self.entity_column
            and self.prediction_length is None
        ):
            raise ValueError(f"sample_mode={self.sample_mode.value!r} requires entity_column")
        if self.sample_mode == SampleMode.SLIDING_WINDOW:
            if self.sequence_length is None or self.sequence_length <= 0:
                raise ValueError("sliding_window requires a positive sequence_length")
            if self.stride is None or self.stride <= 0:
                raise ValueError("sliding_window requires a positive stride")
        if self.prediction_length is not None and self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive when provided")
        if self.max_files_per_entity is not None and self.max_files_per_entity <= 0:
            raise ValueError("max_files_per_entity must be positive")
        if self.max_windows_per_file is not None and self.max_windows_per_file <= 0:
            raise ValueError("max_windows_per_file must be positive")
        if self.resample_seconds <= 0:
            raise ValueError("resample_seconds must be positive")
        if self.format == DatasetFormat.XJTU_SY:
            if self.sample_mode != SampleMode.SLIDING_WINDOW:
                raise ValueError("xjtu_sy datasets require sample_mode='sliding_window'")
            if self.label_columns != ("rul",):
                raise ValueError("xjtu_sy datasets provide exactly one generated label column: 'rul'")
            if self.labels_path is not None:
                raise ValueError("xjtu_sy labels are derived from file position; labels_path is not supported")
        if self.format == DatasetFormat.ESA_ADB:
            if self.sample_mode != SampleMode.SLIDING_WINDOW:
                raise ValueError("esa_adb datasets require sample_mode='sliding_window'")
            if self.split_time_ranges is None or set(self.split_time_ranges) != {"train", "val", "test"}:
                raise ValueError("esa_adb datasets require train/val/test split_time_ranges")
            if any(len(bounds) != 2 for bounds in self.split_time_ranges.values()):
                raise ValueError("Each ESA-ADB split_time_ranges entry must contain start and end timestamps")
            if not self.feature_columns:
                raise ValueError("esa_adb datasets require feature_columns with selected channel names")
            if tuple(self.label_columns) != tuple(self.feature_columns):
                raise ValueError("esa_adb label_columns must match feature_columns for channel-aware evaluation")
            if not self.anomaly_categories:
                raise ValueError("esa_adb anomaly_categories must not be empty")


@dataclass(frozen=True)
class TaskSpec:
    type: TaskType

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", TaskType(self.type))


@dataclass(frozen=True)
class SplitSpec:
    type: SplitType = SplitType.RANDOM
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    cv_folds: int = 4
    rolling_folds: int = 3
    test_groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", SplitType(self.type))
        if self.test_groups is not None:
            object.__setattr__(self, "test_groups", tuple(self.test_groups))
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("train_ratio, val_ratio, and test_ratio must be positive")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.rolling_folds < 2:
            raise ValueError("rolling_folds must be at least 2")
        if self.test_groups is not None and not self.test_groups:
            raise ValueError("test_groups must be non-empty when provided")


@dataclass(frozen=True)
class ResourceSpec:
    device: str = "cpu"
    max_parallel_trials: int = 1
    max_total_minutes: float = 60.0
    timeout_per_trial_minutes: float = 10.0
    max_cpu_workers: int = 1
    max_ram_gb: float | None = None
    max_vram_gb: float | None = None

    def __post_init__(self) -> None:
        if self.device != "auto":
            try:
                import torch

                torch.device(self.device)
            except (RuntimeError, ValueError) as exc:
                raise ValueError("device must be 'auto' or a valid PyTorch device string") from exc
        if self.max_parallel_trials != 1:
            raise ValueError("First-stage execution currently requires max_parallel_trials=1")
        if min(self.max_total_minutes, self.timeout_per_trial_minutes) <= 0:
            raise ValueError("Trial and task time limits must be positive")
        if self.max_cpu_workers <= 0:
            raise ValueError("max_cpu_workers must be positive")


@dataclass(frozen=True)
class SearchSpec:
    strategy: SearchStrategy = SearchStrategy.RANDOM
    candidate_policy: CandidatePolicy = CandidatePolicy.PRODUCTION
    max_trials: int = 20
    preprocessing: tuple[str, ...] = ("standard", "robust", "none")
    algorithms: tuple[str, ...] | None = None
    evaluation_seeds: tuple[int, ...] | None = None
    pruning_enabled: bool = False
    pruning_target_reduction: float = 0.5
    torch_epochs: int = 10
    torch_batch_size: int = 32
    torch_learning_rate: float = 0.001
    torch_optimizer: str = "adam"
    torch_weight_decay: float = 0.0
    torch_loss: str = "mse"
    torch_grad_clip_norm: float | None = None
    torch_lr_scheduler: str = "none"
    torch_early_stopping_patience: int | None = None
    torch_min_delta: float = 0.0
    torch_restore_best: bool = False
    torch_target_constraint: str = "none"
    anomaly_threshold_candidates: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", SearchStrategy(self.strategy))
        object.__setattr__(self, "candidate_policy", CandidatePolicy(self.candidate_policy))
        object.__setattr__(self, "preprocessing", tuple(self.preprocessing))
        if self.algorithms is not None:
            object.__setattr__(self, "algorithms", tuple(self.algorithms))
        if self.evaluation_seeds is not None:
            object.__setattr__(self, "evaluation_seeds", tuple(int(seed) for seed in self.evaluation_seeds))
            if not self.evaluation_seeds:
                raise ValueError("evaluation_seeds must be non-empty when provided")
            if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
                raise ValueError("evaluation_seeds must not contain duplicates")
            if any(seed < 0 for seed in self.evaluation_seeds):
                raise ValueError("evaluation_seeds must be non-negative")
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.torch_epochs <= 0 or self.torch_batch_size <= 0 or self.torch_learning_rate <= 0:
            raise ValueError("torch_epochs, torch_batch_size, and torch_learning_rate must be positive")
        if self.torch_optimizer not in {"adam", "adamw"}:
            raise ValueError("torch_optimizer must be 'adam' or 'adamw'")
        if self.torch_weight_decay < 0:
            raise ValueError("torch_weight_decay must be non-negative")
        if self.torch_loss not in {"mse", "mae", "huber"}:
            raise ValueError("torch_loss must be one of: mse, mae, huber")
        if self.torch_grad_clip_norm is not None and self.torch_grad_clip_norm <= 0:
            raise ValueError("torch_grad_clip_norm must be positive when provided")
        if self.torch_lr_scheduler not in {"none", "cosine", "plateau"}:
            raise ValueError("torch_lr_scheduler must be one of: none, cosine, plateau")
        if self.torch_early_stopping_patience is not None and self.torch_early_stopping_patience <= 0:
            raise ValueError("torch_early_stopping_patience must be positive when provided")
        if self.torch_min_delta < 0:
            raise ValueError("torch_min_delta must be non-negative")
        if self.torch_target_constraint not in {"none", "unit_interval"}:
            raise ValueError("torch_target_constraint must be 'none' or 'unit_interval'")
        if self.anomaly_threshold_candidates < 10:
            raise ValueError("anomaly_threshold_candidates must be at least 10")
        allowed = {"standard", "robust", "minmax", "none"}
        unknown = set(self.preprocessing) - allowed
        if unknown:
            raise ValueError(f"Unsupported preprocessing choices: {sorted(unknown)}")
        if not 0 < self.pruning_target_reduction < 1:
            raise ValueError("pruning_target_reduction must be in (0, 1)")


@dataclass(frozen=True)
class RankingSpec:
    primary_metric: str | None = None
    additional_metrics: tuple[str, ...] = ()
    target_weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "additional_metrics", tuple(self.additional_metrics))
        if self.target_weights is not None:
            if any(weight < 0 for weight in self.target_weights.values()):
                raise ValueError("target_weights must be non-negative")
            if sum(self.target_weights.values()) <= 0:
                raise ValueError("target_weights must contain a positive total weight")


@dataclass(frozen=True)
class PretrainingSpec:
    enabled: bool = False
    mode: str = "existing_checkpoint"
    allow_download: bool = False
    cache_dir: str = ".cache/checkpoints"

    def __post_init__(self) -> None:
        if self.mode not in {"existing_checkpoint", "self_supervised"}:
            raise ValueError("pretraining mode must be existing_checkpoint or self_supervised")
        if self.allow_download and not self.enabled:
            raise ValueError("allow_download requires pretraining.enabled=true")


@dataclass(frozen=True)
class TrustSpec:
    level: LeaderboardTrustLevel = LeaderboardTrustLevel.DEVELOPMENT
    minimum_evaluation_seeds: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", LeaderboardTrustLevel(self.level))
        if self.minimum_evaluation_seeds is not None and self.minimum_evaluation_seeds <= 0:
            raise ValueError("minimum_evaluation_seeds must be positive when provided")
        if (
            self.level == LeaderboardTrustLevel.VALIDATED
            and self.minimum_evaluation_seeds is not None
            and self.minimum_evaluation_seeds < 3
        ):
            raise ValueError("validated trust requires minimum_evaluation_seeds >= 3")


@dataclass(frozen=True)
class ArtifactSpec:
    output_dir: str = "runs"
    save_predictions: bool = True
    save_checkpoints: bool = True
    trusted_local_joblib_only: bool = True


@dataclass(frozen=True)
class LeaderboardConfig:
    task_id: str
    dataset: DatasetSpec
    task: TaskSpec
    split: SplitSpec = field(default_factory=SplitSpec)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    search: SearchSpec = field(default_factory=SearchSpec)
    ranking: RankingSpec = field(default_factory=RankingSpec)
    pretraining: PretrainingSpec = field(default_factory=PretrainingSpec)
    trust: TrustSpec = field(default_factory=TrustSpec)
    artifacts: ArtifactSpec = field(default_factory=ArtifactSpec)
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.task_id or any(char in self.task_id for char in '<>:"/\\|?*'):
            raise ValueError("task_id must be a non-empty filesystem-safe name")
        labels = len(self.dataset.label_columns)
        task_type = self.task.type
        if task_type == TaskType.MULTILABEL_CLASSIFICATION and labels < 2:
            raise ValueError("multilabel_classification requires at least two label columns")
        if task_type == TaskType.MULTI_TARGET_REGRESSION and labels < 2:
            raise ValueError("multi_target_regression requires at least two label columns")
        if (
            task_type
            in {
                TaskType.TABULAR_CLASSIFICATION,
                TaskType.TIME_SERIES_CLASSIFICATION,
                TaskType.TABULAR_REGRESSION,
                TaskType.RUL_REGRESSION,
            }
            and labels != 1
        ):
            raise ValueError(f"{task_type.value} requires exactly one label column")
        if task_type == TaskType.TIME_SERIES_FORECASTING:
            if self.dataset.sample_mode != SampleMode.SLIDING_WINDOW:
                raise ValueError("time_series_forecasting requires sample_mode='sliding_window'")
            if self.dataset.prediction_length is None:
                raise ValueError("time_series_forecasting requires dataset.prediction_length")
            if self.dataset.time_column is None:
                raise ValueError("time_series_forecasting requires dataset.time_column")
            if self.split.type != SplitType.ROLLING_ORIGIN:
                raise ValueError("time_series_forecasting requires split.type='rolling_origin'")
            if (self.dataset.stride or 0) < self.dataset.prediction_length:
                raise ValueError("forecasting v1 requires stride >= prediction_length to avoid overlapping targets")
        if task_type == TaskType.ANOMALY_DETECTION:
            if self.dataset.format != DatasetFormat.ESA_ADB:
                raise ValueError("anomaly_detection v1 requires dataset.format='esa_adb'")
            if self.split.type != SplitType.PREDEFINED:
                raise ValueError("anomaly_detection requires split.type='predefined'")
        if self.dataset.format == DatasetFormat.XJTU_SY and self.split.type not in {
            SplitType.GROUP,
            SplitType.GROUP_CV,
        }:
            raise ValueError("xjtu_sy leaderboard tasks require split.type='group' or 'group_cv' to isolate bearings")
        if (
            self.dataset.sample_mode == SampleMode.SLIDING_WINDOW
            and self.split.type == SplitType.TIME
            and task_type != TaskType.TIME_SERIES_FORECASTING
        ):
            raise ValueError(
                "time split with overlapping sliding windows is disabled in v1; use predefined raw-data splits"
            )
        if (
            self.dataset.sample_mode == SampleMode.SLIDING_WINDOW
            and self.split.type in {SplitType.RANDOM, SplitType.STRATIFIED}
            and (self.dataset.stride or 0) < (self.dataset.sequence_length or 0)
        ):
            raise ValueError(
                "random or stratified splitting of overlapping windows would leak raw rows; "
                "use group/predefined split or stride >= sequence_length"
            )
        if self.ranking.target_weights is not None:
            if set(self.ranking.target_weights) != set(self.dataset.label_columns):
                raise ValueError("target_weights keys must exactly match dataset.label_columns")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LeaderboardConfig:
        return cls(
            task_id=value["task_id"],
            dataset=DatasetSpec(**value["dataset"]),
            task=TaskSpec(**value["task"]),
            split=SplitSpec(**value.get("split", {})),
            resources=ResourceSpec(**value.get("resources", {})),
            search=SearchSpec(**value.get("search", {})),
            ranking=RankingSpec(**value.get("ranking", {})),
            pretraining=PretrainingSpec(**value.get("pretraining", {})),
            trust=TrustSpec(**value.get("trust", {})),
            artifacts=ArtifactSpec(**value.get("artifacts", {})),
            seed=int(value.get("seed", 42)),
        )

    @property
    def run_dir(self) -> Path:
        return Path(self.artifacts.output_dir) / self.task_id
