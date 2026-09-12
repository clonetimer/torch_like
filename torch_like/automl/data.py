"""CSV ingestion, profiling, sampling, fingerprinting, and leakage-aware splitting."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from torch_like.automl.specs import DatasetFormat, DatasetSpec, SampleMode, SplitSpec, SplitType, TaskType


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    samples: int
    features: int
    labels: int
    entities: int | None
    missing_values: int
    duplicate_rows: int
    sample_mode: str
    sequence_min_length: int | None
    sequence_max_length: int | None
    fingerprint: str
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedData:
    x: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    groups: np.ndarray | None
    timestamps: np.ndarray | None
    predefined_splits: np.ndarray | None
    feature_columns: tuple[str, ...]
    is_sequence: bool
    sequence_lengths: tuple[int, ...]
    profile: DatasetProfile


@dataclass(frozen=True)
class DataSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "train": self.train.tolist(),
            "val": self.val.tolist(),
            "test": self.test.tolist(),
        }


@dataclass(frozen=True)
class ValidationFold:
    train: np.ndarray
    val: np.ndarray

    def to_dict(self) -> dict[str, list[int]]:
        return {"train": self.train.tolist(), "val": self.val.tolist()}


@dataclass(frozen=True)
class GroupCVSplit:
    development: np.ndarray
    test: np.ndarray
    folds: tuple[ValidationFold, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "development": self.development.tolist(),
            "test": self.test.tolist(),
            "folds": [fold.to_dict() for fold in self.folds],
        }


@dataclass(frozen=True)
class RollingOriginSplit:
    development: np.ndarray
    test: np.ndarray
    folds: tuple[ValidationFold, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "development": self.development.tolist(),
            "test": self.test.tolist(),
            "folds": [fold.to_dict() for fold in self.folds],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(spec: DatasetSpec) -> str:
    digest = hashlib.sha256()
    for role, path in [("data", Path(spec.path)), ("labels", Path(spec.labels_path) if spec.labels_path else None)]:
        if path is not None:
            digest.update(role.encode("ascii"))
            digest.update(_sha256_file(path).encode("ascii"))
    fingerprint_spec = {
        "labels": spec.label_columns,
        "features": spec.feature_columns,
        "entity": spec.entity_column,
        "time": spec.time_column,
        "sample_mode": spec.sample_mode.value,
        "sequence_length": spec.sequence_length,
        "prediction_length": spec.prediction_length,
        "stride": spec.stride,
        "resample_seconds": spec.resample_seconds,
        "split_time_ranges": spec.split_time_ranges,
        "anomaly_categories": spec.anomaly_categories,
    }
    digest.update(repr(fingerprint_spec).encode("utf-8"))
    return digest.hexdigest()


def _selected_positions(count: int, limit: int | None) -> list[int]:
    if limit is None or limit >= count:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return sorted(set(np.linspace(0, count - 1, num=limit, dtype=int).tolist()))


def _xjtu_files(spec: DatasetSpec) -> list[tuple[str, Path, int, int]]:
    root = Path(spec.path)
    if not root.is_dir():
        raise FileNotFoundError(f"XJTU-SY root directory does not exist: {root}")
    conditions = [root / spec.condition] if spec.condition else sorted(path for path in root.iterdir() if path.is_dir())
    selected_bearings = set(spec.bearing_ids or ())
    records: list[tuple[str, Path, int, int]] = []
    for condition_path in conditions:
        if not condition_path.is_dir():
            raise FileNotFoundError(f"XJTU-SY condition directory does not exist: {condition_path}")
        for bearing_path in sorted(path for path in condition_path.iterdir() if path.is_dir()):
            if selected_bearings and bearing_path.name not in selected_bearings:
                continue
            files = sorted(
                (path for path in bearing_path.glob("*.csv") if path.stem.isdigit()),
                key=lambda path: int(path.stem),
            )
            if not files:
                continue
            bearing_id = f"{condition_path.name}/{bearing_path.name}"
            for position in _selected_positions(len(files), spec.max_files_per_entity):
                records.append((bearing_id, files[position], position + 1, len(files)))
    if not records:
        raise ValueError("No XJTU-SY bearing CSV files matched the dataset specification")
    return records


def _xjtu_fingerprint(spec: DatasetSpec, records: list[tuple[str, Path, int, int]]) -> str:
    digest = hashlib.sha256()
    root = Path(spec.path)
    for _, path, _, _ in records:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    digest.update(repr(spec).encode("utf-8"))
    return digest.hexdigest()


def prepare_xjtu_sy_dataset(spec: DatasetSpec) -> PreparedData:
    """Materialize bounded, evenly distributed windows from XJTU-SY acquisition files."""

    if spec.format != DatasetFormat.XJTU_SY:
        raise ValueError("prepare_xjtu_sy_dataset requires format='xjtu_sy'")
    records = _xjtu_files(spec)
    sequence_length = int(spec.sequence_length or 0)
    stride = int(spec.stride or 0)
    feature_columns = spec.feature_columns or (
        "Horizontal_vibration_signals",
        "Vertical_vibration_signals",
    )
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    missing_values = 0
    raw_rows = 0
    for bearing_id, path, minute, lifetime in records:
        frame = pd.read_csv(path, usecols=list(feature_columns))
        values = frame.to_numpy(dtype=np.float32)
        raw_rows += len(values)
        missing_values += int(np.isnan(values).sum())
        if len(values) < sequence_length:
            raise ValueError(f"{path} has {len(values)} rows, shorter than sequence_length={sequence_length}")
        starts = list(range(0, len(values) - sequence_length + 1, stride))
        starts = [starts[index] for index in _selected_positions(len(starts), spec.max_windows_per_file)]
        for start in starts:
            end = start + sequence_length
            sequences.append(values[start:end])
            labels.append(max(0.0, float(lifetime - minute) / float(lifetime)))
            sample_ids.append(f"{bearing_id}/{path.name}:{start}:{end}")
            groups.append(bearing_id)
    x = _object_array(sequences)
    y = np.asarray(labels, dtype=np.float32)
    lengths = tuple(sequence_length for _ in sequences)
    warnings = (
        "XJTU-SY RUL is generated as (lifetime_files - file_index) / lifetime_files.",
        "Files/windows may be deterministically subsampled; inspect max_files_per_entity and max_windows_per_file.",
    )
    profile = DatasetProfile(
        rows=raw_rows,
        samples=len(x),
        features=len(feature_columns),
        labels=1,
        entities=len(set(groups)),
        missing_values=missing_values,
        duplicate_rows=0,
        sample_mode=spec.sample_mode.value,
        sequence_min_length=sequence_length,
        sequence_max_length=sequence_length,
        fingerprint=_xjtu_fingerprint(spec, records),
        feature_columns=tuple(feature_columns),
        label_columns=("rul",),
        warnings=warnings,
    )
    return PreparedData(
        x=x,
        y=y,
        sample_ids=np.asarray(sample_ids),
        groups=np.asarray(groups),
        timestamps=None,
        predefined_splits=None,
        feature_columns=tuple(feature_columns),
        is_sequence=True,
        sequence_lengths=lengths,
        profile=profile,
    )


_FORMAT_LOADERS: dict[str, Callable[[DatasetSpec], PreparedData]] = {}


def register_dataset_format(format: DatasetFormat, loader: Callable[[DatasetSpec], PreparedData]) -> None:
    """Register a loader for a dataset format.

    Built-in formats are registered at import time; third parties can add their
    own format without editing this module.
    """
    if not callable(loader):
        raise TypeError("loader must be callable")
    _FORMAT_LOADERS[str(format)] = loader


def prepare_dataset(spec: DatasetSpec) -> PreparedData:
    loader = _FORMAT_LOADERS.get(str(spec.format))
    if loader is None:
        raise ValueError(f"Unsupported dataset format: {spec.format}")
    return loader(spec)


def _read_and_merge(spec: DatasetSpec) -> pd.DataFrame:
    data_path = Path(spec.path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {data_path}")
    frame = pd.read_csv(data_path)
    if frame.empty:
        raise ValueError("Dataset CSV is empty")
    if spec.labels_path:
        labels_path = Path(spec.labels_path)
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels CSV does not exist: {labels_path}")
        labels = pd.read_csv(labels_path)
        if spec.label_join_key not in frame or spec.label_join_key not in labels:
            raise ValueError(f"label_join_key={spec.label_join_key!r} must exist in data and labels CSV files")
        if labels[spec.label_join_key].duplicated().any():
            raise ValueError("labels.csv join keys must be unique")
        missing_label_columns = set(spec.label_columns) - set(labels.columns)
        if missing_label_columns:
            raise ValueError(f"labels.csv is missing label columns: {sorted(missing_label_columns)}")
        overlap = (set(spec.label_columns) & set(frame.columns)) - {spec.label_join_key}
        if overlap:
            frame = frame.drop(columns=sorted(overlap))
        frame = frame.merge(
            labels[[spec.label_join_key, *spec.label_columns]],
            on=spec.label_join_key,
            how="left",
            validate="many_to_one",
        )
    missing_labels = set(spec.label_columns) - set(frame.columns)
    if missing_labels:
        raise ValueError(f"Dataset is missing label columns: {sorted(missing_labels)}")
    if frame[list(spec.label_columns)].isna().any().any():
        raise ValueError("Label columns contain missing values or unmatched labels.csv rows")
    return frame


def _resolve_feature_columns(frame: pd.DataFrame, spec: DatasetSpec) -> tuple[str, ...]:
    if spec.feature_columns is not None:
        missing = set(spec.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing feature columns: {sorted(missing)}")
        columns = tuple(spec.feature_columns)
    else:
        excluded = set() if spec.prediction_length is not None else set(spec.label_columns)
        excluded.update(
            column
            for column in (spec.entity_column, spec.time_column, spec.split_column, spec.label_join_key)
            if column
        )
        columns = tuple(column for column in frame.select_dtypes(include=[np.number]).columns if column not in excluded)
    if not columns:
        raise ValueError("No numeric feature columns were selected")
    if spec.prediction_length is not None:
        missing_targets = set(spec.label_columns) - set(columns)
        if missing_targets:
            raise ValueError(f"Forecasting history must include target columns: {sorted(missing_targets)}")
        columns = (*spec.label_columns, *(column for column in columns if column not in spec.label_columns))
    if spec.prediction_length is None and set(columns) & set(spec.label_columns):
        raise ValueError("Feature and label columns must not overlap")
    return columns


def _object_array(items: list[np.ndarray]) -> np.ndarray:
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def _sample_rows(frame: pd.DataFrame, spec: DatasetSpec, features: tuple[str, ...]):
    x = frame.loc[:, features].to_numpy(dtype=np.float64)
    y = frame.loc[:, spec.label_columns].to_numpy()
    ids = (
        frame[spec.label_join_key].astype(str).to_numpy()
        if spec.label_join_key in frame
        else np.asarray([f"row-{index}" for index in range(len(frame))])
    )
    groups = frame[spec.entity_column].to_numpy() if spec.entity_column else None
    timestamps = frame[spec.time_column].to_numpy() if spec.time_column else None
    splits = frame[spec.split_column].astype(str).str.lower().to_numpy() if spec.split_column else None
    return x, y, ids, groups, timestamps, splits, ()


def _sample_sequences(frame: pd.DataFrame, spec: DatasetSpec, features: tuple[str, ...]):
    if spec.entity_column and spec.entity_column not in frame:
        raise ValueError(f"Dataset is missing entity_column={spec.entity_column!r}")
    if spec.time_column and spec.time_column not in frame:
        raise ValueError(f"Dataset is missing time_column={spec.time_column!r}")
    sequences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    groups: list[Any] = []
    timestamps: list[Any] = []
    splits: list[str] = []
    lengths: list[int] = []
    grouped = (
        frame.groupby(spec.entity_column, sort=False, dropna=False) if spec.entity_column else [("__series__", frame)]
    )
    for entity, entity_frame in grouped:
        if spec.time_column:
            entity_frame = entity_frame.sort_values(spec.time_column, kind="stable")
        values = entity_frame.loc[:, features].to_numpy(dtype=np.float64)
        targets = entity_frame.loc[:, spec.label_columns].to_numpy()
        if spec.prediction_length is not None:
            context_length = int(spec.sequence_length or 0)
            prediction_length = int(spec.prediction_length)
            stride = int(spec.stride or 0)
            windows = [
                (start, start + context_length, start + context_length + prediction_length)
                for start in range(0, len(entity_frame) - context_length - prediction_length + 1, stride)
            ]
        elif spec.sample_mode == SampleMode.FULL_SEQUENCE:
            windows = [(0, len(entity_frame))]
        else:
            length = int(spec.sequence_length or 0)
            stride = int(spec.stride or 0)
            windows = [(start, start + length) for start in range(0, len(entity_frame) - length + 1, stride)]
        for window in windows:
            if spec.prediction_length is not None:
                start, context_end, end = window
            else:
                start, end = window
                context_end = end
            if end <= start:
                continue
            sequences.append(values[start:context_end])
            labels.append(targets[context_end:end] if spec.prediction_length is not None else targets[end - 1])
            sample_ids.append(f"{entity}:{start}:{end}")
            groups.append(entity)
            lengths.append(context_end - start)
            if spec.time_column:
                timestamps.append(entity_frame.iloc[end - 1][spec.time_column])
            if spec.split_column:
                split_values = entity_frame.iloc[start:end][spec.split_column].astype(str).str.lower().unique()
                if len(split_values) != 1:
                    raise ValueError(f"Sequence {entity}:{start}:{end} crosses predefined split boundaries")
                splits.append(split_values[0])
    if not sequences:
        raise ValueError("Sampling produced no sequences; check sequence_length, stride, and entity sizes")
    return (
        _object_array(sequences),
        np.asarray(labels),
        np.asarray(sample_ids),
        np.asarray(groups) if spec.entity_column else None,
        np.asarray(timestamps) if spec.time_column else None,
        np.asarray(splits) if spec.split_column else None,
        tuple(lengths),
    )


def prepare_csv_dataset(spec: DatasetSpec) -> PreparedData:
    frame = _read_and_merge(spec)
    features = _resolve_feature_columns(frame, spec)
    missing_values = int(frame.loc[:, features].isna().sum().sum())
    warnings: list[str] = []
    if spec.entity_column and spec.sample_mode == SampleMode.ROW:
        warnings.append("row sampling retains entity ids; random splitting may leak entity-specific information")
    if spec.sample_mode == SampleMode.ROW:
        x, y, ids, groups, timestamps, splits, lengths = _sample_rows(frame, spec, features)
    else:
        x, y, ids, groups, timestamps, splits, lengths = _sample_sequences(frame, spec, features)
    profile = DatasetProfile(
        rows=len(frame),
        samples=len(x),
        features=len(features),
        labels=len(spec.label_columns),
        entities=int(frame[spec.entity_column].nunique()) if spec.entity_column else None,
        missing_values=missing_values,
        duplicate_rows=int(frame.duplicated().sum()),
        sample_mode=spec.sample_mode.value,
        sequence_min_length=min(lengths) if lengths else None,
        sequence_max_length=max(lengths) if lengths else None,
        fingerprint=dataset_fingerprint(spec),
        feature_columns=features,
        label_columns=spec.label_columns,
        warnings=tuple(warnings),
    )
    if spec.prediction_length is None and y.shape[1] == 1:
        y = y[:, 0]
    return PreparedData(
        x=x,
        y=y,
        sample_ids=ids,
        groups=groups,
        timestamps=timestamps,
        predefined_splits=splits,
        feature_columns=features,
        is_sequence=spec.sample_mode != SampleMode.ROW,
        sequence_lengths=lengths,
        profile=profile,
    )


def _split_remaining(indices: np.ndarray, val_fraction: float, seed: int, stratify=None):
    val, test = train_test_split(
        indices,
        train_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.sort(val), np.sort(test)


def split_dataset(data: PreparedData, spec: SplitSpec, task_type: TaskType) -> DataSplit:
    indices = np.arange(len(data.x))
    val_within_remainder = spec.val_ratio / (spec.val_ratio + spec.test_ratio)
    if spec.type == SplitType.PREDEFINED:
        if data.predefined_splits is None:
            raise ValueError("predefined split requires dataset.split_column")
        train = indices[data.predefined_splits == "train"]
        val = indices[data.predefined_splits == "val"]
        test = indices[data.predefined_splits == "test"]
    elif spec.type == SplitType.TIME:
        if data.timestamps is None:
            raise ValueError("time split requires dataset.time_column")
        ordered = indices[np.argsort(data.timestamps, kind="stable")]
        train_end = int(len(ordered) * spec.train_ratio)
        val_end = train_end + int(len(ordered) * spec.val_ratio)
        train, val, test = ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]
    elif spec.type == SplitType.GROUP:
        if data.groups is None:
            raise ValueError("group split requires dataset.entity_column")
        first = GroupShuffleSplit(n_splits=1, train_size=spec.train_ratio, random_state=spec.seed)
        train_position, remainder_position = next(first.split(indices, groups=data.groups))
        remainder = indices[remainder_position]
        second = GroupShuffleSplit(n_splits=1, train_size=val_within_remainder, random_state=spec.seed + 1)
        val_position, test_position = next(second.split(remainder, groups=data.groups[remainder]))
        train, val, test = indices[train_position], remainder[val_position], remainder[test_position]
    else:
        stratify = None
        if spec.type == SplitType.STRATIFIED:
            if task_type == TaskType.MULTILABEL_CLASSIFICATION or data.y.ndim != 1:
                raise ValueError("v1 stratified split supports only single-label classification")
            stratify = data.y
        train, remainder = train_test_split(
            indices,
            train_size=spec.train_ratio,
            random_state=spec.seed,
            shuffle=True,
            stratify=stratify,
        )
        remainder_stratify = data.y[remainder] if stratify is not None else None
        val, test = _split_remaining(remainder, val_within_remainder, spec.seed + 1, remainder_stratify)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Data split produced an empty train, validation, or test set")
    return DataSplit(np.sort(train), np.sort(val), np.sort(test))


def split_group_cv(data: PreparedData, spec: SplitSpec) -> GroupCVSplit:
    """Reserve sealed test groups and create deterministic validation folds."""

    if spec.type != SplitType.GROUP_CV:
        raise ValueError("split_group_cv requires split.type='group_cv'")
    if data.groups is None:
        raise ValueError("group_cv split requires dataset.entity_column or native groups")
    indices = np.arange(len(data.x))
    group_values = data.groups.astype(str)
    unique_groups = np.unique(group_values)
    if spec.test_groups is not None:
        requested = np.asarray(spec.test_groups, dtype=str)
        unknown = sorted(set(requested) - set(unique_groups))
        if unknown:
            raise ValueError(f"Unknown test_groups: {unknown}; available={unique_groups.tolist()}")
        test_mask = np.isin(group_values, requested)
        test = indices[test_mask]
        development = indices[~test_mask]
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=spec.test_ratio, random_state=spec.seed)
        development_position, test_position = next(splitter.split(indices, groups=data.groups))
        development, test = indices[development_position], indices[test_position]
    development_groups = np.unique(group_values[development])
    if spec.cv_folds > len(development_groups):
        raise ValueError(
            f"cv_folds={spec.cv_folds} exceeds the {len(development_groups)} development groups remaining after test"
        )
    shuffled_groups = development_groups.copy()
    np.random.default_rng(spec.seed).shuffle(shuffled_groups)
    folds = []
    for fold_index in range(spec.cv_folds):
        val_groups = shuffled_groups[fold_index :: spec.cv_folds]
        val_mask = np.isin(group_values[development], val_groups)
        train = development[~val_mask]
        val = development[val_mask]
        if min(len(train), len(val)) == 0:
            raise ValueError(f"group_cv fold {fold_index} produced an empty train or validation set")
        folds.append(ValidationFold(np.sort(train), np.sort(val)))
    return GroupCVSplit(np.sort(development), np.sort(test), tuple(folds))


def split_rolling_origin(data: PreparedData, spec: SplitSpec) -> RollingOriginSplit:
    """Create expanding-window validation folds and reserve the latest windows for final testing."""

    if spec.type != SplitType.ROLLING_ORIGIN:
        raise ValueError("split_rolling_origin requires split.type='rolling_origin'")
    if data.timestamps is None:
        raise ValueError("rolling_origin split requires dataset.time_column")
    if data.groups is not None and len(np.unique(data.groups)) > 1:
        raise ValueError("rolling_origin v1 supports one time series; omit entity_column or select one entity")
    ordered = np.arange(len(data.x))[np.argsort(data.timestamps, kind="stable")]
    test_count = max(1, int(round(len(ordered) * spec.test_ratio)))
    development, test = ordered[:-test_count], ordered[-test_count:]
    if len(development) < spec.rolling_folds + 1:
        raise ValueError(
            f"rolling_folds={spec.rolling_folds} requires at least {spec.rolling_folds + 1} development samples"
        )
    blocks = np.array_split(development, spec.rolling_folds + 1)
    folds = []
    for fold_index in range(spec.rolling_folds):
        train = np.concatenate(blocks[: fold_index + 1])
        val = blocks[fold_index + 1]
        if min(len(train), len(val)) == 0:
            raise ValueError(f"rolling_origin fold {fold_index} produced an empty train or validation set")
        folds.append(ValidationFold(train, val))
    return RollingOriginSplit(development, test, tuple(folds))


def _prepare_esa_adb_dataset(spec: DatasetSpec) -> PreparedData:
    from torch_like.automl.esa_data import prepare_esa_adb_dataset

    return prepare_esa_adb_dataset(spec)


register_dataset_format(DatasetFormat.CSV, prepare_csv_dataset)
register_dataset_format(DatasetFormat.XJTU_SY, prepare_xjtu_sy_dataset)
register_dataset_format(DatasetFormat.ESA_ADB, _prepare_esa_adb_dataset)
