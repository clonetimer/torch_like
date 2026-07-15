"""Native ESA-ADB archive ingestion for anomaly leaderboards."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from torch_like.automl.data import DatasetProfile, PreparedData
from torch_like.automl.specs import DatasetSpec

_METADATA_FILES = ("channels.csv", "labels.csv", "anomaly_types.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(cache_root: Path, member: str) -> Path:
    target = (cache_root / member).resolve()
    if cache_root.resolve() not in target.parents:
        raise ValueError(f"Unsafe ESA-ADB archive member: {member}")
    return target


def _materialize_archive(spec: DatasetSpec, archive: Path) -> Path:
    cache_base = Path(spec.cache_dir or "runs/data_cache/esa-adb")
    cache_root = cache_base / f"{archive.stem}-{archive.stat().st_size}"
    with zipfile.ZipFile(archive) as outer:
        names = set(outer.namelist())
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(roots) != 1:
            raise ValueError(f"ESA-ADB archive must contain exactly one mission folder, got {sorted(roots)}")
        mission_name = next(iter(roots))
        members = [f"{mission_name}/{name}" for name in _METADATA_FILES]
        members.extend(f"{mission_name}/channels/{channel}.zip" for channel in spec.feature_columns or ())
        missing = [member for member in members if member not in names]
        if missing:
            raise FileNotFoundError(f"ESA-ADB archive is missing required members: {missing}")
        for member in members:
            target = _safe_member_path(cache_root, member)
            info = outer.getinfo(member)
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with outer.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(target)
    return cache_root / mission_name


def _resolve_mission_root(spec: DatasetSpec) -> tuple[Path, str]:
    source = Path(spec.path)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError("ESA-ADB archive path must use .zip")
        return _materialize_archive(spec, source), _sha256(source)
    if not source.is_dir():
        raise FileNotFoundError(f"ESA-ADB path does not exist: {source}")
    if (source / "labels.csv").is_file():
        mission_root = source
    else:
        missions = [path for path in source.iterdir() if path.is_dir() and (path / "labels.csv").is_file()]
        if len(missions) != 1:
            raise ValueError(f"Expected one extracted ESA-ADB mission below {source}, got {len(missions)}")
        mission_root = missions[0]
    digest = hashlib.sha256()
    for name in _METADATA_FILES:
        digest.update(_sha256(mission_root / name).encode("ascii"))
    for channel in spec.feature_columns or ():
        digest.update(_sha256(mission_root / "channels" / f"{channel}.zip").encode("ascii"))
    return mission_root, digest.hexdigest()


def _timestamp(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize(None) if result.tzinfo is not None else result


def _split_ranges(spec: DatasetSpec) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    assert spec.split_time_ranges is not None
    ranges = {name: (_timestamp(bounds[0]), _timestamp(bounds[1])) for name, bounds in spec.split_time_ranges.items()}
    for name, (start, end) in ranges.items():
        if start >= end:
            raise ValueError(f"ESA-ADB {name} range must have start < end: {start} >= {end}")
    ordered = [ranges[name] for name in ("train", "val", "test")]
    if not (ordered[0][1] < ordered[1][0] and ordered[1][1] < ordered[2][0]):
        raise ValueError("ESA-ADB train/val/test ranges must be chronological and non-overlapping")
    return ranges


def _channel_labels(
    index: pd.DatetimeIndex,
    channel: str,
    labels: pd.DataFrame,
    categories: set[str],
    frequency: pd.Timedelta,
) -> np.ndarray:
    result = np.zeros(len(index), dtype=np.uint8)
    selected = labels[(labels["Channel"] == channel) & labels["Category"].isin(categories)]
    for row in selected.itertuples(index=False):
        # Preserve events shorter than the sampling period by marking every
        # time bucket that overlaps the original event interval.
        start = index.searchsorted(row.StartTime - frequency, side="right")
        end = index.searchsorted(row.EndTime, side="right")
        result[start:end] = 1
    return result


def _windows(
    values: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    sequence_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, len(values) - sequence_length + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(f"ESA-ADB split with {len(values)} rows is shorter than sequence_length={sequence_length}")
    x = np.stack([values[start : start + sequence_length] for start in starts]).astype(np.float32, copy=False)
    ends = starts + sequence_length - 1
    interval_starts = np.concatenate(([starts[0]], ends[:-1] + 1))
    cumulative = np.vstack((np.zeros((1, labels.shape[1]), dtype=np.int64), np.cumsum(labels, axis=0)))
    window_labels = (cumulative[ends + 1] - cumulative[interval_starts] > 0).astype(np.uint8)
    return x, window_labels, timestamps[ends]


def prepare_esa_adb_dataset(spec: DatasetSpec) -> PreparedData:
    """Load selected channels from a raw ESA mission ZIP without expanding the full archive."""

    mission_root, source_fingerprint = _resolve_mission_root(spec)
    channels = tuple(spec.feature_columns or ())
    channel_meta = pd.read_csv(mission_root / "channels.csv")
    known_channels = set(channel_meta["Channel"].astype(str))
    unknown = sorted(set(channels) - known_channels)
    if unknown:
        raise ValueError(f"Unknown ESA-ADB channels: {unknown}")
    non_targets = channel_meta.loc[
        channel_meta["Channel"].isin(channels) & (channel_meta["Target"].str.upper() != "YES"), "Channel"
    ].tolist()
    if non_targets:
        raise ValueError(f"Channel-aware ESA-ADB evaluation requires target channels, got {non_targets}")

    anomaly_types = pd.read_csv(mission_root / "anomaly_types.csv")
    labels = pd.read_csv(mission_root / "labels.csv", parse_dates=["StartTime", "EndTime"])
    labels["StartTime"] = labels["StartTime"].dt.tz_localize(None)
    labels["EndTime"] = labels["EndTime"].dt.tz_localize(None)
    labels = labels.merge(anomaly_types[["ID", "Category"]], on="ID", how="left", validate="many_to_one")
    categories = set(spec.anomaly_categories)
    unknown_categories = sorted(categories - set(anomaly_types["Category"].dropna().astype(str)))
    if unknown_categories:
        raise ValueError(f"Unknown ESA-ADB anomaly_categories: {unknown_categories}")

    ranges = _split_ranges(spec)
    frequency = pd.Timedelta(seconds=spec.resample_seconds)
    split_indices = {
        name: pd.date_range(start.floor(frequency), end.ceil(frequency), freq=frequency)
        for name, (start, end) in ranges.items()
    }
    split_values = {
        name: np.empty((len(index), len(channels)), dtype=np.float32) for name, index in split_indices.items()
    }
    split_labels = {
        name: np.empty((len(index), len(channels)), dtype=np.uint8) for name, index in split_indices.items()
    }
    missing_values = 0
    for channel_index, channel in enumerate(channels):
        frame = pd.read_pickle(mission_root / "channels" / f"{channel}.zip", compression="zip")
        series = frame.iloc[:, 0].astype(np.float32)
        channel_number = int(channel.rsplit("_", 1)[1])
        if 4 <= channel_number <= 11:
            series = pd.Series(np.diff(series.to_numpy(), append=series.iloc[-1]), index=series.index)
        for split_name, index in split_indices.items():
            resampled = series.reindex(index, method="ffill")
            if resampled.isna().any():
                resampled = resampled.bfill()
            missing_values += int(resampled.isna().sum())
            if resampled.isna().any():
                raise ValueError(f"Unable to fill ESA-ADB values for {channel} in split {split_name}")
            split_values[split_name][:, channel_index] = resampled.to_numpy(dtype=np.float32)
            split_labels[split_name][:, channel_index] = _channel_labels(index, channel, labels, categories, frequency)

    sequence_length = int(spec.sequence_length or 0)
    stride = int(spec.stride or 0)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    for split_name in ("train", "val", "test"):
        x_part, y_part, timestamp_part = _windows(
            split_values[split_name],
            split_labels[split_name],
            split_indices[split_name].to_numpy(),
            sequence_length,
            stride,
        )
        x_parts.append(x_part)
        y_parts.append(y_part)
        timestamp_parts.append(timestamp_part)
        split_parts.append(np.full(len(x_part), split_name, dtype=object))

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    timestamps = np.concatenate(timestamp_parts)
    predefined_splits = np.concatenate(split_parts)
    sample_ids = np.asarray([str(value) for value in timestamps], dtype=object)
    fingerprint_payload = (
        source_fingerprint,
        channels,
        spec.resample_seconds,
        spec.split_time_ranges,
        spec.anomaly_categories,
        sequence_length,
        stride,
    )
    fingerprint = hashlib.sha256(repr(fingerprint_payload).encode("utf-8")).hexdigest()
    profile = DatasetProfile(
        rows=sum(len(index) for index in split_indices.values()),
        samples=len(x),
        features=len(channels),
        labels=len(channels),
        entities=1,
        missing_values=missing_values,
        duplicate_rows=0,
        sample_mode=spec.sample_mode.value,
        sequence_min_length=sequence_length,
        sequence_max_length=sequence_length,
        fingerprint=fingerprint,
        feature_columns=channels,
        label_columns=channels,
        warnings=(
            "ESA-ADB values are zero-order-hold resampled; metrics use the selected uniform time grid.",
            "Selected time ranges are a resource-bounded subset and are not the full official ESA-ADB benchmark.",
        ),
    )
    return PreparedData(
        x=x,
        y=y,
        sample_ids=sample_ids,
        groups=np.full(len(x), mission_root.name, dtype=object),
        timestamps=timestamps,
        predefined_splits=predefined_splits,
        feature_columns=channels,
        is_sequence=True,
        sequence_lengths=(sequence_length,) * len(x),
        profile=profile,
    )


__all__ = ["prepare_esa_adb_dataset"]
