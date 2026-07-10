"""Task-oriented data adapters shared by the framework and runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch_like.data.dataset import Dataset_Custom, Dataset_ETT_hour
from torch_like.data.dataset_detection import create_train_test_anomaly_datasets
from torch_like.data.dataset_graph_forecasting import GraphForecastDataset, load_graph_forecast_array
from torch_like.data.dataset_rul import create_train_test_split


def _loader_kwargs(num_workers: int = 0, device: str | None = None):
    use_cuda = device is not None and device.startswith("cuda")
    kwargs = {"num_workers": num_workers}
    if use_cuda:
        kwargs["pin_memory"] = True
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


@dataclass
class ForecastDataConfig:
    dataset_type: str = "ett_hour"
    root_path: str | None = None
    data_path: str | None = None
    adjacency_path: str | None = None
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    batch_size: int = 32
    num_workers: int = 0
    freq: str = "h"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    numpy_data: np.ndarray | None = None
    adjacency: np.ndarray | None = None
    target_indices: list[int] | None = None
    exogenous_indices: list[int] | None = None

    def __post_init__(self):
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if not 0 <= self.label_len <= self.seq_len:
            raise ValueError("label_len must be between 0 and seq_len")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers must be non-negative")
        if not 0 < self.train_ratio < 1 or not 0 < self.val_ratio < 1:
            raise ValueError("train_ratio and val_ratio must be in (0, 1)")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")


@dataclass
class ForecastDataBundle:
    train_set: object
    val_set: object
    test_set: object
    extra: dict


class _ForecastDatasetWrapper(torch.utils.data.Dataset):
    """Normalize forecasting datasets to a dict-based shared batch contract."""

    def __init__(
        self,
        base_dataset,
        seq_len: int,
        label_len: int,
        pred_len: int,
        target_indices: list[int] | None = None,
        exogenous_indices: list[int] | None = None,
        flatten_feature_axes: bool = False,
    ):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_indices = target_indices
        self.exogenous_indices = exogenous_indices
        self.flatten_feature_axes = flatten_feature_axes

    def _normalize_shape(self, seq: torch.Tensor) -> torch.Tensor:
        if self.flatten_feature_axes and seq.dim() > 2:
            return seq.reshape(seq.size(0), -1)
        return seq

    def _split_features(self, seq: torch.Tensor):
        if self.target_indices is None:
            empty = seq[..., :0]
            return empty, empty
        target = seq[..., self.target_indices]
        exog = seq[..., :0]
        if self.exogenous_indices:
            exog = seq[..., self.exogenous_indices]
        return target, exog

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        seq_x, seq_y = self.base_dataset[index]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.zeros((self.seq_len, 4), dtype=torch.float32)
        seq_y_mark = torch.zeros((self.label_len + self.pred_len, 4), dtype=torch.float32)
        x_target_enc, x_exog_enc = self._split_features(seq_x)
        y_target_full, y_exog_full = self._split_features(seq_y)
        seq_x = self._normalize_shape(seq_x)
        seq_y = self._normalize_shape(seq_y)
        x_target_enc = self._normalize_shape(x_target_enc)
        x_exog_enc = self._normalize_shape(x_exog_enc)
        y_target_full = self._normalize_shape(y_target_full)
        y_exog_full = self._normalize_shape(y_exog_full)
        x_target_dec = y_target_full
        x_exog_dec = y_exog_full
        x_dec_target = seq_y if self.target_indices is None else y_target_full
        return {
            "x_enc": seq_x,
            "x_dec_target": x_dec_target,
            "x_mark_enc": seq_x_mark,
            "x_mark_dec": seq_y_mark,
            "x_target_enc": x_target_enc,
            "x_exog_enc": x_exog_enc,
            "x_target_dec": x_target_dec,
            "x_exog_dec": x_exog_dec,
        }


def _resolve_feature_splits(config: ForecastDataConfig, feature_dim: int):
    target_indices = config.target_indices
    exogenous_indices = config.exogenous_indices
    if target_indices is None and exogenous_indices is None:
        return None, None
    if target_indices is None:
        target_indices = [i for i in range(feature_dim) if i not in set(exogenous_indices or [])]
    if exogenous_indices is None:
        exogenous_indices = [i for i in range(feature_dim) if i not in set(target_indices or [])]
    target_indices = list(target_indices)
    exogenous_indices = list(exogenous_indices)
    combined = target_indices + exogenous_indices
    if not target_indices:
        raise ValueError("target_indices must contain at least one feature")
    if len(set(target_indices)) != len(target_indices) or len(set(exogenous_indices)) != len(exogenous_indices):
        raise ValueError("target_indices and exogenous_indices must not contain duplicates")
    invalid = [index for index in combined if not 0 <= index < feature_dim]
    if invalid:
        raise ValueError(f"Feature indices out of range for feature_dim={feature_dim}: {invalid}")
    overlap = sorted(set(target_indices) & set(exogenous_indices))
    if overlap:
        raise ValueError(f"Target and exogenous feature indices overlap: {overlap}")
    return target_indices, exogenous_indices


def _split_boundaries(n: int, config: ForecastDataConfig) -> tuple[int, int]:
    min_required = config.seq_len + config.pred_len
    if n < 3 * min_required:
        raise ValueError(
            f"Forecast data requires at least {3 * min_required} rows for non-empty train/val/test splits, got {n}"
        )
    train_end = min(max(int(n * config.train_ratio), min_required), n - 2 * min_required)
    val_end = min(max(int(n * (config.train_ratio + config.val_ratio)), train_end + min_required), n - min_required)
    return train_end, val_end


def _make_ett_splits(config: ForecastDataConfig):
    if config.root_path is None or config.data_path is None:
        raise ValueError("root_path and data_path are required for dataset_type='ett_hour'")
    train_set = Dataset_ETT_hour(
        root_path=config.root_path,
        data_path=config.data_path,
        flag="train",
        size=[config.seq_len, config.label_len, config.pred_len],
        scale=True,
        freq=config.freq,
    )
    val_set = Dataset_ETT_hour(
        root_path=config.root_path,
        data_path=config.data_path,
        flag="val",
        size=[config.seq_len, config.label_len, config.pred_len],
        scale=True,
        freq=config.freq,
    )
    test_set = Dataset_ETT_hour(
        root_path=config.root_path,
        data_path=config.data_path,
        flag="test",
        size=[config.seq_len, config.label_len, config.pred_len],
        scale=True,
        freq=config.freq,
    )
    return train_set, val_set, test_set


def _make_numpy_splits(config: ForecastDataConfig):
    if config.numpy_data is None:
        raise ValueError("numpy_data must be provided for dataset_type='numpy'")
    data = np.asarray(config.numpy_data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("numpy forecasting data must have shape (time, features) with at least one feature")
    if not np.isfinite(data).all():
        raise ValueError("numpy forecasting data contains NaN or infinite values")
    n = len(data)
    train_end, val_end = _split_boundaries(n, config)
    feature_dim = int(data.shape[-1])
    target_indices, exogenous_indices = _resolve_feature_splits(config, feature_dim)
    train_set = _ForecastDatasetWrapper(
        Dataset_Custom(data[:train_end], config.seq_len, config.label_len, config.pred_len),
        config.seq_len,
        config.label_len,
        config.pred_len,
        target_indices=target_indices,
        exogenous_indices=exogenous_indices,
    )
    val_set = _ForecastDatasetWrapper(
        Dataset_Custom(data[train_end:val_end], config.seq_len, config.label_len, config.pred_len),
        config.seq_len,
        config.label_len,
        config.pred_len,
        target_indices=target_indices,
        exogenous_indices=exogenous_indices,
    )
    test_set = _ForecastDatasetWrapper(
        Dataset_Custom(data[val_end:], config.seq_len, config.label_len, config.pred_len),
        config.seq_len,
        config.label_len,
        config.pred_len,
        target_indices=target_indices,
        exogenous_indices=exogenous_indices,
    )
    extra = {
        "target_indices": target_indices,
        "exogenous_indices": exogenous_indices,
    }
    return train_set, val_set, test_set, extra


def _make_graph_splits(config: ForecastDataConfig):
    data = load_graph_forecast_array(numpy_data=config.numpy_data, data_path=config.data_path)
    if not np.isfinite(data).all():
        raise ValueError("graph forecasting data contains NaN or infinite values")
    n = len(data)
    train_end, val_end = _split_boundaries(n, config)
    split_feature_dim = int(data.shape[1] if data.ndim == 2 else data.shape[2])
    flatten_feature_axes = data.ndim == 3
    target_indices, exogenous_indices = _resolve_feature_splits(config, split_feature_dim)
    train_set = _ForecastDatasetWrapper(
        GraphForecastDataset(data[:train_end], config.seq_len, config.label_len, config.pred_len),
        config.seq_len,
        config.label_len,
        config.pred_len,
        target_indices=target_indices,
        exogenous_indices=exogenous_indices,
        flatten_feature_axes=flatten_feature_axes,
    )
    val_set = _ForecastDatasetWrapper(
        GraphForecastDataset(data[train_end:val_end], config.seq_len, config.label_len, config.pred_len),
        config.seq_len,
        config.label_len,
        config.pred_len,
        target_indices=target_indices,
        exogenous_indices=exogenous_indices,
        flatten_feature_axes=flatten_feature_axes,
    )
    test_set = _ForecastDatasetWrapper(
        GraphForecastDataset(data[val_end:], config.seq_len, config.label_len, config.pred_len),
        config.seq_len,
        config.label_len,
        config.pred_len,
        target_indices=target_indices,
        exogenous_indices=exogenous_indices,
        flatten_feature_axes=flatten_feature_axes,
    )
    adjacency = config.adjacency
    if adjacency is None and config.adjacency_path:
        adjacency = np.load(config.adjacency_path).astype(np.float32)
    if adjacency is not None:
        adjacency = np.asarray(adjacency, dtype=np.float32)
        expected = (int(data.shape[1]), int(data.shape[1]))
        if adjacency.shape != expected:
            raise ValueError(f"adjacency must have shape {expected}, got {tuple(adjacency.shape)}")
        if not np.isfinite(adjacency).all():
            raise ValueError("adjacency contains NaN or infinite values")
    node_adjacency = adjacency
    if adjacency is not None and flatten_feature_axes:
        adjacency = np.kron(adjacency, np.ones((split_feature_dim, split_feature_dim), dtype=np.float32))
    model_num_nodes = int(data.shape[1] * (split_feature_dim if flatten_feature_axes else 1))
    extra = {
        "dataset_type": config.dataset_type,
        "num_nodes": int(data.shape[1]),
        "feature_dim": int(1 if data.ndim == 2 else data.shape[2]),
        "model_num_nodes": model_num_nodes,
        "target_indices": target_indices,
        "exogenous_indices": exogenous_indices,
        "adjacency": adjacency.tolist() if adjacency is not None else None,
        "node_adjacency": node_adjacency.tolist() if node_adjacency is not None else None,
    }
    return train_set, val_set, test_set, extra


def load_forecast_data(config: ForecastDataConfig) -> ForecastDataBundle:
    if config.dataset_type == "ett_hour":
        train_set, val_set, test_set = _make_ett_splits(config)
        extra = {
            "dataset_type": config.dataset_type,
            "data_path": config.data_path,
            "target_indices": config.target_indices,
            "exogenous_indices": config.exogenous_indices,
        }
    elif config.dataset_type == "numpy":
        train_set, val_set, test_set, split_extra = _make_numpy_splits(config)
        extra = {"dataset_type": config.dataset_type, **split_extra}
    elif config.dataset_type in {"graph_npy", "graph_csv"}:
        train_set, val_set, test_set, extra = _make_graph_splits(config)
    else:
        raise ValueError(f"Unsupported forecast dataset_type: {config.dataset_type}")
    return ForecastDataBundle(train_set=train_set, val_set=val_set, test_set=test_set, extra=extra)


def create_forecast_loaders(config: ForecastDataConfig, device: str | None = None):
    bundle = load_forecast_data(config)
    loader_kwargs = _loader_kwargs(config.num_workers, device=device)
    train_loader = DataLoader(
        bundle.train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=len(bundle.train_set) >= config.batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(bundle.val_set, batch_size=config.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(bundle.test_set, batch_size=config.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    return bundle, train_loader, val_loader, test_loader


@dataclass
class DetectionDataConfig:
    data_root: str | None = None
    train_path: str | None = None
    test_path: str | None = None
    seq_len: int = 10
    stride: int = 1
    batch_size: int = 128
    num_workers: int = 0
    max_train_rows: int | None = None
    max_test_rows: int | None = None

    def __post_init__(self):
        if self.seq_len < 2 or self.stride <= 0:
            raise ValueError("seq_len must be at least 2 and stride must be positive")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers must be non-negative")
        if self.max_train_rows is not None and self.max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive when provided")
        if self.max_test_rows is not None and self.max_test_rows <= 0:
            raise ValueError("max_test_rows must be positive when provided")


@dataclass
class DetectionDataBundle:
    train_data: np.ndarray
    test_data: np.ndarray
    test_labels: np.ndarray | None
    extra: dict


def load_numpy_detection_root(data_root: str):
    root = Path(data_root)
    train = np.load(root / "train.npy")
    test = np.load(root / "test.npy")
    labels_path = root / "test_labels.npy"
    labels = np.load(labels_path) if labels_path.exists() else None
    return train, test, labels


def load_esa_csv(path: str, max_rows: int | None = None):
    df = pd.read_csv(path, nrows=max_rows)
    value_cols = [c for c in df.columns if c != "timestamp" and not c.startswith("is_anomaly_")]
    label_cols = [c for c in df.columns if c.startswith("is_anomaly_")]
    if not value_cols:
        raise ValueError(f"No value columns found in detection CSV: {path}")
    data = df[value_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Detection CSV contains NaN or infinite values: {path}")
    labels = None
    if label_cols:
        label_matrix = df[label_cols].to_numpy(dtype=np.float32)
        labels = (label_matrix.max(axis=1) > 0).astype(np.int64)
    return data, labels, value_cols, label_cols


def load_detection_data(config: DetectionDataConfig) -> DetectionDataBundle:
    if config.train_path and config.test_path:
        train, train_labels, value_cols, _ = load_esa_csv(config.train_path, max_rows=config.max_train_rows)
        test, test_labels, test_value_cols, _ = load_esa_csv(config.test_path, max_rows=config.max_test_rows)
        if value_cols != test_value_cols:
            raise ValueError(f"Train/test value columns differ: {value_cols} != {test_value_cols}")
        if train.shape[1] != test.shape[1]:
            raise ValueError(f"Train/test feature dimensions differ: {train.shape[1]} != {test.shape[1]}")
        return DetectionDataBundle(
            train_data=train,
            test_data=test,
            test_labels=test_labels,
            extra={"columns": value_cols, "train_labels_present": train_labels is not None},
        )
    if config.data_root:
        train, test, labels = load_numpy_detection_root(config.data_root)
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
            raise ValueError("Detection train/test arrays must be 2D with matching feature dimensions")
        if not np.isfinite(train).all() or not np.isfinite(test).all():
            raise ValueError("Detection arrays contain NaN or infinite values")
        if labels is not None:
            if labels.ndim != 1:
                raise ValueError(f"Detection test labels must be one-dimensional, got shape {tuple(labels.shape)}")
            if len(labels) != len(test):
                raise ValueError(f"Expected {len(test)} test labels, got {len(labels)}")
        return DetectionDataBundle(train_data=train, test_data=test, test_labels=labels, extra={})
    raise ValueError("Either data_root or both train_path/test_path must be provided")


def create_detection_loaders(config: DetectionDataConfig, device: str | None = None):
    bundle = load_detection_data(config)
    train_set, test_set = create_train_test_anomaly_datasets(
        bundle.train_data,
        bundle.test_data,
        seq_len=config.seq_len,
        stride=config.stride,
        test_labels=bundle.test_labels,
    )
    loader_kwargs = _loader_kwargs(config.num_workers, device=device)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    return bundle, train_loader, test_loader


@dataclass
class RULDataConfig:
    root_dir: str
    condition: str = "35Hz12kN"
    seq_len: int = 2048
    stride: int = 1024
    train_ratio: float = 0.6
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0

    def __post_init__(self):
        if self.seq_len <= 0 or self.stride <= 0:
            raise ValueError("seq_len and stride must be positive")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be in (0, 1)")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers must be non-negative")


@dataclass
class RULDataBundle:
    train_set: object
    test_set: object
    extra: dict


def load_rul_data(config: RULDataConfig) -> RULDataBundle:
    train_set, test_set = create_train_test_split(
        config.root_dir,
        condition=config.condition,
        seq_len=config.seq_len,
        stride=config.stride,
        train_ratio=config.train_ratio,
        seed=config.seed,
    )
    return RULDataBundle(
        train_set=train_set,
        test_set=test_set,
        extra={"condition": config.condition, "seq_len": config.seq_len, "stride": config.stride},
    )


def create_rul_loaders(config: RULDataConfig, device: str | None = None):
    bundle = load_rul_data(config)
    loader_kwargs = _loader_kwargs(config.num_workers, device=device)
    train_loader = DataLoader(
        bundle.train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=len(bundle.train_set) >= config.batch_size,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        bundle.test_set,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return bundle, train_loader, test_loader


__all__ = [
    "ForecastDataConfig",
    "ForecastDataBundle",
    "DetectionDataConfig",
    "DetectionDataBundle",
    "RULDataConfig",
    "RULDataBundle",
    "load_forecast_data",
    "create_forecast_loaders",
    "load_detection_data",
    "create_detection_loaders",
    "load_rul_data",
    "create_rul_loaders",
    "load_numpy_detection_root",
    "load_esa_csv",
]
