"""Graph-style forecasting datasets built on top of sliding windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass
class GraphForecastMeta:
    num_nodes: int
    feature_dim: int
    adjacency: np.ndarray | None = None


class GraphForecastDataset(Dataset):
    """Sliding-window dataset for graph/multivariate time series.

    Expected input shape:
    - (time, nodes)
    - (time, nodes, features)
    """

    def __init__(self, data, seq_len: int, label_len: int, pred_len: int):
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError("GraphForecastDataset expects data with shape (time, nodes) or (time, nodes, features)")
        if seq_len <= 0 or pred_len <= 0 or not 0 <= label_len <= seq_len:
            raise ValueError("seq_len/pred_len must be positive and label_len must be between 0 and seq_len")
        if len(arr) < seq_len + pred_len:
            raise ValueError(f"data length must be at least {seq_len + pred_len}, got {len(arr)}")
        self.data = arr
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_nodes = arr.shape[1]
        self.feature_dim = arr.shape[2]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        if self.feature_dim == 1:
            seq_x = seq_x[..., 0]
            seq_y = seq_y[..., 0]
        return seq_x, seq_y


def load_graph_forecast_array(numpy_data=None, data_path: str | None = None):
    if numpy_data is not None:
        return np.asarray(numpy_data, dtype=np.float32)
    if data_path is None:
        raise ValueError("Either numpy_data or data_path must be provided for graph forecasting data")
    if data_path.lower().endswith(".npy"):
        return np.load(data_path).astype(np.float32)
    if data_path.lower().endswith(".csv"):
        return np.loadtxt(data_path, delimiter=",", dtype=np.float32)
    raise ValueError(f"Unsupported graph forecasting file: {data_path}")
