"""Datasets for anomaly detection with sliding windows."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowAnomalyDataset(Dataset):
    """Create fixed-length windows for sequence reconstruction/anomaly scoring.

    Each sample is one window `x[index:index+seq_len]`. Optional labels are
    aligned to the last timestamp of the window for evaluation.
    """

    def __init__(self, data, seq_len: int, stride: int = 1, labels: Optional[np.ndarray] = None):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("data must have shape (T, C)")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if len(data) < seq_len:
            raise ValueError(f"data length must be at least seq_len={seq_len}, got {len(data)}")
        if labels is not None and len(labels) != len(data):
            raise ValueError(f"Expected {len(data)} labels, got {len(labels)}")
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        self.labels = None if labels is None else np.asarray(labels)
        self._starts = list(range(0, len(data) - seq_len + 1, stride))

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, index):
        start = self._starts[index]
        end = start + self.seq_len
        window = torch.tensor(self.data[start:end], dtype=torch.float32)
        if self.labels is None:
            return window
        label = self.labels[end - 1]
        return window, torch.tensor(label)


def create_train_test_anomaly_datasets(
    train_data,
    test_data,
    seq_len: int,
    stride: int = 1,
    test_labels: Optional[np.ndarray] = None,
):
    train_set = SlidingWindowAnomalyDataset(train_data, seq_len=seq_len, stride=stride)
    test_set = SlidingWindowAnomalyDataset(test_data, seq_len=seq_len, stride=stride, labels=test_labels)
    return train_set, test_set
