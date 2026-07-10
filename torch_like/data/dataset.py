"""Generic time series dataset for Torch DataLoader.

Supports common time series benchmark formats (ETT, Electricity, etc.)
with sliding window generation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class Dataset_ETT_hour(Dataset):
    """Hourly ETT (Electricity Transformer Temperature) dataset.

    Standard sliding window: each sample = (seq_len + label_len + pred_len) steps.
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        data_path="ETTh1.csv",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="h",
        cols=None,
    ):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported ETT split: {flag!r}")
        if size is None:
            size = [96, 48, 24]  # seq_len, label_len, pred_len
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.cols = cols
        self.__read_data__()

    def __read_data__(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        path = Path(self.root_path) / self.data_path
        if not path.is_file():
            raise FileNotFoundError(f"ETT data file not found: {path}")
        df_raw = pd.read_csv(path)
        cols_data = df_raw.columns[1:] if not self.cols else self.cols
        df_data = df_raw[cols_data]

        # Train/val/test split (12 months each for hourly ETT)
        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[{"train": 0, "val": 1, "test": 2}[self.flag]]
        border2 = border2s[{"train": 0, "val": 1, "test": 2}[self.flag]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler = StandardScaler()
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if len(self.data_x) < self.seq_len + self.pred_len:
            raise ValueError(
                f"{self.flag} split requires at least {self.seq_len + self.pred_len} rows, got {len(self.data_x)}"
            )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.zeros((self.seq_len, 4))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data


class Dataset_Custom(Dataset):
    """Generic sliding window dataset from raw numpy arrays."""

    def __init__(self, data, seq_len, label_len, pred_len, time_features=None):
        if seq_len <= 0 or pred_len <= 0 or not 0 <= label_len <= seq_len:
            raise ValueError("seq_len/pred_len must be positive and label_len must be between 0 and seq_len")
        if len(data) < seq_len + pred_len:
            raise ValueError(f"data length must be at least {seq_len + pred_len}, got {len(data)}")
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.time_features = time_features

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        return self.data[s_begin:s_end], self.data[r_begin:r_end]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
