"""XJTU-SY bearing RUL prediction dataset.

Data format per bearing:
  - Each CSV = 1 minute of vibration (32,768 points × 2 channels).
  - Files named 1.csv ... N.csv (N = bearing lifetime in minutes).
  - RUL(t) = (N - t) / N  (normalized to [0, 1]).

Usage:
  dataset = XJTUSYDataset(root_dir, condition="35Hz12kN", seq_len=2048)
  x, y = dataset[0]  # x: (seq_len, 2), y: scalar RUL
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class XJTUSYDataset(Dataset):
    """XJTU-SY bearing RUL regression dataset with sliding windows.

    Each bearing's life-cycle is split into sliding windows. Each window
    is a contiguous segment of raw vibration signal; its label is the
    normalized RUL at the window's end time.

    RUL definition:  rul(t) = (max_life - t) / max_life,  where t is the
    file index (in minutes) and max_life is the total number of CSV files.

    Args:
        root_dir:       Path to XJTU-SY_Bearing_Datasets directory.
        condition:      Subset of operating conditions, e.g. "35Hz12kN".
                        None uses all conditions.
        seq_len:        Number of sampling points per window.
        stride:         Step between consecutive windows (in sampling points).
                        Defaults to seq_len // 2.
        bearing_ids:    List of bearing names to include.
                        None includes all bearings in the condition(s).
        train_bearings: Bearings used for training (overrides bearing-based split).
        test_bearings:  Bearings used for testing.
    """

    def __init__(
        self,
        root_dir: str,
        condition: Optional[str] = None,
        seq_len: int = 2048,
        stride: Optional[int] = None,
        bearing_ids: Optional[List[str]] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)

        # Discover bearings
        self.bearing_paths, self.bearing_names = self._discover_bearings(condition, bearing_ids)

        if len(self.bearing_paths) == 0:
            raise ValueError(f"No bearings found in {root_dir} for condition={condition}")

        # Pre-scan: read all CSV file counts (life length in minutes)
        self.life_lengths = []  # N of CSV files per bearing
        for bpath in self.bearing_paths:
            n_files = len([f for f in os.listdir(bpath) if f.endswith(".csv")])
            self.life_lengths.append(n_files)

        # Index mapping: global index → (bearing_idx, window_start_point)
        self.index_map = self._build_index()

    # ------------------------------------------------------------------
    #  Discovery
    # ------------------------------------------------------------------

    def _discover_bearings(self, condition, bearing_ids):
        pairs = []
        names = []
        for entry in sorted(os.listdir(self.root_dir)):
            entry_path = os.path.join(self.root_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            # Condition filter
            if condition is not None and entry != condition:
                continue
            for sub in sorted(os.listdir(entry_path)):
                sub_path = os.path.join(entry_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                if bearing_ids is not None and sub not in bearing_ids:
                    continue
                pairs.append(sub_path)
                names.append(sub)
        return pairs, names

    # ------------------------------------------------------------------
    #  Index
    # ------------------------------------------------------------------

    def _build_index(self):
        """Build a global sample index: (bearing_idx, start_point)."""
        index_map = []
        for b_idx, n_minutes in enumerate(self.life_lengths):
            total_points = n_minutes * 32768  # 32768 points per file
            for start in range(0, total_points - self.seq_len + 1, self.stride):
                index_map.append((b_idx, start))
        return index_map

    def __len__(self):
        return len(self.index_map)

    # ------------------------------------------------------------------
    #  Data access
    # ------------------------------------------------------------------

    def _read_bearing_signal(self, bearing_idx: int) -> np.ndarray:
        """Full concatenated signal for a bearing, cached."""
        if not hasattr(self, "_signal_cache"):
            self._signal_cache: dict = {}
        if bearing_idx not in self._signal_cache:
            bpath = self.bearing_paths[bearing_idx]
            n_files = self.life_lengths[bearing_idx]
            # Read and concatenate all CSVs
            chunks = []
            for i in range(1, n_files + 1):
                fp = os.path.join(bpath, f"{i}.csv")
                df = pd.read_csv(fp, header=0)  # 2 cols: Horizontal, Vertical
                chunks.append(df.values.astype(np.float32))
            self._signal_cache[bearing_idx] = np.concatenate(chunks, axis=0)  # (total_points, 2)
        return self._signal_cache[bearing_idx]

    def __getitem__(self, idx):
        b_idx, start = self.index_map[idx]
        signal = self._read_bearing_signal(b_idx)
        window = signal[start : start + self.seq_len]  # (seq_len, 2)

        # RUL label: normalized remaining life
        n_minutes = self.life_lengths[b_idx]
        # The end of the window determines the current time
        end_point = start + self.seq_len
        end_minute = end_point / 32768.0  # continuous minute index
        rul_minutes = n_minutes - end_minute
        rul_norm = float(rul_minutes / n_minutes)  # [0, 1]

        return torch.tensor(window, dtype=torch.float32), torch.tensor(rul_norm, dtype=torch.float32)

    # ------------------------------------------------------------------
    #  Convenience
    # ------------------------------------------------------------------

    def num_bearings(self) -> int:
        return len(self.bearing_paths)

    @property
    def num_channels(self) -> int:
        """Number of sensor channels (always 2 for XJTU-SY)."""
        return 2


def create_train_test_split(
    root_dir: str,
    condition: str = "35Hz12kN",
    seq_len: int = 2048,
    stride: Optional[int] = None,
    train_ratio: float = 0.6,
    seed: int = 42,
) -> Tuple[XJTUSYDataset, XJTUSYDataset]:
    """Create train/test datasets by splitting bearings.

    Args:
        root_dir:     Path to XJTU-SY_Bearing_Datasets.
        condition:    Operating condition string.
        seq_len:      Window length (sampling points).
        stride:       Stride between windows.
        train_ratio:  Fraction of bearings for training.
        seed:         Random seed for reproducibility.

    Returns:
        train_set, test_set
    """
    # Discover all bearings in this condition
    cond_path = os.path.join(root_dir, condition)
    all_bearings = sorted([d for d in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, d))])

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_bearings))
    n_train = max(1, int(len(all_bearings) * train_ratio))
    train_ids = [all_bearings[i] for i in indices[:n_train]]
    test_ids = [all_bearings[i] for i in indices[n_train:]]

    train_set = XJTUSYDataset(
        root_dir,
        condition=condition,
        seq_len=seq_len,
        stride=stride,
        bearing_ids=train_ids,
    )
    test_set = XJTUSYDataset(
        root_dir,
        condition=condition,
        seq_len=seq_len,
        stride=stride,
        bearing_ids=test_ids,
    )

    return train_set, test_set
