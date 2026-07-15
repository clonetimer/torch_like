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
from collections import OrderedDict
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
        self.points_per_file = []
        for bpath in self.bearing_paths:
            n_files = len([f for f in os.listdir(bpath) if f.endswith(".csv")])
            self.life_lengths.append(n_files)
            first_file = os.path.join(bpath, "1.csv")
            with open(first_file, encoding="utf-8") as handle:
                points = max(0, sum(1 for _ in handle) - 1)
            if points <= 0:
                raise ValueError(f"XJTU-SY acquisition file is empty: {first_file}")
            self.points_per_file.append(points)

        # Index mapping: global index → (bearing_idx, window_start_point)
        self.index_map = self._build_index()
        self._signal_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

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
            total_points = n_minutes * self.points_per_file[b_idx]
            for start in range(0, total_points - self.seq_len + 1, self.stride):
                index_map.append((b_idx, start))
        return index_map

    def __len__(self):
        return len(self.index_map)

    # ------------------------------------------------------------------
    #  Data access
    # ------------------------------------------------------------------

    def _read_signal_file(self, bearing_idx: int, file_index: int) -> np.ndarray:
        key = (bearing_idx, file_index)
        if key in self._signal_cache:
            self._signal_cache.move_to_end(key)
            return self._signal_cache[key]
        path = os.path.join(self.bearing_paths[bearing_idx], f"{file_index + 1}.csv")
        signal = pd.read_csv(path).to_numpy(dtype=np.float32)
        self._signal_cache[key] = signal
        if len(self._signal_cache) > 8:
            self._signal_cache.popitem(last=False)
        return signal

    def _read_window(self, bearing_idx: int, start: int) -> np.ndarray:
        """Read only the CSV chunks touched by one continuous window."""

        pieces = []
        remaining = self.seq_len
        position = start
        while remaining > 0:
            file_index, offset = divmod(position, self.points_per_file[bearing_idx])
            signal = self._read_signal_file(bearing_idx, file_index)
            piece = signal[offset : offset + remaining]
            if len(piece) == 0:
                raise IndexError(f"Unable to read RUL window at point {position}")
            pieces.append(piece)
            remaining -= len(piece)
            position += len(piece)
        return np.concatenate(pieces, axis=0)

    def __getitem__(self, idx):
        b_idx, start = self.index_map[idx]
        window = self._read_window(b_idx, start)

        # RUL label: normalized remaining life
        n_minutes = self.life_lengths[b_idx]
        # The end of the window determines the current time
        end_point = start + self.seq_len
        end_minute = end_point / float(self.points_per_file[b_idx])
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


class XJTUSYFileWindowDataset(XJTUSYDataset):
    """Bounded XJTU-SY view where each sample stays inside one acquisition file.

    This view samples the degradation trajectory without constructing every
    overlapping raw window or concatenating a complete bearing life cycle.
    """

    def __init__(
        self,
        root_dir: str,
        condition: Optional[str] = None,
        seq_len: int = 2048,
        stride: Optional[int] = None,
        bearing_ids: Optional[List[str]] = None,
        max_files_per_bearing: Optional[int] = None,
        max_windows_per_file: Optional[int] = 1,
    ):
        self.max_files_per_bearing = max_files_per_bearing
        self.max_windows_per_file = max_windows_per_file
        if max_files_per_bearing is not None and max_files_per_bearing <= 0:
            raise ValueError("max_files_per_bearing must be positive")
        if max_windows_per_file is not None and max_windows_per_file <= 0:
            raise ValueError("max_windows_per_file must be positive")
        super().__init__(root_dir, condition, seq_len, stride, bearing_ids)

    @staticmethod
    def _even_positions(count: int, limit: Optional[int]) -> List[int]:
        if limit is None or limit >= count:
            return list(range(count))
        if limit == 1:
            return [count // 2]
        return sorted(set(np.linspace(0, count - 1, num=limit, dtype=int).tolist()))

    def _build_index(self):
        index_map = []
        for bearing_idx, n_files in enumerate(self.life_lengths):
            file_indices = self._even_positions(n_files, self.max_files_per_bearing)
            points = self.points_per_file[bearing_idx]
            possible_starts = list(range(0, points - self.seq_len + 1, self.stride))
            if not possible_starts:
                raise ValueError(f"seq_len cannot exceed the {points} rows in one XJTU-SY acquisition file")
            starts = [
                possible_starts[index]
                for index in self._even_positions(len(possible_starts), self.max_windows_per_file)
            ]
            for file_index in file_indices:
                for start in starts:
                    index_map.append((bearing_idx, file_index, start))
        return index_map

    def __getitem__(self, idx):
        bearing_idx, file_index, start = self.index_map[idx]
        signal = self._read_signal_file(bearing_idx, file_index)
        window = signal[start : start + self.seq_len]
        lifetime = self.life_lengths[bearing_idx]
        rul = max(0.0, float(lifetime - (file_index + 1)) / float(lifetime))
        return torch.from_numpy(window.copy()), torch.tensor(rul, dtype=torch.float32)


def create_train_test_split(
    root_dir: str,
    condition: str = "35Hz12kN",
    seq_len: int = 2048,
    stride: Optional[int] = None,
    train_ratio: float = 0.6,
    seed: int = 42,
    sample_unit: str = "continuous",
    max_files_per_bearing: Optional[int] = None,
    max_windows_per_file: Optional[int] = 1,
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

    if sample_unit not in {"continuous", "file_window"}:
        raise ValueError("sample_unit must be 'continuous' or 'file_window'")
    dataset_class = XJTUSYDataset if sample_unit == "continuous" else XJTUSYFileWindowDataset
    extra = {}
    if sample_unit == "file_window":
        extra = {
            "max_files_per_bearing": max_files_per_bearing,
            "max_windows_per_file": max_windows_per_file,
        }
    train_set = dataset_class(
        root_dir,
        condition=condition,
        seq_len=seq_len,
        stride=stride,
        bearing_ids=train_ids,
        **extra,
    )
    test_set = dataset_class(
        root_dir,
        condition=condition,
        seq_len=seq_len,
        stride=stride,
        bearing_ids=test_ids,
        **extra,
    )

    return train_set, test_set
