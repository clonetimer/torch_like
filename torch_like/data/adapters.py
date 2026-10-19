"""Reusable adapters from in-memory arrays/tensors to deterministic DataLoaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset


@runtime_checkable
class DataAdapter(Protocol):
    """Protocol for materializing framework DataLoaders from task data."""

    def loader(self, features: Any, targets: Any = None, **kwargs: Any) -> DataLoader: ...


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 32
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


class TensorDataAdapter:
    """Create TensorDataset/DataLoader objects without changing tensor dtype."""

    @staticmethod
    def _tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return torch.as_tensor(value)

    @staticmethod
    def _select(value: torch.Tensor, indices: Any | None) -> torch.Tensor:
        if indices is None:
            return value
        if isinstance(indices, torch.Tensor):
            index = indices.to(dtype=torch.long, device=value.device)
        else:
            index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index)

    def dataset(self, features: Any, targets: Any = None, *, indices: Any | None = None) -> TensorDataset:
        x = self._select(self._tensor(features), indices)
        tensors = [x]
        if targets is not None:
            y = self._select(self._tensor(targets), indices)
            if len(y) != len(x):
                raise ValueError(f"features and targets must contain the same samples: {len(x)} != {len(y)}")
            tensors.append(y)
        return TensorDataset(*tensors)

    def loader_from_dataset(
        self,
        dataset: Dataset,
        *,
        config: DataLoaderConfig | None = None,
        sampler: Sampler | None = None,
        **overrides: Any,
    ) -> DataLoader:
        """Create a DataLoader from an already-materialized dataset.

        Supplying ``sampler`` keeps sharding policy outside the adapter and is
        compatible with PyTorch ``DistributedSampler``. ``shuffle`` must be
        disabled when an explicit sampler is provided.
        """

        if config is not None and overrides:
            raise ValueError("pass either config or loader keyword overrides, not both")
        config = config or DataLoaderConfig(**overrides)
        if sampler is not None and config.shuffle:
            raise ValueError("DataLoaderConfig.shuffle must be False when sampler is provided")
        generator = None
        if config.seed is not None:
            generator = torch.Generator().manual_seed(config.seed)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            generator=generator,
            persistent_workers=config.num_workers > 0,
        )

    def loader(
        self,
        features: Any,
        targets: Any = None,
        *,
        indices: Any | None = None,
        config: DataLoaderConfig | None = None,
        **overrides: Any,
    ) -> DataLoader:
        dataset = self.dataset(features, targets, indices=indices)
        return self.loader_from_dataset(dataset, config=config, **overrides)


__all__ = ["DataAdapter", "DataLoaderConfig", "TensorDataAdapter"]
