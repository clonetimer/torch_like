"""Process identity, explicit process-group ownership, and data sharding.

The session never launches processes itself: use ``torchrun`` or a caller-owned
process group. Only groups created by the session are destroyed on exit.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler, Sampler


@dataclass(frozen=True)
class DistributedContext:
    """Process identity for a potentially distributed training job.

    ``rank`` is the global rank, ``local_rank`` is the device-local process
    index, and ``world_size`` is the total number of processes.  A context with
    ``world_size == 1`` represents ordinary single-process training.
    """

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        if self.local_rank < 0:
            raise ValueError("local_rank must be non-negative")

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DistributedContext":
        if value is None:
            return cls()
        return cls(
            rank=int(value.get("rank", 0)),
            world_size=int(value.get("world_size", 1)),
            local_rank=int(value.get("local_rank", 0)),
        )

    @classmethod
    def from_environment(cls) -> "DistributedContext":
        """Read standard torchrun rank variables without initializing a process group."""

        return cls(
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        )

    def require_single_process(self, *, feature: str = "Trainer") -> None:
        """Fail explicitly when a single-process-only component receives DDP metadata."""

        if self.is_distributed:
            raise NotImplementedError(
                f"{feature} currently supports one process only; received rank={self.rank}, "
                f"world_size={self.world_size}. DistributedContext is metadata/merge-contract support, "
                "not a DDP/FSDP execution switch."
            )


@dataclass(frozen=True)
class DistributedSamplerConfig:
    """Configuration for PyTorch's standard :class:`DistributedSampler`."""

    shuffle: bool = True
    seed: int = 0
    drop_last: bool = False


def create_distributed_sampler(
    dataset: Dataset,
    context: DistributedContext,
    config: DistributedSamplerConfig | None = None,
) -> DistributedSampler:
    """Create a rank-aware PyTorch sampler without initializing DDP.

    Call ``sampler.set_epoch(epoch)`` before each epoch when ``shuffle=True``.
    The helper delegates partitioning to PyTorch instead of reimplementing its
    distributed sampling algorithm.
    """

    config = config or DistributedSamplerConfig()
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=config.shuffle,
        seed=config.seed,
        drop_last=config.drop_last,
    )


@dataclass(frozen=True)
class DDPConfig:
    """Conservative DDP options; single optimizer and equal training steps only."""

    find_unused_parameters: bool = False
    broadcast_buffers: bool = True


class DistributedEvalSampler(Sampler[int]):
    """Exact strided validation shards: no padding and no discarded examples.

    Unlike the training sampler, ranks may have unequal or zero batches. Use
    with Trainer validation, not the synchronized DDP training forward path.
    Every rank must use the same dataset in the same order.
    """

    def __init__(self, dataset: Dataset, context: DistributedContext):
        self.dataset = dataset
        self.context = context

    def __iter__(self):
        return iter(range(self.context.rank, len(self.dataset), self.context.world_size))

    def __len__(self) -> int:
        return len(range(self.context.rank, len(self.dataset), self.context.world_size))


def _require_process_group(context: DistributedContext) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "DDP requires an initialized process group; use DistributedSession under torchrun "
            "or initialize torch.distributed explicitly"
        )
    if (dist.get_rank(), dist.get_world_size()) != (context.rank, context.world_size):
        raise ValueError("DistributedContext rank/world_size does not match the active process group")


def _distributed_device(requested: str, context: DistributedContext) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("DDP execution supports CPU/Gloo or CUDA/NCCL only")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.index is not None and device.index != context.local_rank:
            raise ValueError("DDP CUDA device index must match LOCAL_RANK")
        if context.local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices")
        device = torch.device("cuda", context.local_rank)
        torch.cuda.set_device(device)
    return device


class DistributedSession:
    """Own or borrow a default process group, with deterministic cleanup.

    A single-process session is a no-op. Multi-process sessions select Gloo for
    CPU and NCCL for CUDA. On errors no final barrier is attempted: a dead peer
    must not make cleanup wait forever. Peer process termination and failures
    inside native collectives are handled by the launcher and group timeout.
    """

    def __init__(
        self,
        context: DistributedContext | None = None,
        *,
        device: str = "auto",
        backend: str | None = None,
        init_method: str = "env://",
        timeout_seconds: float = 120.0,
    ) -> None:
        import math

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.context = context or DistributedContext.from_environment()
        self.device = _distributed_device(device, self.context)
        expected_backend = "nccl" if self.device.type == "cuda" else "gloo"
        self.backend = backend or expected_backend
        if self.backend != expected_backend:
            raise ValueError("Use CPU/Gloo or CUDA/NCCL; device and backend must agree")
        self.init_method = init_method
        self.timeout_seconds = timeout_seconds
        self._owns_group = False
        self._entered = False

    @property
    def owns_process_group(self) -> bool:
        return self._owns_group

    def __enter__(self) -> "DistributedSession":
        if self._entered:
            raise RuntimeError("DistributedSession cannot be entered twice concurrently")
        if self.context.is_distributed:
            if not dist.is_available():
                raise RuntimeError("This PyTorch build has no distributed support")
            if dist.is_initialized():
                _require_process_group(self.context)
                if dist.get_backend() != self.backend:
                    raise ValueError("Existing process group backend differs from requested backend")
            else:
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    rank=self.context.rank,
                    world_size=self.context.world_size,
                    timeout=timedelta(seconds=self.timeout_seconds),
                )
                self._owns_group = True
        elif dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Single-process context cannot borrow a multi-process group")
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._owns_group and dist.is_initialized():
                dist.destroy_process_group()
        finally:
            self._owns_group = False
            self._entered = False


__all__ = [
    "DDPConfig", "DistributedContext", "DistributedEvalSampler", "DistributedSamplerConfig",
    "DistributedSession", "create_distributed_sampler",
]
