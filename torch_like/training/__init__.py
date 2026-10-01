"""Public training runtime for torch_like PyTorch models."""

from torch_like.training.adapters import (
    AdaptedBatch,
    BatchAdapter,
    CallableBatchAdapter,
    DefaultBatchAdapter,
    ForecastingBatchAdapter,
    IdentityBatchAdapter,
    resolve_batch_adapter,
)
from torch_like.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from torch_like.training.config import TrainerConfig
from torch_like.training.data import infer_batch_size, move_to_device
from torch_like.training.distributed import (
    DDPConfig,
    DistributedContext,
    DistributedEvalSampler,
    DistributedSession,
    DistributedSamplerConfig,
    create_distributed_sampler,
)
from torch_like.training.events import training_events
from torch_like.training.state import EpochResult, TrainingResult
from torch_like.training.trainer import Trainer, train_model

__all__ = [
    "AdaptedBatch",
    "BatchAdapter",
    "CallableBatchAdapter",
    "DefaultBatchAdapter",
    "DDPConfig",
    "DistributedContext",
    "DistributedEvalSampler",
    "DistributedSession",
    "DistributedSamplerConfig",
    "ForecastingBatchAdapter",
    "IdentityBatchAdapter",
    "EpochResult",
    "Trainer",
    "TrainerConfig",
    "TrainingResult",
    "create_distributed_sampler",
    "infer_batch_size",
    "load_training_checkpoint",
    "move_to_device",
    "resolve_batch_adapter",
    "save_training_checkpoint",
    "train_model",
    "training_events",
]
