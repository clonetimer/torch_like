"""Private DDP mechanics; epoch/optimizer lifecycle stays in the public Trainer.

Object collectives carry trusted-worker metadata and bounded metric/RNG states,
never a validation prediction dataset. This is a correctness-first small-world
runtime, not a high-throughput distributed checkpoint implementation.
"""

from __future__ import annotations

import hashlib
import uuid
import weakref
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from torch_like.evaluation import MergeableMetric
from torch_like.training.checkpoint import _atomic_save, _build_payload, _capture_rng, _torch_load
from torch_like.training.distributed import DDPConfig, DistributedEvalSampler, _require_process_group


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


class _LossModule(torch.nn.Module):
    """Put the entire domain objective inside DDP.forward, including model.loss."""

    def __init__(self, model: torch.nn.Module, loss: Callable):
        super().__init__()
        self.model = model
        # A bound Trainer method must not keep the trainer and its DDP wrapper
        # alive after the calling scenario exits. Native DDP reducers may have
        # pending background state that should finalize before group teardown.
        self._loss = weakref.WeakMethod(loss)

    def forward(self, batch: Any) -> torch.Tensor:
        loss = self._loss()
        if loss is None:
            raise RuntimeError("DDP training objective owner is no longer alive")
        return loss(batch)


class _DDPExecution:
    def __init__(self, trainer: Any, config: DDPConfig):
        # Trainer owns this execution helper; avoid a strong reference cycle.
        self.trainer = weakref.proxy(trainer)
        self.context = trainer.distributed_context
        self.device = trainer.device
        self.config = config
        _require_process_group(self.context)
        expected = "nccl" if self.device.type == "cuda" else "gloo"
        if dist.get_backend() != expected:
            raise ValueError("DDP Trainer requires CPU/Gloo or CUDA/NCCL with a matching active group")
        self.ddp: DistributedDataParallel | None = None
        self.resume_signature: dict[str, Any] = {}
        self.val_loader: Any = None
        self.run_id: str | None = None

    def gather(self, value: Any) -> list[Any]:
        output: list[Any] = [None] * self.context.world_size
        dist.all_gather_object(output, value)
        return output

    def check_error(self, stage: str, error: Exception | None) -> None:
        flag = torch.tensor(int(error is not None), device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            messages = self.gather(None if error is None else f"{type(error).__name__}: {error}")
            detail = "; ".join(f"rank {rank}: {msg}" for rank, msg in enumerate(messages) if msg is not None)
            raise RuntimeError(f"Distributed {stage} failed: {detail}") from error

    def guard(self, stage: str, function: Callable[[], Any]) -> Any:
        error = None
        value = None
        try:
            value = function()
        except Exception as exc:
            error = exc
        self.check_error(stage, error)
        return value

    def _loader_description(self, loader: Any, *, training: bool) -> dict[str, Any]:
        if not isinstance(loader, DataLoader) or loader.batch_size is None:
            raise TypeError("DDP requires a map-style DataLoader with a fixed batch_size")
        if type(loader.batch_sampler) is not BatchSampler:
            raise TypeError("Custom batch samplers are outside the DDP 0.9 contract")
        if loader.num_workers != 0:
            raise ValueError("DDP 0.9 requires num_workers=0 for a restorable data/RNG lifecycle")
        sampler = loader.sampler
        if training:
            if type(sampler) is not DistributedSampler:
                raise TypeError("DDP training requires the standard DistributedSampler")
            if (sampler.rank, sampler.num_replicas) != (self.context.rank, self.context.world_size):
                raise ValueError("training sampler rank/world_size mismatch")
            if len(loader) <= 0:
                raise ValueError("DDP training shard has no batches")
            sampling = {"seed": sampler.seed, "shuffle": sampler.shuffle, "drop_last": sampler.drop_last}
        else:
            if type(sampler) is not DistributedEvalSampler or sampler.context != self.context:
                raise TypeError("DDP validation requires DistributedEvalSampler for exact, unpadded evaluation")
            if loader.drop_last:
                raise ValueError("DDP validation must not drop samples")
            sampling = {}
        if sampler.dataset is not loader.dataset:
            raise ValueError("sampler and loader must refer to the same dataset object")
        return {
            "dataset_size": len(loader.dataset),
            "batch_size": loader.batch_size,
            "drop_last": loader.drop_last,
            "sampling": sampling,
            **({"steps": len(loader)} if training else {}),
        }

    def _preflight(self, train_loader: Any, val_loader: Any, resume_from: Any) -> dict[str, Any]:
        trainer = self.trainer
        if isinstance(trainer.model, DistributedDataParallel):
            raise TypeError("Pass the native model, not an already DDP-wrapped model")
        if any(isinstance(module, torch.nn.SyncBatchNorm) for module in trainer.model.modules()):
            raise TypeError("SyncBatchNorm and model-internal collectives are outside the DDP 0.9 contract")
        if trainer.metric_fns:
            raise ValueError("DDP rejects legacy batch metric_fns; use mergeable streaming metrics")
        evaluator = trainer.metric_evaluator
        metric_identity = {}
        if evaluator is not None:
            if not evaluator.fully_streaming or not all(
                isinstance(metric, MergeableMetric) for metric in evaluator.metrics.values()
            ):
                raise TypeError("DDP requires streaming mergeable metrics; buffered callables are single-process only")
            evaluator.reset()
            metric_identity = {
                name: {
                    "class": f"{type(metric).__module__}.{type(metric).__qualname__}",
                    "state": repr(metric.state_dict()),
                }
                for name, metric in evaluator.metrics.items()
            }
        # Rank-local user hyperparameters must not silently produce divergent optimizer steps.
        optimizer = trainer.optimizer
        if optimizer.state and self.ddp is None:
            raise ValueError("Construct a fresh optimizer and use resume_from for DDP optimizer restoration")
        optimizer_groups = [
            repr({key: val for key, val in group.items() if key != "params"}) for group in optimizer.param_groups
        ]
        config = trainer.config.to_dict()
        signature_config = {
            key: config[key]
            for key in ("optimizer", "learning_rate", "weight_decay", "lr_scheduler", "grad_clip_norm", "amp")
        }
        if config["lr_scheduler"] == "cosine":
            signature_config["cosine_total_epochs"] = config["epochs"]
        signature = {
            "training": self._loader_description(train_loader, training=True),
            "validation": self._loader_description(val_loader, training=False) if val_loader is not None else None,
            "config": signature_config,
            "ddp_config": asdict(self.config),
            "model_class": f"{type(trainer.model).__module__}.{type(trainer.model).__qualname__}",
            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "scheduler_class": None if trainer.scheduler is None else type(trainer.scheduler).__qualname__,
            "metrics": metric_identity,
        }
        self.resume_signature = signature
        digest = None
        if resume_from is not None:
            path = Path(resume_from).expanduser().resolve()
            saved = _torch_load(path, "cpu")
            if saved.get("schema_version") != 3:
                raise ValueError("DDP resume requires a coordinated schema-v3 checkpoint, not a rank-local v1/v2 file")
            if saved.get("resume_signature") != signature:
                raise ValueError("DDP resume data/optimizer/metric signature differs from the checkpoint")
            if int(saved["epoch"]) > config["epochs"]:
                raise ValueError("checkpoint epoch exceeds the configured target epochs")
            self.run_id = saved.get("run_id")
            if not isinstance(self.run_id, str) or not self.run_id:
                raise ValueError("DDP checkpoint is missing a run identity")
            if trainer.config.restore_best:
                if not trainer.config.save_best:
                    raise ValueError("DDP restore_best resume requires save_best=True and a persisted best checkpoint")
                best_root = (
                    Path(config["checkpoint_dir"]).expanduser().resolve() if config["checkpoint_dir"] else path.parent
                )
                best = best_root / "best_checkpoint.pt"
                if not best.is_file():
                    raise ValueError("restore_best resume requires the matching best_checkpoint.pt in checkpoint_dir")
                best_payload = _torch_load(best, "cpu")
                if best_payload.get("run_id") != self.run_id or best_payload.get("epoch") != saved.get("best_epoch"):
                    raise ValueError("best checkpoint belongs to a different run or best epoch")
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
            digest = sha.hexdigest()
        comparable = {key: val for key, val in config.items() if key != "device"}
        return {
            "signature": signature,
            "config": comparable,
            "device_type": self.device.type,
            "optimizer_groups": optimizer_groups,
            "resume_digest": digest,
            "resume_from": str(Path(resume_from).expanduser().resolve()) if resume_from is not None else None,
        }

    def setup(self, train_loader: Any, val_loader: Any, resume_from: Any) -> None:
        self.val_loader = val_loader
        identity = self.guard("preflight", lambda: self._preflight(train_loader, val_loader, resume_from))
        identities = self.gather(identity)
        if any(item != identities[0] for item in identities[1:]):
            raise ValueError("DDP ranks disagree on configuration, data layout, metrics or resume checkpoint")
        if self.run_id is None:
            self.run_id = self.gather(uuid.uuid4().hex if self.context.is_primary else None)[0]
        if self.ddp is None:
            objective = _LossModule(self.trainer.model, self.trainer._local_loss)
            self.ddp = DistributedDataParallel(
                objective,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                find_unused_parameters=self.config.find_unused_parameters,
                broadcast_buffers=self.config.broadcast_buffers,
            )

    def batches(self, loader: Any):
        iterator = self.guard("data iterator", lambda: iter(loader))
        for _ in range(len(loader)):
            batch = self.guard("batch preparation", lambda: self.trainer._prepare_batch(next(iterator)))
            count = self.guard("sample count", batch.resolved_sample_count)
            counts = self.gather(count)
            if any(value <= 0 or value != counts[0] for value in counts):
                raise ValueError("DDP requires equal positive sample_count across ranks for each mean-loss step")
            yield batch

    def loss(self, batch: Any) -> torch.Tensor:
        if self.ddp is None:
            raise RuntimeError("DDP execution was not initialized")
        return self.guard("forward/loss", lambda: self.ddp(batch))

    def sum_values(self, values: list[float]) -> list[float]:
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def maximum(self, value: float) -> float:
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def sync_buffers(self) -> None:
        # Evaluation bypasses DDP.forward so uneven/empty validation shards are safe.
        # One rank-0 broadcast before evaluation defines a common inference model.
        with torch.no_grad():
            for value in self.trainer.model.buffers():
                dist.broadcast(value, src=0)

    def merge_metrics(self) -> dict[str, float]:
        evaluator = self.trainer.metric_evaluator
        if evaluator is None:
            return {}
        state = self.guard("metric state", evaluator.state_dict)
        states = self.gather(state)

        def compute():
            evaluator.reset()
            for shard in states:
                evaluator.merge_state(shard)
            return evaluator.compute()

        return self.guard("metric merge", compute)

    def save_checkpoint(self, path: Path, **kwargs: Any) -> None:
        trainer = self.trainer

        def capture():
            return {
                "distributed_context": self.context.to_dict(),
                **_capture_rng(kwargs.get("train_loader"), self.val_loader),
                "model_buffers": {name: value.detach().cpu().clone() for name, value in trainer.model.named_buffers()},
                "scaler_state_dict": trainer.scaler.state_dict(),
                "sampler_epoch": getattr(kwargs.get("train_loader").sampler, "epoch", None),
            }

        rank_state = self.guard("checkpoint snapshot", capture)
        states = self.gather(rank_state)

        def save():
            if not self.context.is_primary:
                return
            payload = _build_payload(**kwargs, val_loader=self.val_loader)
            payload.update(
                {
                    "schema_version": 3,
                    "rng_state_scope": "per_rank",
                    "rank_states": states,
                    "resume_signature": self.resume_signature,
                    "run_id": self.run_id,
                }
            )
            _atomic_save(path, _cpu(payload))

        self.guard("checkpoint write", save)
