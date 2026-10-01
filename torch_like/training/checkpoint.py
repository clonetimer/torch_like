"""Atomic training-state checkpoints for exact epoch-boundary resume."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch_like.training.distributed import DistributedContext

_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3}


def _torch_load(path: Path, map_location: str | torch.device | None) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch releases
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError("Training checkpoint payload must be a dictionary")
    return payload


def _capture_rng(train_loader: Any = None, val_loader: Any = None) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    train_generator = getattr(train_loader, "generator", None)
    val_generator = getattr(val_loader, "generator", None)
    return {
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "cuda_rng_state": torch.cuda.get_rng_state().cpu() if torch.cuda.is_initialized() else None,
        "loader_generator_state": train_generator.get_state() if train_generator is not None else None,
        "val_loader_generator_state": val_generator.get_state() if val_generator is not None else None,
    }


def _restore_rng(state: dict[str, Any], train_loader: Any = None, val_loader: Any = None) -> None:
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    numpy_state = state.get("numpy_rng_state")
    if numpy_state is not None:
        np.random.set_state((
            numpy_state["algorithm"], numpy_state["keys"].cpu().numpy().astype(np.uint32),
            numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
        ))
    if torch.cuda.is_available():
        if state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu())
        elif state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_state_all"]])
    for loader, key in ((train_loader, "loader_generator_state"), (val_loader, "val_loader_generator_state")):
        generator = getattr(loader, "generator", None)
        if generator is not None and state.get(key) is not None:
            generator.set_state(state[key].cpu())


def _atomic_save(target: Path, payload: dict[str, Any]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _build_payload(
    *, model: Any, optimizer: Any, scheduler: Any = None, scaler: Any = None,
    epoch: int, history: list[dict[str, Any]], best_monitored_loss: float, best_epoch: int,
    epochs_without_improvement: int, trainer_config: dict[str, Any], train_loader: Any = None,
    distributed_context: DistributedContext | None = None, val_loader: Any = None,
) -> dict[str, Any]:
    context = distributed_context or DistributedContext()
    from torch_like import __version__

    return {
        "schema_version": _SCHEMA_VERSION,
        "library": "torch_like",
        "library_version": __version__,
        "model_class_path": f"{type(model).__module__}.{type(model).__qualname__}",
        "kind": "training_checkpoint",
        "distributed_context": context.to_dict(),
        "rng_state_scope": "rank_local",
        "epoch": int(epoch),
        "history": history,
        "best_monitored_loss": float(best_monitored_loss),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "trainer_config": dict(trainer_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        **_capture_rng(train_loader, val_loader),
    }


def save_training_checkpoint(
    path: str | os.PathLike[str], *, model: Any, optimizer: Any, scheduler: Any = None,
    scaler: Any = None, epoch: int, history: list[dict[str, Any]], best_monitored_loss: float,
    best_epoch: int, epochs_without_improvement: int, trainer_config: dict[str, Any],
    train_loader: Any = None, distributed_context: DistributedContext | None = None,
    val_loader: Any = None,
) -> Path:
    """Atomically save a rank-local checkpoint (DDP coordination belongs to Trainer)."""

    payload = _build_payload(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch,
        history=history, best_monitored_loss=best_monitored_loss, best_epoch=best_epoch,
        epochs_without_improvement=epochs_without_improvement, trainer_config=trainer_config,
        train_loader=train_loader, distributed_context=distributed_context, val_loader=val_loader,
    )
    return _atomic_save(Path(path).expanduser().resolve(), payload)


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    train_loader: Any = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
    distributed_context: DistributedContext | None = None,
    val_loader: Any = None,
) -> dict[str, Any]:
    """Load a training checkpoint into already-constructed runtime objects."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {source}")
    payload = _torch_load(source, map_location)
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS or payload.get("kind") != "training_checkpoint":
        raise ValueError(f"Unsupported torch_like training checkpoint: {source}")
    saved_context = (
        DistributedContext.from_mapping(payload.get("distributed_context"))
        if schema_version >= 2
        else DistributedContext()
    )
    expected_context = distributed_context or DistributedContext()
    if saved_context.world_size != expected_context.world_size:
        raise ValueError(
            "training checkpoint world_size does not match current context: "
            f"{saved_context.world_size} != {expected_context.world_size}"
        )
    local_state = payload
    if schema_version == 3:
        rank_states = payload.get("rank_states")
        if not isinstance(rank_states, list) or len(rank_states) != expected_context.world_size:
            raise ValueError("distributed checkpoint is missing rank-local states")
        local_state = rank_states[expected_context.rank]
        rank_context = DistributedContext.from_mapping(local_state.get("distributed_context"))
        if rank_context != expected_context:
            raise ValueError("training checkpoint rank/local_rank/world_size mapping does not match current context")
        saved_context = rank_context
    if saved_context.rank != expected_context.rank:
        raise ValueError(
            "training checkpoint rank does not match current context: "
            f"{saved_context.rank} != {expected_context.rank}"
        )
    if "model_state_dict" not in payload:
        raise KeyError("Training checkpoint is missing 'model_state_dict'")
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if schema_version == 3:
        buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, value in local_state.get("model_buffers", {}).items():
                if name not in buffers:
                    raise ValueError(f"checkpoint has an unknown model buffer: {name}")
                buffers[name].copy_(value.to(buffers[name].device))
        if scaler is not None and local_state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(local_state["scaler_state_dict"])
    if restore_rng:
        _restore_rng(local_state, train_loader, val_loader)
    return payload


__all__ = ["load_training_checkpoint", "save_training_checkpoint"]
