"""Shared runner utilities."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import torch


def ensure_results_dir(base_dir: str | Path) -> Path:
    out_dir = Path(base_dir) / "results"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def resolve_results_dir(results_dir: str | Path | None, default_base_dir: str | Path) -> Path:
    if results_dir is None:
        return ensure_results_dir(default_base_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(path: str | Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def print_epoch_metrics(epoch: int, epochs: int, **metrics):
    formatted = " ".join(f"{name}={value:.6f}" for name, value in metrics.items())
    print(f"Epoch {epoch:03d}/{epochs} {formatted}")


def average_total(total: float, count: int) -> float:
    if count <= 0:
        raise ValueError("Cannot average metrics from an empty loader")
    return total / count


def _to_serializable_config(config):
    if is_dataclass(config):
        return {field.name: _to_serializable_config(getattr(config, field.name)) for field in fields(config)}
    if isinstance(config, np.ndarray):
        return {"type": "ndarray", "shape": list(config.shape), "dtype": str(config.dtype)}
    if isinstance(config, torch.Tensor):
        return {"type": "tensor", "shape": list(config.shape), "dtype": str(config.dtype)}
    if isinstance(config, Path):
        return str(config)
    if isinstance(config, dict):
        return {str(key): _to_serializable_config(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [_to_serializable_config(value) for value in config]
    return config


def build_run_payload(config, history, extra=None, metrics=None, results=None):
    payload = {
        "config": _to_serializable_config(config),
        "history": history,
        "extra": extra or {},
    }
    if metrics is not None:
        payload["metrics"] = metrics
    if results is not None:
        payload["results"] = results
    return payload
