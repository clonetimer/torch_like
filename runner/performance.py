"""Shared execution policies for auxiliary training runners."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass
class ExecutionConfig:
    """Runtime-only performance options that do not change model semantics."""

    mixed_precision: bool = False
    amp_dtype: str = "bfloat16"
    non_blocking: bool = True
    zero_grad_set_to_none: bool = True

    def __post_init__(self) -> None:
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")


def _device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def move_to_device(tensor: torch.Tensor, device: str | torch.device, config: ExecutionConfig) -> torch.Tensor:
    non_blocking = config.non_blocking and _device_type(device) == "cuda"
    return tensor.to(device, non_blocking=non_blocking)


def autocast_context(device: str | torch.device, config: ExecutionConfig):
    if not config.mixed_precision:
        return nullcontext()
    dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type=_device_type(device), dtype=dtype)


def create_grad_scaler(device: str | torch.device, config: ExecutionConfig):
    if not (config.mixed_precision and config.amp_dtype == "float16" and _device_type(device) == "cuda"):
        return None
    grad_scaler = getattr(torch.amp, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda")
    return torch.cuda.amp.GradScaler()


def backward_and_step(loss: torch.Tensor, optimizer, scaler, config: ExecutionConfig) -> None:
    optimizer.zero_grad(set_to_none=config.zero_grad_set_to_none)
    if scaler is None:
        loss.backward()
        optimizer.step()
        return
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def add_execution_args(parser) -> None:
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--blocking-transfers", action="store_true", help="Disable asynchronous CUDA transfers.")
    parser.add_argument(
        "--zero-grad-values", action="store_true", help="Zero gradients instead of setting them to None."
    )


def execution_config_from_args(args) -> ExecutionConfig:
    return ExecutionConfig(
        mixed_precision=args.amp,
        amp_dtype=args.amp_dtype,
        non_blocking=not args.blocking_transfers,
        zero_grad_set_to_none=not args.zero_grad_values,
    )
