"""Optional torch.compile helpers for runner-level acceleration."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CompileConfig:
    enabled: bool = False
    backend: str | None = None
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None


def maybe_compile_model(model, config: CompileConfig):
    if not config.enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    kwargs = {"fullgraph": config.fullgraph}
    if config.backend is not None:
        kwargs["backend"] = config.backend
    if config.mode is not None:
        kwargs["mode"] = config.mode
    if config.dynamic is not None:
        kwargs["dynamic"] = config.dynamic
    return compile_fn(model, **kwargs)
