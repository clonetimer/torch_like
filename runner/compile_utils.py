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
    options: dict | None = None


def maybe_compile_model(model, config: CompileConfig):
    if not config.enabled:
        return model
    kwargs = {"fullgraph": config.fullgraph}
    if config.backend is not None:
        kwargs["backend"] = config.backend
    if config.mode is not None:
        kwargs["mode"] = config.mode
    if config.dynamic is not None:
        kwargs["dynamic"] = config.dynamic
    if config.options is not None:
        kwargs["options"] = config.options

    module_compile = getattr(model, "compile", None)
    if callable(module_compile):
        module_compile(**kwargs)
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    return compile_fn(model, **kwargs)
