"""Common model base classes and helpers."""

from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from functools import wraps
from typing import Any, Dict

import torch.nn as nn
from torch.nn.parameter import UninitializedParameter


class BaseModel(nn.Module):
    """Shared base for torch_like models."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__dict__.get("__init__")
        if original_init is None or getattr(original_init, "_captures_init_config", False):
            return
        signature = inspect.signature(original_init)

        @wraps(original_init)
        def wrapped_init(self, *args, **init_kwargs):
            bound = signature.bind(self, *args, **init_kwargs)
            bound.apply_defaults()
            original_init(self, *args, **init_kwargs)
            captured = {}
            for name, value in bound.arguments.items():
                if name == "self":
                    continue
                parameter = signature.parameters[name]
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    captured.update(value)
                elif parameter.kind != inspect.Parameter.VAR_POSITIONAL:
                    captured[name] = value
            self._init_config = captured

        wrapped_init._captures_init_config = True
        cls.__init__ = wrapped_init

    def __init__(self):
        super().__init__()
        self._param_count = None

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not isinstance(p, UninitializedParameter))

    def config(self) -> Dict[str, Any]:
        init_config = getattr(self, "_init_config", None)
        if init_config is not None:
            return dict(init_config)
        cfg = getattr(self, "_config", None)
        if cfg is None:
            return {}
        if is_dataclass(cfg):
            return asdict(cfg)
        if isinstance(cfg, dict):
            return dict(cfg)
        return {"value": cfg}

    @classmethod
    def from_config(cls, config):
        if is_dataclass(config):
            return cls(**asdict(config))
        if isinstance(config, dict):
            return cls(**config)
        raise TypeError("config must be a dataclass instance or dict")
