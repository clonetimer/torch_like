"""Core utilities for torch_like: effect scopes and events.

These primitives are a small, Python-native adaptation of the Cordis plugin
runtime (Context / Effect / Event). They are intentionally minimal -- only the
mechanisms the model library actually needs -- and have no dependency on
PyTorch, so they can be imported cheaply and resolved safely inside worker
subprocesses.
"""

from torch_like.core.events import EventEmitter
from torch_like.core.scope import Disposer, Scope, ScopeDisposeError

__all__ = [
    "Disposer",
    "EventEmitter",
    "Scope",
    "ScopeDisposeError",
]
