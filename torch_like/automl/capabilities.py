"""Config-driven capability assembly for one leaderboard run.

A run mounts the backends its config allows into a disposable
:class:`~torch_like.core.Scope`: anything the config leaves out is suppressed
for the duration of the run and restored when the scope is disposed. This is the
Cordis "Context" idea -- capabilities are assembled per run instead of being
globally hard-wired -- and it is what makes :mod:`torch_like.automl.backends`
and :mod:`torch_like.automl.candidate_registry` revocable in practice.

When a config does not restrict backends the returned scope stays empty, so an
unrestricted run touches no global state at all.
"""

from __future__ import annotations

from torch_like.automl.backends import has_backend, list_backends, suppress_backend
from torch_like.automl.candidate_registry import suppress_candidates_for_backend
from torch_like.automl.specs import LeaderboardConfig
from torch_like.core import Scope


def assemble_capabilities(config: LeaderboardConfig, scope: Scope | None = None) -> Scope:
    """Mount the config-enabled backends onto ``scope`` (created when omitted).

    ``config.search.backends`` acts as an allowlist: every other registered
    backend -- and every candidate targeting one -- is suppressed until the
    returned scope is disposed.
    """
    active = scope if scope is not None else Scope(f"capabilities:{config.task_id}")
    requested = config.search.backends
    if not requested:
        return active
    unknown = sorted(name for name in requested if not has_backend(name))
    if unknown:
        known = sorted(spec.name for spec in list_backends())
        raise ValueError(f"Unknown backend(s) in search.backends: {unknown}; registered={known}")
    enabled = set(requested)
    for spec in list_backends():
        if spec.name in enabled:
            continue
        suppress_backend(spec.name, active)
        suppress_candidates_for_backend(spec.name, active)
    return active


__all__ = ["assemble_capabilities"]
