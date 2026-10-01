"""Training event plumbing for AutoML runs.

Trials execute in spawned subprocesses, so a subscriber sitting in the parent
process never observes the events ``torch_backend`` emits. This module bridges
that gap without abandoning the in-process bus:

* :data:`training_events` stays the in-process bus (tests and direct calls).
* When enabled, a trial appends its events to ``<trial_dir>/events.jsonl`` -- an
  auditable artifact alongside ``metrics.json`` and ``training_summary.json``.
* The orchestrator replays each trial's log onto a **per-run** emitter mounted on
  the run's :class:`~torch_like.core.Scope`, so one run can be observed as a
  single stream and no state leaks between runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch_like.core.events import EventEmitter
from torch_like.core.scope import Scope
from torch_like.training.events import training_events

EVENT_LOG_NAME = "events.jsonl"


_current_run: EventEmitter | None = None


def current_run_events() -> EventEmitter | None:
    """The emitter mounted for the run in progress, or ``None`` outside a run."""
    return _current_run


def bind_run_events(scope: Scope) -> EventEmitter:
    """Mount a per-run emitter onto ``scope``; disposing the scope restores the previous one."""
    global _current_run
    previous = _current_run
    emitter = EventEmitter(scope=scope, name="run_events")
    _current_run = emitter

    def _restore() -> None:
        global _current_run
        _current_run = previous

    scope.add(_restore)
    return emitter


def event_log_path(trial_dir: Path) -> Path:
    return trial_dir / EVENT_LOG_NAME


def install_event_log(trial_dir: Path) -> Callable[[], None]:
    """Record this process's training events into ``trial_dir/events.jsonl``.

    Returns the unsubscribe callable. Installed inside the trial subprocess, so
    events that would otherwise die with the child are captured instead.
    """
    path = event_log_path(trial_dir)

    def _record(name: str, **kwargs: Any) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": name, **kwargs}, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # an audit log must never fail a trial

    return training_events.on(_record)


def read_event_log(trial_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read back a trial's recorded events as ``(name, payload)`` pairs."""
    path = event_log_path(trial_dir)
    if not path.is_file():
        return []
    events: list[tuple[str, dict[str, Any]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = dict(json.loads(stripped))
        events.append((str(record.pop("event")), record))
    return events


def replay_event_log(trial_dir: Path, emitter: EventEmitter, *, trial_id: str) -> int:
    """Re-emit a trial's recorded events onto ``emitter``, tagged with ``trial_id``."""
    count = 0
    for name, payload in read_event_log(trial_dir):
        emitter.emit(name, trial_id=trial_id, **payload)
        count += 1
    return count


__all__ = [
    "EVENT_LOG_NAME",
    "bind_run_events",
    "current_run_events",
    "event_log_path",
    "install_event_log",
    "read_event_log",
    "replay_event_log",
    "training_events",
]
