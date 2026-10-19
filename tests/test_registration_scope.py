"""Revocable registration: Cordis "registrations are effects" via Scope.

A plugin may register a candidate and/or a backend inside a disposable
:class:`torch_like.core.Scope`; disposing the scope removes exactly those
entries, leaving builtins and unrelated registrations intact.
"""

import unittest

from torch_like.automl.backends import BackendSpec, get_backend, register_backend
from torch_like.automl.candidate_registry import (
    _CANDIDATES,
    CandidateSpec,
    list_candidate_specs,
    register_candidate,
)
from torch_like.automl.specs import TaskType
from torch_like.core import Scope


def _candidate_names(task_type: TaskType = TaskType.TABULAR_REGRESSION) -> set[str]:
    return {spec.name for spec in list_candidate_specs(task_type, is_sequence=False, equal_sequence_lengths=True)}


class CandidateRegistrationScopeTests(unittest.TestCase):
    def test_registration_removed_on_scope_dispose(self):
        scope = Scope("plugin")
        spec = CandidateSpec("ScopeProbeModel", (TaskType.TABULAR_REGRESSION,), "sklearn", ({"alpha": 1.0},))
        self.assertNotIn("ScopeProbeModel", _candidate_names())
        register_candidate(spec, scope=scope)
        self.assertIn("ScopeProbeModel", _candidate_names())
        scope.dispose()
        self.assertNotIn("ScopeProbeModel", _candidate_names())

    def test_scope_context_manager_removes_registration(self):
        spec = CandidateSpec("ScopeCtxModel", (TaskType.TABULAR_REGRESSION,), "sklearn", ({"alpha": 1.0},))
        with Scope("plugin") as scope:
            register_candidate(spec, scope=scope)
            self.assertIn("ScopeCtxModel", _candidate_names())
        self.assertNotIn("ScopeCtxModel", _candidate_names())

    def test_recan_register_after_dispose(self):
        spec = CandidateSpec("ScopeReuseModel", (TaskType.TABULAR_REGRESSION,), "sklearn", ({"alpha": 1.0},))
        first = Scope("first")
        register_candidate(spec, scope=first)
        first.dispose()
        self.assertNotIn("ScopeReuseModel", _candidate_names())
        second = Scope("second")
        register_candidate(spec, scope=second)
        self.assertIn("ScopeReuseModel", _candidate_names())
        second.dispose()
        self.assertNotIn("ScopeReuseModel", _candidate_names())

    def test_dispose_skips_entry_replaced_by_another(self):
        spec = CandidateSpec("ScopeClobberProbe", (TaskType.TABULAR_REGRESSION,), "sklearn", ({"alpha": 1.0},))
        other = CandidateSpec("ScopeClobberProbe", (TaskType.TABULAR_REGRESSION,), "sklearn", ({"alpha": 2.0},))
        scope = Scope("plugin")
        register_candidate(spec, scope=scope)
        # Pretend another writer replaced the entry without going through this scope.
        _CANDIDATES["ScopeClobberProbe"] = other
        scope.dispose()  # Guard: must NOT delete the unrelated 'other' entry.
        self.assertIs(_CANDIDATES.get("ScopeClobberProbe"), other)
        del _CANDIDATES["ScopeClobberProbe"]


class BackendRegistrationScopeTests(unittest.TestCase):
    def test_registration_removed_on_scope_dispose(self):
        scope = Scope("plugin")
        spec = BackendSpec("scope_torch", "torch", "torch_like")
        register_backend(spec, scope=scope)
        self.assertIs(get_backend("scope_torch"), spec)
        scope.dispose()
        with self.assertRaises(KeyError):
            get_backend("scope_torch")

        scope2 = Scope("plugin2")
        register_backend(spec, scope=scope2)
        self.assertIs(get_backend("scope_torch"), spec)
        scope2.dispose()
        with self.assertRaises(KeyError):
            get_backend("scope_torch")

    def test_builtins_persist_independent_of_plugin_scopes(self):
        scope = Scope("plugin")
        register_backend(BackendSpec("ephemeral_backend", "torch", "torch_like"), scope=scope)
        scope.dispose()
        self.assertIsNotNone(get_backend("sklearn"))
        self.assertIsNotNone(get_backend("pytorch"))
        self.assertIsNotNone(get_backend("baseline"))
        with self.assertRaises(KeyError):
            get_backend("ephemeral_backend")


class PluginScopeIntegrationTests(unittest.TestCase):
    def test_single_scope_tears_down_candidate_and_backend(self):
        with Scope("whole_plugin") as plugin:
            register_candidate(
                CandidateSpec("PluginModel", (TaskType.TABULAR_REGRESSION,), "sklearn", ({"alpha": 1.0},)),
                scope=plugin,
            )
            register_backend(BackendSpec("plugin_backend", "sklearn", "baseline"), scope=plugin)
            self.assertIn("PluginModel", _candidate_names())
            self.assertIsNotNone(get_backend("plugin_backend"))
        self.assertNotIn("PluginModel", _candidate_names())
        with self.assertRaises(KeyError):
            get_backend("plugin_backend")


if __name__ == "__main__":
    unittest.main()
