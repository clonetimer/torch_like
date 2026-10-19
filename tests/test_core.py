"""Tests for the Cordis-style core primitives and the AutoML extension seams.

These exercise the new, untested-by-import code paths so the project's
coverage gate stays satisfied after the refactor.
"""

import unittest

from torch_like.automl.backends import (
    BackendSpec,
    backends_by_kind,
    get_backend,
    list_backends,
    register_backend,
)
from torch_like.automl.data import register_dataset_format
from torch_like.automl.sklearn_backend import AlgorithmCandidate
from torch_like.core.events import EventEmitter
from torch_like.core.scope import Scope, ScopeDisposeError


class ScopeTests(unittest.TestCase):
    def test_dispose_runs_in_lifo_order(self):
        order: list[int] = []
        with Scope("s") as sc:
            sc.add(lambda: order.append(1))
            sc.add(lambda: order.append(2))
        self.assertEqual(order, [2, 1])

    def test_add_after_dispose_raises(self):
        sc = Scope()
        sc.dispose()
        with self.assertRaises(RuntimeError):
            sc.add(lambda: None)

    def test_double_dispose_is_idempotent(self):
        sc = Scope()
        sc.add(lambda: None)
        sc.dispose()
        sc.dispose()

    def test_dispose_aggregates_errors(self):
        def boom():
            raise ValueError("nope")

        sc = Scope()
        sc.add(boom)
        sc.add(lambda: None)
        with self.assertRaises(ScopeDisposeError) as ctx:
            sc.dispose()
        self.assertEqual(len(ctx.exception.errors), 1)

    def test_child_nesting(self):
        parent = Scope("parent")
        child = parent.child("child")
        self.assertIs(child.parent, parent)
        self.assertTrue(child.active)


class EventEmitterTests(unittest.TestCase):
    def test_emit_broadcasts(self):
        received: list[int] = []
        emitter = EventEmitter()
        dispose = emitter.on(lambda value: received.append(value))
        emitter.emit(7)
        self.assertEqual(received, [7])
        dispose()
        emitter.emit(8)
        self.assertEqual(received, [7])

    def test_waterfall_transforms(self):
        emitter = EventEmitter()
        emitter.on(lambda v: v + 1)
        emitter.on(lambda v: v * 2)
        self.assertEqual(emitter.waterfall(3), 8)

    def test_waterfall_keeps_previous_on_none(self):
        emitter = EventEmitter()
        emitter.on(lambda v: None)
        emitter.on(lambda v: v * 10)
        self.assertEqual(emitter.waterfall(5), 50)

    def test_non_callable_handler_raises(self):
        with self.assertRaises(TypeError):
            EventEmitter().on(123)  # type: ignore[arg-type]


class BackendRegistryTests(unittest.TestCase):
    def test_builtins_present(self):
        names = {spec.name for spec in list_backends()}
        for expected in {"sklearn", "baseline", "pytorch", "anomaly_pytorch", "anomaly_baseline"}:
            self.assertIn(expected, names)

    def test_kind_and_family(self):
        self.assertEqual(get_backend("pytorch").kind, "torch")
        self.assertEqual(get_backend("pytorch").model_family, "torch_like")
        self.assertEqual(get_backend("sklearn").kind, "sklearn")
        self.assertEqual(get_backend("sklearn").model_family, "baseline")

    def test_model_family_via_candidate(self):
        candidate = AlgorithmCandidate(name="DLinear", preprocessing="none", parameters={}, backend="pytorch")
        self.assertEqual(candidate.model_family, "torch_like")
        baseline = AlgorithmCandidate(name="LastValue", preprocessing="none", parameters={}, backend="baseline")
        self.assertEqual(baseline.model_family, "baseline")

    def test_backends_by_kind(self):
        torch_backends = {spec.name for spec in backends_by_kind("torch")}
        self.assertEqual(torch_backends, {"pytorch", "anomaly_pytorch"})

    def test_register_validation(self):
        with self.assertRaises(ValueError):
            register_backend(BackendSpec("", "torch", "torch_like"))
        with self.assertRaises(ValueError):
            register_backend(BackendSpec("weird", "cuda", "torch_like"))
        with self.assertRaises(ValueError):
            register_backend(BackendSpec("weird", "torch", "unknown"))

    def test_unknown_backend_raises(self):
        with self.assertRaises(KeyError):
            get_backend("nonexistent")


class DatasetFormatRegistryTests(unittest.TestCase):
    def test_builtins_registered(self):
        from torch_like.automl import data as data_mod
        from torch_like.automl.specs import DatasetFormat

        keys = {str(fmt) for fmt in DatasetFormat}
        self.assertTrue(keys <= set(data_mod._FORMAT_LOADERS))

    def test_dispatch_and_unknown(self):
        import torch_like.automl.data as data_mod
        from torch_like.automl.specs import DatasetSpec

        original = dict(data_mod._FORMAT_LOADERS)
        try:
            data_mod._FORMAT_LOADERS.clear()
            with self.assertRaises(ValueError):
                data_mod.prepare_dataset(DatasetSpec(path="x", label_columns=("y",)))
        finally:
            data_mod._FORMAT_LOADERS.clear()
            data_mod._FORMAT_LOADERS.update(original)

    def test_non_callable_loader_rejected(self):
        from torch_like.automl.specs import DatasetFormat

        with self.assertRaises(TypeError):
            register_dataset_format(DatasetFormat.CSV, 123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
