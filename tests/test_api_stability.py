import copy
import unittest
from datetime import date

from tools.audit_public_api import (
    build_manifest,
    load_deprecations,
    load_policy,
    render_manifest,
    validate_committed_baseline,
    validate_transition,
)
from torch_like._deprecation import warn_deprecated


class PublicAPIStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()
        cls.manifest = build_manifest(cls.policy)

    def test_policy_and_empty_deprecation_ledger_are_valid(self):
        self.assertEqual(self.policy["minimum_minor_releases"], 2)
        self.assertEqual(self.policy["minimum_deprecation_days"], 90)
        self.assertEqual(load_deprecations(self.policy), [])

    def test_manifest_covers_declared_namespaces_and_is_deterministic(self):
        self.assertEqual(set(self.manifest["namespaces"]), set(self.policy["protected_namespaces"]))
        self.assertIn("torch_like.create_model", self.manifest["symbols"])
        self.assertIn("torch_like.PatchTST", self.manifest["symbols"])
        self.assertIn("torch_like.automl.LeaderboardTask", self.manifest["symbols"])
        self.assertIn("torch_like.nn.FullAttention", self.manifest["symbols"])
        self.assertEqual(render_manifest(self.manifest), render_manifest(build_manifest(self.policy)))

    def test_enum_signatures_are_interpreter_stable(self):
        signature = self.manifest["symbols"]["torch_like.automl.TaskType"]["signature"]
        self.assertEqual([item["name"] for item in signature], ["value", "names", "module", "qualname", "type", "start"])

    def test_repository_public_api_baseline_is_current(self):
        self.assertEqual(validate_committed_baseline(self.manifest), [])

    def test_additive_api_change_needs_no_deprecation(self):
        current = copy.deepcopy(self.manifest)
        current["namespaces"]["torch_like"]["exports"].append("NewPublicSymbol")
        current["symbols"]["torch_like.NewPublicSymbol"] = {
            "kind": "callable",
            "target": "torch_like.new_symbol",
            "signature": [],
        }
        self.assertEqual(validate_transition(self.manifest, current, []), [])

    def test_removal_without_deprecation_is_rejected(self):
        current = self._without_symbol("torch_like.create_model")
        errors = validate_transition(self.manifest, current, [])
        self.assertTrue(any("lacks deprecation record" in error for error in errors))

    def test_removal_requires_elapsed_policy_and_removed_status(self):
        current = self._without_symbol("torch_like.create_model")
        current["package_version"] = "0.5.0"
        record = self._record("active")
        errors = validate_transition(self.manifest, current, [record], today=date(2026, 4, 2))
        self.assertTrue(any("must be marked removed" in error for error in errors))

        record["status"] = "removed"
        self.assertEqual(validate_transition(self.manifest, current, [record], today=date(2026, 4, 2)), [])

    def test_signature_change_requires_specific_deprecation(self):
        current = copy.deepcopy(self.manifest)
        current["symbols"]["torch_like.create_model"]["signature"] = []
        errors = validate_transition(self.manifest, current, [])
        self.assertTrue(any("signature" in error for error in errors))

    def test_adding_an_optional_parameter_is_compatible(self):
        current = copy.deepcopy(self.manifest)
        current["symbols"]["torch_like.list_models"]["signature"].append(
            {
                "name": "include_hidden",
                "kind": "KEYWORD_ONLY",
                "required": False,
                "default": False,
            }
        )
        self.assertEqual(validate_transition(self.manifest, current, []), [])

    def test_runtime_warning_is_user_visible_and_actionable(self):
        with self.assertWarnsRegex(FutureWarning, "Use torch_like.new_api instead"):
            warn_deprecated(
                "torch_like.old_api",
                deprecated_in="0.3.0",
                remove_in="0.5.0",
                replacement="torch_like.new_api",
            )

    def _without_symbol(self, symbol: str):
        current = copy.deepcopy(self.manifest)
        namespace, name = symbol.rsplit(".", maxsplit=1)
        current["namespaces"][namespace]["exports"].remove(name)
        current["symbols"].pop(symbol)
        return current

    @staticmethod
    def _record(status: str):
        return {
            "symbol": "torch_like.create_model",
            "change_type": "remove",
            "deprecated_in": "0.3.0",
            "remove_after": "0.5.0",
            "announced_on": "2026-01-01",
            "remove_after_date": "2026-04-02",
            "replacement": None,
            "reason": "Test transition",
            "status": status,
        }


if __name__ == "__main__":
    unittest.main()
