import tempfile
import unittest
from pathlib import Path

import torch_like
from tools.audit_model_catalog import (
    build_audit,
    load_policy,
    render_json,
    render_markdown,
    validate_committed_outputs,
    validate_evidence_registry,
)


class ModelCatalogAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()
        cls.audit = build_audit(cls.policy)

    def test_every_registry_model_is_audited_and_mapped(self):
        expected = sum(len(torch_like.list_models(domain)) for domain in torch_like.list_domains())
        self.assertEqual(self.audit["summary"]["model_count"], expected)
        self.assertEqual(self.audit["summary"]["migration_ready_count"], expected)
        self.assertEqual(self.audit["summary"]["migration_unmapped_count"], 0)
        self.assertTrue(all(model["migration_status"] == "ready" for model in self.audit["models"]))
        self.assertTrue(all("catalog_v2" in model for model in self.audit["models"]))

    def test_only_explicit_evidence_promotes_maturity(self):
        promoted = [model for model in self.audit["models"] if model["catalog_v2"]["maturity"] != "unassessed"]
        self.assertEqual(len(promoted), self.audit["summary"]["benchmarked_model_count"])
        self.assertEqual(len(promoted), 19)
        self.assertTrue(all(model["catalog_v2"]["evidence_record"] for model in promoted))
        self.assertTrue(all(model["catalog_v2"]["benchmark_evidence"] == "single_dataset" for model in promoted))

    def test_evidence_registry_has_valid_local_provenance(self):
        registry = validate_evidence_registry(self.policy)
        self.assertEqual(len(registry["records"]), 4)
        self.assertEqual(len(registry["model_evidence"]), 19)

    def test_all_registered_classes_follow_nn_module_object_model(self):
        self.assertEqual(self.audit["api_contract_counts"]["nn_module"], len(self.audit["models"]))

    def test_rendering_is_deterministic(self):
        self.assertEqual(render_json(self.audit), render_json(build_audit(self.policy)))
        self.assertEqual(render_markdown(self.audit), render_markdown(build_audit(self.policy)))

    def test_repository_audit_outputs_are_current(self):
        self.assertEqual(validate_committed_outputs(self.audit), [])

    def test_stale_outputs_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "audit.json"
            markdown_path = root / "audit.md"
            json_path.write_text("{}\n", encoding="utf-8")
            markdown_path.write_text("stale\n", encoding="utf-8")
            errors = validate_committed_outputs(self.audit, json_path, markdown_path)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("Stale generated catalog audit" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
