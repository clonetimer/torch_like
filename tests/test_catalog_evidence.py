import unittest

import torch_like
from torch_like.models.catalog_evidence import (
    evidence_metadata_for_model,
    get_evidence_record,
    list_evidence_records,
    load_model_evidence_registry,
)


class CatalogEvidenceTests(unittest.TestCase):
    def test_registry_is_defensive_and_queryable(self):
        registry = load_model_evidence_registry()
        registry["records"].clear()
        self.assertEqual(len(load_model_evidence_registry()["records"]), 4)
        self.assertEqual(len(list_evidence_records("rul")), 2)

    def test_audited_model_receives_evidence_metadata(self):
        info = torch_like.get_model_info("DLinear", domain="forecasting")
        self.assertEqual(info["maturity"], "benchmarked")
        self.assertEqual(info["benchmark_evidence"], "single_dataset")
        self.assertEqual(info["evidence_record"], "ett-h1-forecasting-rolling-v1")

    def test_unassessed_model_is_not_promoted(self):
        self.assertEqual(evidence_metadata_for_model("forecasting", "ModernTCN"), {})
        self.assertEqual(torch_like.get_model_info("ModernTCN", domain="forecasting")["maturity"], "unassessed")

    def test_record_query_is_defensive(self):
        record = get_evidence_record("esa-adb-mission1-lite-v3")
        record["evaluated_models"].clear()
        self.assertEqual(len(get_evidence_record("esa-adb-mission1-lite-v3")["evaluated_models"]), 9)

    def test_unknown_record_fails_explicitly(self):
        with self.assertRaisesRegex(KeyError, "Unknown model evidence record"):
            get_evidence_record("missing")


if __name__ == "__main__":
    unittest.main()
