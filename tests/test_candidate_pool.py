import copy
import tempfile
import unittest
from datetime import date
from pathlib import Path

from tools.validate_candidate_pool import (
    CandidateScore,
    load_policy,
    load_registry_baseline,
    load_task_adapter_approvals,
    parse_hard_gates,
    parse_review_register,
    parse_scorecard,
    validate_document,
    validate_scores,
)


class CandidatePoolValidationTests(unittest.TestCase):
    def test_task_adapter_approvals_are_structurally_valid(self):
        approvals = load_task_adapter_approvals()
        self.assertEqual(
            {record["name"] for record in approvals},
            {"TCNRUL", "PatchTSTRUL", "TransformerRUL", "CBAMTCNDetector"},
        )

    @classmethod
    def setUpClass(cls):
        cls.document = Path("docs/MODEL_CANDIDATE_POOL.md").read_text(encoding="utf-8")

    def validate_text(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.md"
            path.write_text(document, encoding="utf-8")
            return validate_document(path)

    def test_repository_candidate_pool_passes(self):
        errors = validate_document(Path("docs/MODEL_CANDIDATE_POOL.md"))
        self.assertEqual(errors, [])

    def test_parser_requires_canonical_scorecard(self):
        with self.assertRaisesRegex(ValueError, "canonical scorecard"):
            parse_scorecard("# empty")

    def test_hard_gate_parser_reads_all_candidates(self):
        gates = parse_hard_gates(self.document)
        self.assertEqual(len(gates), 16)
        self.assertEqual(sum(gate.gate == "fail" for gate in gates), 2)

    def test_invalid_value_final_and_decision_are_reported_together(self):
        row = CandidateScore(
            model="Broken",
            dimensions=(25, 20, 15, 10, 10, 10, 10),
            value=99,
            evidence="B",
            risk=0,
            final=99,
            decision="rejected",
        )
        errors = validate_scores([row])
        self.assertTrue(any("dimension sum" in error for error in errors))
        self.assertTrue(any("calculated" in error for error in errors))
        self.assertTrue(any("decision" in error for error in errors))

    def test_dimension_bounds_and_evidence_grade_are_enforced(self):
        row = CandidateScore(
            model="Broken",
            dimensions=(26, 20, 15, 10, 10, 10, 10),
            value=101,
            evidence="Z",
            risk=-1,
            final=0,
            decision="observation",
        )
        errors = validate_scores([row])
        self.assertTrue(any("outside" in error for error in errors))
        self.assertTrue(any("unknown evidence" in error for error in errors))

    def test_missing_dimension_rationale_is_rejected(self):
        line = next(line for line in self.document.splitlines() if line.startswith("| EGNN | geometric graph gap"))
        document = self.document.replace(line, "", 1)
        errors = self.validate_text(document)
        self.assertIn("EGNN: missing Dimension Rationale row", errors)

    def test_failed_hard_gate_cannot_receive_formal_score(self):
        document = self.document.replace(
            "| RT-DETR (CVPR 2024) | yes | yes | yes | yes | yes | Apache-2.0 | pass |",
            "| RT-DETR (CVPR 2024) | yes | yes | yes | yes | yes | Apache-2.0 | **fail** |",
            1,
        )
        errors = self.validate_text(document)
        self.assertTrue(any("failed hard gate must not receive" in error for error in errors))

    def test_passing_gate_cannot_hide_unknown_license(self):
        document = self.document.replace(
            "| RT-DETR (CVPR 2024) | yes | yes | yes | yes | yes | Apache-2.0 | pass |",
            "| RT-DETR (CVPR 2024) | yes | yes | yes | yes | yes | unknown | pass |",
            1,
        )
        errors = self.validate_text(document)
        self.assertTrue(any("passing hard gate contains unknown" in error for error in errors))

    def test_detailed_record_requires_all_fields_and_two_peers(self):
        document = self.document.replace("- Core value: efficient hybrid", "- Value: efficient hybrid", 1)
        document = document.replace("than `DETR`", "than DETR", 1)
        errors = self.validate_text(document)
        self.assertIn("RT-DETR (CVPR 2024): detailed record is missing 'Core value'", errors)
        self.assertIn("RT-DETR (CVPR 2024): Peer comparison must name at least two peer models", errors)

    def test_all_three_decision_classes_are_required(self):
        document = self.document.replace("| observation |", "| rejected |")
        document = document.replace("| observation, no code |", "| rejected, no code |")
        document = document.replace("| observation, implemented |", "| rejected, implemented |")
        errors = self.validate_text(document)
        self.assertTrue(any("Scorecard must contain" in error and "observation" in error for error in errors))

    def test_evidence_grade_requires_labeled_source_types(self):
        document = self.document.replace("[independent: YOLOv10 counter-comparison]", "[comparison: YOLOv10]", 1)
        document = document.replace("[independent: DETR review comparison]", "[review: DETR comparison]", 1)
        errors = self.validate_text(document)
        self.assertTrue(any("RT-DETR: evidence grade A requires 2 independent" in error for error in errors))

    def test_detailed_score_must_match_scorecard(self):
        document = self.document.replace(
            "- Score: value 91; evidence A; risk 4; final 87.",
            "- Score: value 91; evidence B; risk 4; final 87.",
            1,
        )
        errors = self.validate_text(document)
        self.assertIn("RT-DETR: detailed Score does not match scorecard", errors)

    def test_implemented_registry_claim_must_match_real_catalog(self):
        document = self.document.replace(
            "- Registry: `RTDETR`; integration level `paper_aligned_portable_subset`.",
            "- Registry: `RTDETR`; integration level `paper_aligned_reference`.",
            1,
        )
        errors = self.validate_text(document)
        self.assertTrue(any("RT-DETR: claimed integration level" in error for error in errors))

    def test_decision_must_match_detailed_section(self):
        document = self.document.replace(
            "| Mamba-2 / SSD | 23 | 18 | 14 | 10 | 9 | 9 | 9 | 92 | B | 5 | 77.8 | observation, implemented |",
            "| Mamba-2 / SSD | 23 | 18 | 14 | 10 | 9 | 9 | 9 | 92 | B | 5 | 77.8 | rejected, implemented |",
            1,
        )
        errors = self.validate_text(document)
        self.assertTrue(any("Mamba-2 / SSD: decision" in error for error in errors))

    def test_declared_minimum_candidate_count_is_enforced(self):
        document = self.document.replace("- Minimum candidate count: 12.", "- Minimum candidate count: 17.", 1)
        errors = self.validate_text(document)
        self.assertIn("Hard-gate pool has 16 candidates, below declared minimum 17", errors)

    def test_scope_metadata_is_required(self):
        document = self.document.replace("- Target domain:", "- Domain:", 1)
        document = document.replace("- Evaluation date:", "- Date:", 1)
        errors = self.validate_text(document)
        self.assertIn("Scope is missing a non-empty Target domain", errors)
        self.assertIn("Scope is missing Evaluation date", errors)

    def test_failed_gate_record_still_requires_complete_fields(self):
        document = self.document.replace(
            "- Missing information: official code, explicit license", "- Unknowns: official code, explicit license", 1
        )
        errors = self.validate_text(document)
        self.assertIn("Titans (NeurIPS 2025): detailed record is missing 'Missing information'", errors)

    def test_repository_policy_is_machine_readable(self):
        policy = load_policy(Path("governance/model_admission_policy.json"))
        self.assertEqual(policy["policy_version"], "1.0")
        self.assertEqual(sum(policy["dimensions"].values()), 100)
        self.assertGreater(policy["thresholds"]["core"], policy["thresholds"]["observation"])

    def test_every_candidate_has_one_review_record(self):
        gates = parse_hard_gates(self.document)
        reviews = parse_review_register(self.document)
        self.assertEqual(len(reviews), len(gates))
        self.assertTrue(all(review.owner == "maintainers" for review in reviews))

    def test_expired_review_fails_closed(self):
        errors = validate_document(
            Path("docs/MODEL_CANDIDATE_POOL.md"),
            as_of=date(2028, 1, 1),
        )
        self.assertTrue(any("review expired" in error for error in errors))

    def test_missing_review_record_is_rejected(self):
        line = next(line for line in self.document.splitlines() if line.startswith("| DINOv2 with registers | 2026"))
        errors = self.validate_text(self.document.replace(line, "", 1))
        self.assertTrue(
            any("DINOv2 with registers: hard gate must have exactly one review row" in error for error in errors)
        )

    def test_policy_version_drift_is_rejected(self):
        document = self.document.replace("version 1.0.", "version 0.9.", 1)
        errors = self.validate_text(document)
        self.assertTrue(any("Governance policy version" in error for error in errors))

    def test_policy_threshold_change_controls_decision(self):
        policy = copy.deepcopy(load_policy())
        policy["thresholds"]["core"] = 90
        row = CandidateScore(
            model="PolicyControlled",
            dimensions=(25, 20, 15, 10, 10, 10, 10),
            value=100,
            evidence="B",
            risk=5,
            final=85,
            decision="core",
        )
        errors = validate_scores([row], policy)
        self.assertTrue(any("must be 'observation'" in error for error in errors))

    def test_wrong_review_owner_is_rejected(self):
        document = self.document.replace("| maintainers | benchmark, API", "| unknown | benchmark, API", 1)
        errors = self.validate_text(document)
        self.assertTrue(any("owner must be 'maintainers'" in error for error in errors))

    def test_rejected_candidate_cannot_authorize_implementation(self):
        row = CandidateScore(
            model="RejectedImplementation",
            dimensions=(10, 10, 10, 5, 5, 5, 5),
            value=50,
            evidence="A",
            risk=0,
            final=50,
            decision="rejected, implemented",
        )
        errors = validate_scores([row])
        self.assertIn("RejectedImplementation: rejected candidate cannot authorize implementation", errors)

    def test_unknown_decision_qualifier_is_rejected(self):
        row = CandidateScore(
            model="UnknownQualifier",
            dimensions=(25, 20, 15, 10, 10, 10, 10),
            value=100,
            evidence="A",
            risk=0,
            final=100,
            decision="core, maybe",
        )
        errors = validate_scores([row])
        self.assertTrue(any("unsupported decision qualifier" in error for error in errors))

    def test_registry_baseline_matches_current_governed_catalog(self):
        baseline = load_registry_baseline()
        self.assertEqual(len(baseline), 161)
        errors = validate_document(Path("docs/MODEL_CANDIDATE_POOL.md"), registry_baseline=baseline)
        self.assertFalse(any("registry" in error.lower() for error in errors))

    def test_unapproved_registry_addition_is_rejected(self):
        baseline = load_registry_baseline()
        baseline.remove(("graph", "GCN"))
        errors = validate_document(Path("docs/MODEL_CANDIDATE_POOL.md"), registry_baseline=baseline)
        self.assertIn(
            "Ungoverned registry addition graph:GCN; add and approve a candidate-pool record first",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
