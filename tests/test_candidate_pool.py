import tempfile
import unittest
from pathlib import Path

from tools.validate_candidate_pool import (
    CandidateScore,
    parse_hard_gates,
    parse_scorecard,
    validate_document,
    validate_scores,
)


class CandidatePoolValidationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
