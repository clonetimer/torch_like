"""ABIGX adversarial-reconstruction and integrated-gradient tests."""

import unittest

import torch

from torch_like.diagnostics.abigx import ABIGXExplainer, ABIGXExplanation, make_classification_spe


class ABIGXExplainerTests(unittest.TestCase):
    def test_l2_reconstruction_reduces_fault_score_within_budget(self):
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        score = lambda x: x.square().sum(dim=1)
        explanation = ABIGXExplainer(norm=2, epsilon=0.5, steps=20, step_size=0.05, ig_steps=32).explain(
            inputs, score
        )

        self.assertIsInstance(explanation, ABIGXExplanation)
        self.assertTrue(torch.all(explanation.baseline_scores < explanation.original_scores))
        delta = (explanation.baseline - inputs).flatten(start_dim=1)
        self.assertTrue(torch.all(torch.linalg.vector_norm(delta, ord=2, dim=1) <= 0.500001))
        self.assertEqual(explanation.attributions.shape, inputs.shape)
        self.assertTrue(torch.isfinite(explanation.completeness_residual).all())

    def test_linf_l1_and_l0_constraints_are_projected_per_sample(self):
        inputs = torch.tensor([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
        score = lambda x: x.sum(dim=1)

        linf = ABIGXExplainer(norm=float("inf"), epsilon=0.25, steps=5, step_size=0.1).explain(inputs, score)
        self.assertLessEqual(float((linf.baseline - inputs).abs().max()), 0.250001)

        l1 = ABIGXExplainer(norm=1, epsilon=0.5, steps=5, step_size=0.1).explain(inputs, score)
        self.assertTrue(torch.all((l1.baseline - inputs).abs().sum(dim=1) <= 0.500001))

        l0 = ABIGXExplainer(norm=0, epsilon=1, steps=5, step_size=0.1).explain(inputs, score)
        changed = (l0.baseline - inputs).abs() > 1e-8
        self.assertTrue(torch.all(changed.sum(dim=1) <= 1))

    def test_integrated_gradients_completeness_for_linear_fault_score(self):
        inputs = torch.tensor([[1.0, -2.0, 3.0], [2.0, 1.0, -1.0]], dtype=torch.float64)
        weights = torch.tensor([0.5, -1.5, 2.0], dtype=torch.float64)
        score = lambda x: (x * weights).sum(dim=1)
        explanation = ABIGXExplainer(norm=2, epsilon=0.4, steps=8, step_size=0.05, ig_steps=20).explain(
            inputs, score
        )

        torch.testing.assert_close(
            explanation.completeness_residual,
            torch.zeros_like(explanation.completeness_residual),
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(
            explanation.attributions.flatten(start_dim=1).sum(dim=1),
            explanation.original_scores - explanation.baseline_scores,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_classification_spe_uses_normal_representation_centroid(self):
        def representation(x):
            return torch.stack((x[:, 0] + 2.0 * x[:, 1], x[:, 1] - x[:, 0]), dim=1)

        normal = torch.tensor([[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0]])
        score = make_classification_spe(representation, normal)
        values = score(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(torch.all(values > 0))
        explanation = ABIGXExplainer(norm=2, epsilon=0.25, steps=8, step_size=0.03, ig_steps=8).explain(
            torch.tensor([[1.0, 0.0]]), score
        )
        self.assertLess(float(explanation.baseline_scores[0]), float(explanation.original_scores[0]))

    def test_zero_gradient_keeps_the_original_sample_as_baseline(self):
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        explanation = ABIGXExplainer(norm=2, epsilon=1.0, steps=3).explain(
            inputs, lambda x: x.sum(dim=1) * 0.0 + 1.0
        )
        torch.testing.assert_close(explanation.baseline, inputs)
        torch.testing.assert_close(explanation.attributions, torch.zeros_like(inputs))

    def test_score_function_must_be_differentiable_and_batch_aligned(self):
        inputs = torch.ones((2, 3))
        explainer = ABIGXExplainer()
        with self.assertRaisesRegex(ValueError, "differentiable"):
            explainer.explain(inputs, lambda x: x.detach().sum(dim=1))
        with self.assertRaisesRegex(ValueError, "leading batch dimension"):
            explainer.explain(inputs, lambda x: torch.ones((3,), device=x.device))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            explainer.explain(inputs, lambda x: torch.full((x.shape[0],), float("nan"), device=x.device))

    def test_invalid_inputs_and_configuration_are_rejected(self):
        for kwargs in (
            {"norm": 3},
            {"norm": 0, "epsilon": 0},
            {"norm": 0, "epsilon": 1.5},
            {"norm": 2, "epsilon": 0},
            {"steps": 0},
            {"ig_steps": 1},
            {"step_size": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ABIGXExplainer(**kwargs)

        with self.assertRaisesRegex(ValueError, "finite floating-point"):
            ABIGXExplainer().explain(torch.tensor([[float("nan")]]), lambda x: x.sum(dim=1))
        with self.assertRaisesRegex(ValueError, "nonempty feature"):
            ABIGXExplainer().explain(torch.empty((1, 0)), lambda x: x.sum(dim=1))

    def test_explanation_summary_is_json_compatible(self):
        inputs = torch.tensor([[1.0, 2.0]])
        explanation = ABIGXExplainer(norm=float("inf"), epsilon=0.2, steps=2, ig_steps=2).explain(
            inputs, lambda x: x.square().sum(dim=1)
        )
        summary = explanation.to_dict()
        self.assertEqual(summary["method"], "ABIGX")
        self.assertEqual(summary["norm"], "inf")
        self.assertEqual(len(summary["feature_ranking"]), 1)
        self.assertEqual(len(summary["attributions"]), 1)


if __name__ == "__main__":
    unittest.main()
