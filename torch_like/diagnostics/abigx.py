"""ABIGX-style adversarial fault reconstruction and integrated-gradient explanations.

This is an independent, portable gradient-based implementation of the method
published in IEEE TPAMI 2026. It does not import or copy the author's reference
repository. The caller supplies a differentiable fault score, keeping the
explainer model-agnostic at the diagnostics boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Callable

import torch


ScoreFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ABIGXExplanation:
    """Per-sample attribution and the fault-aware counterfactual baseline."""

    attributions: torch.Tensor
    baseline: torch.Tensor
    original_scores: torch.Tensor
    baseline_scores: torch.Tensor
    completeness_residual: torch.Tensor
    norm: float
    epsilon: float

    def __post_init__(self) -> None:
        tensors = {
            "attributions": self.attributions,
            "baseline": self.baseline,
            "original_scores": self.original_scores,
            "baseline_scores": self.baseline_scores,
            "completeness_residual": self.completeness_residual,
        }
        if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("explanation values must be torch tensors")
        if self.attributions.shape != self.baseline.shape or self.attributions.ndim < 2:
            raise ValueError("attributions and baseline must share a batched input shape")
        batch_size = self.baseline.shape[0]
        for name in ("original_scores", "baseline_scores", "completeness_residual"):
            if tensors[name].shape != (batch_size,):
                raise ValueError(f"{name} must contain one scalar per input sample")
        for name, value in tensors.items():
            if not torch.is_floating_point(value) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating-point values")
        if self.norm not in {0.0, 1.0, 2.0, inf}:
            raise ValueError("norm must be one of 0, 1, 2 or infinity")
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

        for name, value in tensors.items():
            object.__setattr__(self, name, value.detach().clone())

    @property
    def absolute_attributions(self) -> torch.Tensor:
        """Return absolute contributions with the same batch/input shape."""
        return self.attributions.abs()

    @property
    def feature_ranking(self) -> torch.Tensor:
        """Return flattened feature indices ordered by descending absolute contribution."""
        return self.absolute_attributions.flatten(start_dim=1).argsort(dim=1, descending=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary; tensors are converted to CPU lists."""
        return {
            "method": "ABIGX",
            "norm": self.norm,
            "epsilon": self.epsilon,
            "attributions": self.attributions.cpu().tolist(),
            "baseline": self.baseline.cpu().tolist(),
            "original_scores": self.original_scores.cpu().tolist(),
            "baseline_scores": self.baseline_scores.cpu().tolist(),
            "completeness_residual": self.completeness_residual.cpu().tolist(),
            "feature_ranking": self.feature_ranking.cpu().tolist(),
        }


class ABIGXExplainer:
    """Adversarial fault reconstruction followed by integrated gradients.

    The score function must return one differentiable fault score per sample,
    or a tensor whose leading dimension is the batch dimension. Lower scores
    must represent a more normal model state. The function must be batch
    separable: the score for one sample cannot depend on other batch members.

    Supported perturbation constraints are L0 (epsilon is a feature budget),
    L1, L2 and L-infinity. The portable implementation uses projected gradient
    descent for every norm; it does not claim the global optimum of the
    mixed-integer program described for linear models in the paper.
    """

    def __init__(
        self,
        *,
        norm: float = 2,
        epsilon: float = 1.0,
        steps: int = 40,
        step_size: float | None = None,
        ig_steps: int = 32,
    ) -> None:
        if isinstance(norm, bool) or norm not in {0, 1, 2, inf}:
            raise ValueError("norm must be one of 0, 1, 2 or infinity")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not isfinite(epsilon):
            raise ValueError("epsilon must be a finite number")
        if norm == 0:
            if not isinstance(epsilon, int) or epsilon < 1:
                raise ValueError("L0 epsilon must be a positive integer feature budget")
        elif epsilon <= 0:
            raise ValueError("epsilon must be positive for L1, L2 and L-infinity constraints")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(ig_steps, bool) or not isinstance(ig_steps, int) or ig_steps < 2:
            raise ValueError("ig_steps must be an integer of at least two")
        if step_size is not None and (
            isinstance(step_size, bool)
            or not isinstance(step_size, (int, float))
            or not isfinite(step_size)
            or step_size <= 0
        ):
            raise ValueError("step_size must be finite and positive")

        self.norm = float(norm)
        self.epsilon = float(epsilon)
        self.steps = steps
        self.ig_steps = ig_steps
        self.step_size = float(step_size) if step_size is not None else (
            1.0 / steps if norm == 0 else float(epsilon) / steps
        )

    @staticmethod
    def _validate_inputs(inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch tensor")
        if inputs.ndim < 2 or inputs.shape[0] < 1 or any(size < 1 for size in inputs.shape[1:]):
            raise ValueError("inputs must have a nonempty batch dimension and nonempty feature dimensions")
        if not torch.is_floating_point(inputs) or not torch.isfinite(inputs).all():
            raise ValueError("inputs must contain finite floating-point values")
        return inputs.detach()

    @staticmethod
    def _scores(score_fn: ScoreFunction, inputs: torch.Tensor) -> torch.Tensor:
        values = score_fn(inputs)
        if not isinstance(values, torch.Tensor):
            raise TypeError("score_fn must return a torch tensor")
        if values.device != inputs.device or not torch.is_floating_point(values):
            raise ValueError("score_fn output must be floating point and on the input device")
        batch_size = inputs.shape[0]
        if values.ndim == 0:
            if batch_size != 1:
                raise ValueError("a scalar score is valid only for a batch of size one")
            values = values.reshape(1)
        elif values.shape[0] != batch_size:
            raise ValueError("score_fn output must have the same leading batch dimension as inputs")
        else:
            values = values.reshape(batch_size, -1).mean(dim=1)
        if not torch.isfinite(values).all():
            raise ValueError("score_fn returned nonfinite fault scores")
        return values

    def _project(self, delta: torch.Tensor) -> torch.Tensor:
        flat = delta.flatten(start_dim=1)
        if self.norm == 0:
            budget = min(int(self.epsilon), flat.shape[1])
            if budget == flat.shape[1]:
                return delta
            indices = flat.abs().topk(budget, dim=1).indices
            mask = torch.zeros_like(flat)
            mask.scatter_(1, indices, 1.0)
            return (flat * mask).reshape_as(delta)

        if self.norm == inf:
            return flat.clamp(min=-self.epsilon, max=self.epsilon).reshape_as(delta)

        if self.norm == 2:
            lengths = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
            factors = (self.epsilon / lengths.clamp_min(torch.finfo(flat.dtype).eps)).clamp(max=1.0)
            return (flat * factors).reshape_as(delta)

        # Euclidean projection onto the L1 ball (Duchi et al.).
        absolute = flat.abs()
        outside = absolute.sum(dim=1, keepdim=True) > self.epsilon
        ordered = torch.sort(absolute, dim=1, descending=True).values
        cumulative = ordered.cumsum(dim=1) - self.epsilon
        indices = torch.arange(1, flat.shape[1] + 1, device=flat.device, dtype=flat.dtype).unsqueeze(0)
        active = ordered - cumulative / indices > 0
        rho = active.sum(dim=1).clamp_min(1) - 1
        theta = cumulative.gather(1, rho.unsqueeze(1)) / (rho.unsqueeze(1) + 1).to(flat.dtype)
        projected = flat.sign() * (absolute - theta).clamp_min(0)
        result = torch.where(outside, projected, flat)
        return result.reshape_as(delta)

    def _direction(self, gradient: torch.Tensor) -> torch.Tensor:
        flat = gradient.flatten(start_dim=1)
        if self.norm == 0:
            budget = min(int(self.epsilon), flat.shape[1])
            indices = flat.abs().topk(budget, dim=1).indices
            mask = torch.zeros_like(flat)
            mask.scatter_(1, indices, 1.0)
            return (flat.sign() * mask).reshape_as(gradient)
        if self.norm == inf or self.norm == 1:
            return gradient.sign()
        lengths = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
        return (flat / lengths.clamp_min(torch.finfo(flat.dtype).eps)).reshape_as(gradient)

    def _reconstruct(
        self,
        inputs: torch.Tensor,
        score_fn: ScoreFunction,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            original_scores = self._scores(score_fn, inputs)
        best = inputs.clone()
        best_scores = original_scores.clone()
        current = inputs.clone()

        for _ in range(self.steps):
            point = current.detach().requires_grad_(True)
            scores = self._scores(score_fn, point)
            try:
                gradient = torch.autograd.grad(scores.sum(), point, only_inputs=True)[0]
            except RuntimeError as exc:
                raise ValueError("score_fn must be differentiable with respect to its input") from exc
            if not torch.isfinite(gradient).all():
                raise ValueError("score_fn produced nonfinite input gradients")

            candidate = point.detach() - self.step_size * self._direction(gradient.detach())
            candidate = inputs + self._project(candidate - inputs)
            candidate = candidate.detach()
            with torch.no_grad():
                candidate_scores = self._scores(score_fn, candidate)
                improved = candidate_scores < best_scores
                broadcast = improved.reshape((-1,) + (1,) * (inputs.ndim - 1))
                best = torch.where(broadcast, candidate, best)
                best_scores = torch.minimum(best_scores, candidate_scores)
            current = candidate

        return best.detach(), original_scores.detach(), best_scores.detach()

    def explain(self, inputs: torch.Tensor, score_fn: ScoreFunction) -> ABIGXExplanation:
        """Explain a batch using an AFR baseline and straight-line IG integration."""
        if not callable(score_fn):
            raise TypeError("score_fn must be callable")
        values = self._validate_inputs(inputs)
        baseline, original_scores, baseline_scores = self._reconstruct(values, score_fn)
        difference = values - baseline
        accumulated = torch.zeros_like(values)

        with torch.enable_grad():
            for index in range(self.ig_steps):
                alpha = (index + 0.5) / self.ig_steps
                point = (baseline + alpha * difference).detach().requires_grad_(True)
                scores = self._scores(score_fn, point)
                try:
                    gradient = torch.autograd.grad(scores.sum(), point, only_inputs=True)[0]
                except RuntimeError as exc:
                    raise ValueError("score_fn must be differentiable along the IG path") from exc
                if not torch.isfinite(gradient).all():
                    raise ValueError("score_fn produced nonfinite IG gradients")
                accumulated += gradient.detach()

        attributions = difference * (accumulated / self.ig_steps)
        attribution_sum = attributions.flatten(start_dim=1).sum(dim=1)
        completeness_residual = (original_scores - baseline_scores) - attribution_sum
        return ABIGXExplanation(
            attributions=attributions,
            baseline=baseline,
            original_scores=original_scores,
            baseline_scores=baseline_scores,
            completeness_residual=completeness_residual,
            norm=self.norm,
            epsilon=self.epsilon,
        )

    __call__ = explain


def make_classification_spe(
    representation_fn: Callable[[torch.Tensor], torch.Tensor],
    normal_reference: torch.Tensor,
) -> ScoreFunction:
    """Build the paper's classification SPE: distance to normality representation barycenter.

    The representation function should expose the penultimate (or another
    explicitly selected) feature layer. The returned score is the squared
    Euclidean distance to the mean representation of the supplied normal
    reference samples; it is not a fault-class logit.
    """
    if not callable(representation_fn):
        raise TypeError("representation_fn must be callable")
    reference = ABIGXExplainer._validate_inputs(normal_reference)
    with torch.no_grad():
        representations = representation_fn(reference)
        if not isinstance(representations, torch.Tensor) or representations.shape[0] != reference.shape[0]:
            raise ValueError("representation_fn must return a tensor with the reference batch dimension")
        representations = representations.reshape(representations.shape[0], -1)
        if not torch.isfinite(representations).all():
            raise ValueError("normal reference representations must be finite")
        centroid = representations.mean(dim=0).detach()

    def score_fn(inputs: torch.Tensor) -> torch.Tensor:
        current = representation_fn(inputs)
        if not isinstance(current, torch.Tensor) or current.shape[0] != inputs.shape[0]:
            raise ValueError("representation_fn must return a tensor with the input batch dimension")
        current = current.reshape(current.shape[0], -1)
        if current.shape[1] != centroid.numel():
            raise ValueError("representation dimension differs from the normal reference")
        return (current - centroid).square().sum(dim=1)

    return score_fn


__all__ = ["ABIGXExplanation", "ABIGXExplainer", "make_classification_spe"]
