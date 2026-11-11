# Diagnostics R8: ABIGX fault explanations

Date: 2026-11-11

Status: portable diagnostic explainer implementation; no model-registry entry and no benchmark-performance claim.

## Literature basis

Yue Zhuo, Jinchuan Qian, Junhua Zheng, Xiaoyu Jiang, Zhihuan Song, Duxin Chen, Wenwu Yu and Zhiqiang Ge, “ABIGX: A Unified Framework for Explainable Fault Detection and Classification”, IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(9), 10906–10923, 2026. DOI: 10.1109/TPAMI.2026.3689327.

The paper introduces Adversarial Fault Reconstruction (AFR) as an adaptive, model-induced baseline for Integrated Gradients. For fault detection, the fault score can be a detector's reconstruction/projection error. For classification, the paper defines a squared-distance score from the penultimate representation to the centroid of normal-reference representations.

## Public API

```python
import torch
from torch_like.diagnostics import ABIGXExplainer, make_classification_spe

# Detection: lower scores must mean a more normal model state.
def detector_score(x: torch.Tensor) -> torch.Tensor:
    reconstruction = detector(x)
    return (reconstruction - x).square().flatten(1).mean(dim=1)

explainer = ABIGXExplainer(norm=2, epsilon=0.5, steps=40, ig_steps=32)
explanation = explainer.explain(fault_windows, detector_score)

# Classification: expose a selected representation layer and a normal reference.
classification_score = make_classification_spe(
    representation_fn=penultimate_features,
    normal_reference=normal_windows,
)
# The attribution target is the scalar confidence of the explained fault class.
def fault_confidence(x: torch.Tensor) -> torch.Tensor:
    return classifier(x).softmax(dim=-1)[:, explained_class]

explanation = explainer.explain(
    fault_windows,
    classification_score,
    attribution_fn=fault_confidence,
)
```

The AFR score function and IG attribution function must be differentiable, batch-separable, and return one scalar per sample (or a tensor with the batch dimension first). For fault detection they can be the same anomaly score. For fault classification, use normal-centroid SPE for AFR and a separate scalar output such as the explained fault-class confidence for IG. The caller controls model evaluation mode, preprocessing, valid input ranges and the choice of representation layer.

## Algorithm

1. **Fault-aware baseline (AFR)**: initialise the reconstructed sample at the explained input. Iteratively descend the fault score and project the perturbation onto a caller-selected L0, L1, L2 or L-infinity budget. The best score-reducing iterate is retained.
2. **Attribution (IG)**: integrate gradients of the FDC output (for classification, the selected fault confidence) along the straight path from the AFR baseline to the original sample. Multiply the average path gradient by the input-baseline difference. The AFR objective and IG attribution target are separate inputs; they are the same score only when the detector output itself is the desired explanation target.
3. **Output**: return per-feature signed attributions, AFR baseline, original/baseline scores and an Integrated-Gradients completeness residual.

The implementation follows the minimisation direction in the paper's AFR objective and Eq. (20). The paper's Algorithm 1 pseudocode prints an additive gradient update, which conflicts with the stated minimisation objective and the subtractive update in Eq. (20); this implementation deliberately uses gradient descent and tests that the score decreases.

## Implementation boundary

- This is an independent PyTorch implementation from the published method description; the reference repository was not copied. Its repository does not declare an explicit software license.
- The portable path uses projected gradient descent for all supported norms. It does not implement the mixed-integer/global optimisation option described for linear models and does not claim to find the global AFR optimum.
- Classification SPE uses a caller-selected representation function and a normal-reference centroid. It is not a class-logit explanation and does not explain why one fault class was selected over another.
- The explainer requires gradients. It is not compatible with non-differentiable tree ensembles or classical Bayesian networks without a separately validated gradient approximation.
- AFR is a model-induced normal-like baseline, not a guarantee of membership in the physical normal-data manifold. Valid-domain constraints or physics priors may be supplied by wrapping the score/model; they are not inferred automatically.
- Attributions are associational explanations of model behaviour, not certified causal root causes. They should narrow candidate fault sources, not replace physical verification.
- Iterative AFR and IG require multiple model evaluations; this implementation is intended for offline explanation and research analysis, not a real-time onboard guarantee.

## Verification

`tests/test_diagnostics_abigx.py` covers:

- L2, L1, L-infinity and L0 perturbation budgets.
- Fault-score reduction and zero-gradient behaviour.
- Integrated-Gradients completeness on a linear score oracle.
- Normal-reference representation-centroid classification score.
- Batch alignment, differentiability, finite values and invalid configuration boundaries.

These tests establish the API and numerical contracts only. No TEP, MVTec-AD or user telemetry benchmark has been reproduced by this repository.
