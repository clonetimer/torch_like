"""Registry-driven anomaly candidate discovery for leaderboard tasks."""

from __future__ import annotations

from typing import Any

from torch_like.models.detection.common import DetectionModel
from torch_like.models.registry import list_models

ANOMALY_BASELINE_GRIDS: dict[str, list[dict[str, Any]]] = {
    "GlobalSTD": [{}],
    "IsolationForest": [{"n_estimators": 50, "max_samples": 4096}],
}

# Resource-balanced starting points for fair first-pass comparisons. These are
# leaderboard defaults, not claims that one configuration reproduces a paper.
DETECTION_MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "CBAMTCNDetector": [
        {
            "d_model": 32,
            "e_layers": 2,
            "kernel_size": 5,
            "cbam_reduction": 8,
            "cbam_kernel_size": 7,
            "dropout": 0.1,
            "use_cbam": True,
        },
        {
            "d_model": 32,
            "e_layers": 2,
            "kernel_size": 5,
            "cbam_reduction": 8,
            "cbam_kernel_size": 7,
            "dropout": 0.1,
            "use_cbam": False,
        },
    ],
    "TranAD": [{"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64, "dropout": 0.1}],
    "USAD": [{"latent_dim": 32, "hidden_dim": 64, "dropout": 0.1}],
    "OmniAnomaly": [{"d_model": 32, "latent_dim": 16, "hidden_dim": 64, "num_layers": 1, "beta": 0.1}],
    "MEMTO": [
        {
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff": 64,
            "mem_size": 16,
            "latent_dim": 32,
            "dropout": 0.1,
        }
    ],
    "RTDetector": [{"ff_dim": 16, "projector_hidden_dim": 16, "dropout": 0.1}],
    "GDN": [{"d_model": 32, "node_dim": 16, "graph_mode": "learned", "dropout": 0.1}],
    "DCDetector": [{"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64, "discrepancy_weight": 0.1, "dropout": 0.1}],
    "AnomalyTransformer": [
        {"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64, "discrepancy_weight": 0.1, "dropout": 0.1}
    ],
    "MTADGAT": [
        {
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff": 64,
            "forecast_weight": 1.0,
            "recon_weight": 1.0,
            "dropout": 0.1,
        }
    ],
}


def discover_detection_model_grids(*, include_experimental: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Return executable detection models in registry order with audited defaults."""

    discovered: dict[str, list[dict[str, Any]]] = {}
    missing_defaults: list[str] = []
    invalid_contracts: list[str] = []
    for name, metadata in list_models("detection").items():
        model_class = metadata["cls"]
        if not isinstance(model_class, type) or not issubclass(model_class, DetectionModel):
            invalid_contracts.append(name)
            continue
        if name not in DETECTION_MODEL_GRIDS:
            missing_defaults.append(name)
            continue
        if metadata.get("automl_default", True) is False and not include_experimental:
            continue
        discovered[name] = [dict(parameters) for parameters in DETECTION_MODEL_GRIDS[name]]
    if invalid_contracts:
        raise RuntimeError(f"Detection models do not implement DetectionModel: {sorted(invalid_contracts)}")
    if missing_defaults:
        raise RuntimeError(f"Detection models lack AutoML candidate defaults: {sorted(missing_defaults)}")
    return discovered


__all__ = ["ANOMALY_BASELINE_GRIDS", "DETECTION_MODEL_GRIDS", "discover_detection_model_grids"]
