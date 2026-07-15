"""RUL model catalog and registration."""

from torch_like.models.forecasting.rul_adapters import AutoformerRUL, InformerRUL, LogTransRUL
from torch_like.models.registry import register_model_catalog
from torch_like.models.rul_backbones import TCNRUL, PatchTSTRUL, TransformerRUL
from torch_like.models.thgnn import THGNN

model_zoo = {
    "THGNN": {
        "cls": THGNN,
        "family": "graph_temporal_rul",
        "layer": "RUL adapter",
        "integration_level": "framework_oriented",
        "supports": ["graph_mode"],
        "paper": "Temporal and Heterogeneous Graph Neural Network for Remaining Useful Life Prediction (arXiv 2024), inspired implementation",
        "description": "Graph-temporal RUL regressor with learned or prior sensor graph, temporal conv, graph propagation, and FiLM fusion",
        "complexity": "O(T * N^2 + T * N * D) depending on graph mode and sensor count",
    },
    "InformerRUL": {
        "cls": InformerRUL,
        "family": "rul_adapter",
        "layer": "RUL adapter",
        "integration_level": "framework_oriented",
        "supports": [],
        "paper": "Informer adapted for RUL regression (encoder-only + pooling head)",
        "description": "ProbSparse encoder -> AdaptiveAvgPool -> Linear -> scalar RUL",
        "complexity": "O(L log L)",
    },
    "AutoformerRUL": {
        "cls": AutoformerRUL,
        "family": "rul_adapter",
        "layer": "RUL adapter",
        "integration_level": "framework_oriented",
        "supports": [],
        "paper": "Autoformer adapted for RUL regression (encoder-only, trend-decomposition)",
        "description": "AutoCorrelation encoder + SeriesDecomp -> pool -> scalar RUL",
        "complexity": "O(L log L)",
    },
    "LogTransRUL": {
        "cls": LogTransRUL,
        "family": "rul_adapter",
        "layer": "RUL adapter",
        "integration_level": "framework_oriented",
        "supports": [],
        "paper": "LogTrans adapted for RUL regression (encoder-only)",
        "description": "LogSparse encoder -> AdaptiveAvgPool -> Linear -> scalar RUL",
        "complexity": "O(L log L)",
    },
    "TCNRUL": {
        "cls": TCNRUL,
        "family": "rul_adapter",
        "layer": "RUL adapter",
        "integration_level": "framework_oriented",
        "supports": ["dilated_convolution", "loss_step"],
        "paper": "Temporal convolutional network baseline adapted for RUL regression",
        "description": "Dilated residual temporal convolutions, global pooling, and scalar RUL head",
        "complexity": "O(L * D * K * B)",
    },
    "PatchTSTRUL": {
        "cls": PatchTSTRUL,
        "family": "rul_adapter",
        "layer": "RUL adapter",
        "integration_level": "framework_oriented",
        "supports": ["patching", "channel_independent", "loss_step"],
        "paper": "PatchTST backbone adapted for RUL regression",
        "description": "Channel-independent patch encoder with pooled scalar RUL head",
        "complexity": "O(C * P^2 * D)",
    },
    "TransformerRUL": {
        "cls": TransformerRUL,
        "family": "rul_adapter",
        "layer": "RUL adapter",
        "integration_level": "framework_oriented",
        "supports": ["full_attention", "loss_step"],
        "paper": "Encoder-only Transformer baseline adapted for RUL regression",
        "description": "Full-attention temporal encoder with global pooling and scalar RUL head",
        "complexity": "O(L^2 * D)",
    },
}

register_model_catalog("rul", model_zoo)

__all__ = [
    "THGNN",
    "InformerRUL",
    "AutoformerRUL",
    "LogTransRUL",
    "TCNRUL",
    "PatchTSTRUL",
    "TransformerRUL",
    "model_zoo",
]
