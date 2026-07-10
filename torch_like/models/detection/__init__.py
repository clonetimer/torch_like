"""Detection task interfaces and models."""

from torch_like.models.detection.anomaly_transformer import AnomalyTransformer, AnomalyTransformerConfig
from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel
from torch_like.models.detection.dc_detector import DCDetector, DCDetectorConfig
from torch_like.models.detection.gdn import GDN, GDNConfig
from torch_like.models.detection.memto import MEMTO, MEMTOConfig
from torch_like.models.detection.mtad_gat import MTADGAT, MTADGATConfig
from torch_like.models.detection.rt_detector import RTDetector, RTDetectorConfig
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "MEMTO": {
        "cls": MEMTO,
        "family": "memory_transformer",
        "layer": "Anomaly detection",
        "integration_level": "framework_oriented",
        "supports": ["reconstruction", "memory"],
        "paper": "MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection (NeurIPS 2023), framework-oriented integration",
        "description": "Memory-guided reconstruction transformer with latent memory matching for multivariate anomaly detection",
        "complexity": "O(L^2 + M * D) with Transformer encoding and memory lookup",
    },
    "RTDetector": {
        "cls": RTDetector,
        "family": "transformer_detection",
        "layer": "Anomaly detection",
        "integration_level": "framework_oriented",
        "supports": ["reconstruction", "trend"],
        "paper": "RTdetector: Deep Transformer Networks for Time Series Anomaly Detection Based on Reconstruction Trend (IJCAI 2025)",
        "description": "Reconstruction-trend transformer with de-stationary conditioning for multivariate anomaly detection",
        "complexity": "O(L^2)",
    },
    "GDN": {
        "cls": GDN,
        "family": "graph_deviation",
        "layer": "Anomaly detection",
        "integration_level": "framework_oriented",
        "supports": ["graph_mode", "graph_structure_learning", "reconstruction"],
        "paper": "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series (AAAI 2021)",
        "description": "Learned sensor graph with graph attention and reconstruction-style anomaly scoring",
        "complexity": "O(B * L * C * D + B * C^2 * D) with channels C and hidden width D",
    },
    "DCDetector": {
        "cls": DCDetector,
        "family": "dual_attention_detection",
        "layer": "Anomaly detection",
        "integration_level": "framework_oriented",
        "supports": ["dual_attention", "contrastive_representation", "reconstruction"],
        "paper": "DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection (KDD 2023)",
        "description": "Dual time/channel attention detector with reconstruction and representation-discrepancy anomaly scores",
        "complexity": "O(B * (L^2 + C^2) * D) with window length L and channels C",
    },
    "AnomalyTransformer": {
        "cls": AnomalyTransformer,
        "family": "association_discrepancy",
        "layer": "Anomaly detection",
        "integration_level": "framework_oriented",
        "supports": ["association_discrepancy", "attention", "reconstruction"],
        "paper": "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy (ICLR 2022)",
        "description": "Transformer detector that compares learned series associations with Gaussian prior associations",
        "complexity": "O(B * H * L^2 + B * L * D_ff) with heads H and window length L",
    },
    "MTADGAT": {
        "cls": MTADGAT,
        "family": "graph_attention_detection",
        "layer": "Anomaly detection",
        "integration_level": "framework_oriented",
        "supports": ["feature_attention", "temporal_attention", "forecasting", "reconstruction"],
        "paper": "MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Network (ICDM 2020)",
        "description": "Feature and temporal graph-attention detector with joint forecasting and reconstruction anomaly scores",
        "complexity": "O(B * (C^2 + L^2) * D + B * L * D^2) with channels C and window length L",
    },
}

register_model_catalog("detection", model_zoo)

__all__ = [
    "DetectionConfig",
    "DetectionBatch",
    "DetectionModel",
    "AnomalyTransformerConfig",
    "AnomalyTransformer",
    "DCDetectorConfig",
    "DCDetector",
    "GDNConfig",
    "GDN",
    "MEMTOConfig",
    "MEMTO",
    "MTADGATConfig",
    "MTADGAT",
    "RTDetectorConfig",
    "RTDetector",
    "model_zoo",
]
