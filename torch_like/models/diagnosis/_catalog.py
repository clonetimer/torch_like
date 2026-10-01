"""Diagnosis task interfaces."""

from torch_like.models.diagnosis.common import DiagnosisBatch, DiagnosisConfig, DiagnosisModel
from torch_like.models.diagnosis.inception_time import InceptionTime, InceptionTimeConfig
from torch_like.models.diagnosis.rocket import ROCKET, ROCKETConfig
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "InceptionTime": {
        "cls": InceptionTime,
        "family": "inception_time_series",
        "layer": "Time-series diagnosis / classification",
        "integration_level": "paper_aligned",
        "supports": ["multiscale_convolution", "residual_connection", "time_series_classification"],
        "paper": "InceptionTime: Finding AlexNet for Time Series Classification (DMKD 2020)",
        "description": "Residual multiscale Inception network for time-series classification and diagnosis",
        "complexity": "O(B * L * D * sum(kernel_sizes) * blocks) multiscale temporal convolution",
    },
    "ROCKET": {
        "cls": ROCKET,
        "family": "random_convolution_features",
        "layer": "Time-series diagnosis / classification",
        "integration_level": "framework_oriented",
        "supports": ["random_convolution", "time_series_classification", "fixed_features", "linear_classifier"],
        "paper": "ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels (KDD 2020)",
        "description": "Fixed random convolutional kernel features with trainable linear classifier for diagnosis",
        "complexity": "O(B * kernels * L * input_dim) random convolution feature extraction",
    },
}

register_model_catalog("diagnosis", model_zoo)

__all__ = [
    "DiagnosisConfig",
    "DiagnosisBatch",
    "DiagnosisModel",
    "InceptionTimeConfig",
    "InceptionTime",
    "ROCKETConfig",
    "ROCKET",
    "model_zoo",
]
