"""Diagnosis task interfaces."""

from torch_like.models.diagnosis.common import DiagnosisBatch, DiagnosisConfig, DiagnosisModel

model_zoo = {}

__all__ = [
    "DiagnosisConfig",
    "DiagnosisBatch",
    "DiagnosisModel",
    "model_zoo",
]
