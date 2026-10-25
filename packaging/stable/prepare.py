"""Build an isolated, explicit algorithm-only wheel from the maintained sources.

Usage:
    python packaging/stable/prepare.py --destination build/stable

The staged wheel has its own package name (torch_like_stable), so it can coexist
with full torch_like, and contains no orchestration, CLI, dataset audit or
evaluation-platform modules. The source allowlist is the release contract.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STABLE_MODULES = (
    "classical/__init__.py",
    "classical/_registry.py",
    "diagnostics/types.py",
    "diagnostics/decisions.py",
    "diagnostics/streaming.py",
    "diagnostics/score_projection.py",
    "diagnostics/composition.py",
    "diagnostics/rul_interface.py",
    "diagnostics/registry.py",
    "diagnostics/statistical.py",
    "diagnostics/pca.py",
    "diagnostics/features.py",
    "diagnostics/sklearn_adapter.py",
    "diagnostics/classification.py",
    "diagnostics/torch_classifier.py",
    "diagnostics/health.py",
    "diagnostics/fault_sources.py",
    "diagnostics/events.py",
    "models/base.py",
    "models/rul.py",
    "models/rul_backbones.py",
    "models/diagnosis/common.py",
    "models/diagnosis/inception_time.py",
    "models/diagnosis/rocket.py",
)

ROOT_INIT = '''"""Standalone, algorithm-only torch_like stable distribution.

No data provenance audit, experimental workflow, AutoML, CLI or training platform.
"""
from torch_like_stable import classical, diagnostics, models
from torch_like_stable.diagnostics import (
    CUSUMDetector,
    DiagnosticBundleResult,
    DiagnosticDecision,
    DecisionMode,
    DiagnosticParameterSpec,
    StreamingThreshold,
    AnomalyScoreProjection,
    project_anomaly_scores,
    DiagnosticPipeline,
    DiagnosticStage,
    RULRegressor,
    RULForecast,
    create_rul_regressor,
    list_rul_models,
    DiagnosticAlgorithmSpec,
    DiagnosticRegistry,
    DiagnosticResult,
    DiagnosticTask,
    EWMADetector,
    FaultLabelSchema,
    GaussianHealthIndex,
    MahalanobisHealthIndex,
    PCASPEDetector,
    PCAT2Detector,
    SignalBatch,
    SklearnWindowAnomalyDetector,
    SklearnWindowFaultClassifier,
    TorchWindowFaultClassifier,
    ZScoreDetector,
    create_diagnostic_algorithm,
    get_diagnostic_algorithm_info,
    list_diagnostic_algorithms,
    register_diagnostic_algorithm,
)
from torch_like_stable.models import (
    DiagnosisBatch,
    InceptionTime,
    PatchTSTRUL,
    ROCKET,
    RULBatch,
    TCNRUL,
    TransformerRUL,
)

__version__ = "0.3.0"
__all__ = [
    "__version__", "classical", "diagnostics", "models", "SignalBatch",
    "DiagnosticStage", "DiagnosticPipeline", "DiagnosticBundleResult",
    "RULRegressor", "RULForecast", "create_rul_regressor", "list_rul_models",
    "DiagnosticResult", "DiagnosticTask", "DiagnosticAlgorithmSpec",
    "DiagnosticDecision", "DecisionMode", "DiagnosticParameterSpec",
    "StreamingThreshold", "AnomalyScoreProjection", "project_anomaly_scores",
    "DiagnosticRegistry", "create_diagnostic_algorithm",
    "get_diagnostic_algorithm_info", "list_diagnostic_algorithms",
    "register_diagnostic_algorithm", "ZScoreDetector", "CUSUMDetector",
    "EWMADetector", "PCAT2Detector", "PCASPEDetector",
    "SklearnWindowAnomalyDetector", "SklearnWindowFaultClassifier",
    "TorchWindowFaultClassifier", "GaussianHealthIndex",
    "MahalanobisHealthIndex", "FaultLabelSchema", "InceptionTime",
    "ROCKET", "DiagnosisBatch", "RULBatch", "TCNRUL",
    "PatchTSTRUL", "TransformerRUL",
]
'''

DIAGNOSTICS_INIT = '''"""Stable algorithm and numerical interface exports only."""

from torch_like_stable.diagnostics.composition import DiagnosticBundleResult, DiagnosticPipeline, DiagnosticStage
from torch_like_stable.diagnostics.decisions import DecisionMode, DiagnosticDecision
from torch_like_stable.diagnostics.streaming import StreamingThreshold
from torch_like_stable.diagnostics.score_projection import AnomalyScoreProjection, project_anomaly_scores
from torch_like_stable.diagnostics.rul_interface import (
    RULForecast, RULRegressor, create_rul_regressor, list_rul_models,
)
from torch_like_stable.diagnostics.classification import (
    FaultLabelSchema, SklearnWindowFaultClassifier, register_sklearn_fault_classifiers,
)
from torch_like_stable.diagnostics.events import DiagnosticEvent, extract_anomaly_events
from torch_like_stable.diagnostics.fault_sources import (
    FaultSourceEvidenceMapper, FaultSourceRulebook, register_fault_source_mapper,
)
from torch_like_stable.diagnostics.features import (
    Spectrum, fft_amplitude, time_domain_features, welch_psd,
)
from torch_like_stable.diagnostics.health import (
    GaussianHealthIndex, MahalanobisHealthIndex, register_health_index_estimators,
)
from torch_like_stable.diagnostics.pca import (
    PCASPEDetector, PCAT2Detector, register_pca_detectors,
)
from torch_like_stable.diagnostics.registry import (
    DiagnosticRegistry, create_diagnostic_algorithm, get_diagnostic_algorithm_info,
    list_diagnostic_algorithms, register_diagnostic_algorithm,
)
from torch_like_stable.diagnostics.sklearn_adapter import (
    DEFAULT_WINDOW_FEATURES, SklearnWindowAnomalyDetector,
    WindowFeatureTransformer, register_sklearn_window_detectors,
)
from torch_like_stable.diagnostics.statistical import (
    CUSUMDetector, EWMADetector, ZScoreDetector, register_statistical_detectors,
)
from torch_like_stable.diagnostics.torch_classifier import (
    TorchWindowFaultClassifier, register_torch_fault_classifiers,
)
from torch_like_stable.diagnostics.types import (
    DiagnosticAlgorithmSpec, DiagnosticOutputKind, DiagnosticParameterSpec, DiagnosticResult,
    DiagnosticTask, SignalBatch,
)

__all__ = [
    "DiagnosticBundleResult", "DiagnosticPipeline", "DiagnosticStage",
    "RULForecast", "RULRegressor", "create_rul_regressor", "list_rul_models",
    "FaultLabelSchema", "SklearnWindowFaultClassifier",
    "register_sklearn_fault_classifiers", "DiagnosticEvent",
    "extract_anomaly_events", "FaultSourceEvidenceMapper",
    "FaultSourceRulebook", "register_fault_source_mapper",
    "Spectrum", "fft_amplitude", "time_domain_features", "welch_psd",
    "GaussianHealthIndex", "MahalanobisHealthIndex",
    "register_health_index_estimators", "PCASPEDetector", "PCAT2Detector",
    "register_pca_detectors", "DiagnosticRegistry",
    "create_diagnostic_algorithm", "get_diagnostic_algorithm_info",
    "list_diagnostic_algorithms", "register_diagnostic_algorithm",
    "DEFAULT_WINDOW_FEATURES", "SklearnWindowAnomalyDetector",
    "WindowFeatureTransformer", "register_sklearn_window_detectors",
    "CUSUMDetector", "EWMADetector", "ZScoreDetector",
    "register_statistical_detectors", "TorchWindowFaultClassifier",
    "register_torch_fault_classifiers", "DiagnosticAlgorithmSpec",
    "DiagnosticOutputKind", "DiagnosticParameterSpec", "DiagnosticResult", "DiagnosticTask", "SignalBatch",
    "DecisionMode", "DiagnosticDecision", "StreamingThreshold",
    "AnomalyScoreProjection", "project_anomaly_scores",
]
'''

MODELS_INIT = '''"""Only stable fault-classification and directly callable RUL backbones."""

from torch_like_stable.models.base import BaseModel
from torch_like_stable.models.diagnosis import (
    DiagnosisBatch, DiagnosisConfig, DiagnosisModel, InceptionTime, ROCKET,
)
from torch_like_stable.models.rul import RULBatch, RULConfig, RULModel
from torch_like_stable.models.rul_backbones import (
    PatchTSTRUL, TCNRUL, TransformerRUL,
)

__all__ = [
    "BaseModel", "DiagnosisBatch", "DiagnosisConfig", "DiagnosisModel",
    "InceptionTime", "ROCKET", "RULBatch", "RULConfig", "RULModel",
    "PatchTSTRUL", "TCNRUL", "TransformerRUL",
]
'''

DIAGNOSIS_INIT = '''"""Native PyTorch time-series fault classifiers."""

from torch_like_stable.models.diagnosis.common import (
    DiagnosisBatch, DiagnosisConfig, DiagnosisModel,
)
from torch_like_stable.models.diagnosis.inception_time import InceptionTime
from torch_like_stable.models.diagnosis.rocket import ROCKET

__all__ = [
    "DiagnosisBatch", "DiagnosisConfig", "DiagnosisModel", "InceptionTime", "ROCKET",
]
'''

PROJECT = """[build-system]
requires = ["setuptools>=77.0"]
build-backend = "setuptools.build_meta"

[project]
name = "torch-like-algorithms"
version = "0.3.0"
description = "Standalone stable diagnostic, health-index and RUL algorithms; no data audit or orchestration"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
dependencies = ["numpy>=1.20", "scikit-learn>=1.0", "torch>=2.0"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["torch_like_stable*"]
"""

FORBIDDEN = (
    "torch_like_stable/automl",
    "torch_like_stable/training",
    "torch_like_stable/evaluation",
    "torch_like_stable/serialization",
    "torch_like_stable/runtime",
    "torch_like_stable/data",
    "torch_like_stable/runner",
    "torch_like_stable/native",
    "torch_like_stable/diagnostics/workflow.py",
    "torch_like_stable/diagnostics/generalization.py",
    "torch_like_stable/diagnostics/selection.py",
    "torch_like_stable/diagnostics/uncertainty.py",
    "torch_like_stable/models/forecasting",
    "torch_like_stable/models/vision",
)


def prepare(destination: Path) -> dict[str, object]:
    """Stage a clean, deterministic package tree; never mutate source code."""
    destination = destination.resolve()
    if destination == REPO or REPO in destination.parents and destination.name == "torch_like":
        raise ValueError("destination must be a dedicated packaging stage, not the source package")
    if destination.exists():
        shutil.rmtree(destination)
    pkg = destination / "src" / "torch_like_stable"
    pkg.mkdir(parents=True)
    for relative in STABLE_MODULES:
        source = REPO / "torch_like" / relative
        if not source.is_file():
            raise FileNotFoundError(f"stable source module missing: {relative}")
        target = pkg / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original = source.read_text(encoding="utf-8")
        # The copied implementation stays byte-equivalent apart from namespace rebasing.
        text = original.replace("torch_like.", "torch_like_stable.")
        target.write_text(text, encoding="utf-8")
    for relative, body in (
        ("__init__.py", ROOT_INIT),
        ("diagnostics/__init__.py", DIAGNOSTICS_INIT),
        ("models/__init__.py", MODELS_INIT),
        ("models/diagnosis/__init__.py", DIAGNOSIS_INIT),
    ):
        target = pkg / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    (destination / "pyproject.toml").write_text(PROJECT, encoding="utf-8")
    (destination / "README.md").write_text(
        (REPO / "packaging/stable/README.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (destination / "LICENSE").write_bytes((REPO / "LICENSE").read_bytes())
    file_set = {p.relative_to(destination).as_posix() for p in destination.rglob("*.py")}
    for path in FORBIDDEN:
        if any(x == "src/" + path or x.startswith("src/" + path + "/") for x in file_set):
            raise ValueError(f"non-algorithm module leaked into stable package: {path}")
    manifest = {
        "distribution": "torch-like-algorithms",
        "version": "0.3.0",
        "import_package": "torch_like_stable",
        "source_modules": list(STABLE_MODULES),
        "python_files": sorted(file_set),
        "scope": "algorithms_and_interfaces_only",
        "dataset_source_audit": False,
        "automl_included": False,
        "runner_included": False,
        "evaluation_platform_included": False,
    }
    (destination / "stable-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("build/stable"))
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.destination), indent=2))
