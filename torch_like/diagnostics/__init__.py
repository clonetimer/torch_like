"""Industrial diagnostics task contracts and backend-preserving algorithm registry."""

from torch_like.diagnostics.classification import (
    FaultLabelSchema,
    SklearnWindowFaultClassifier,
    register_sklearn_fault_classifiers,
)
from torch_like.diagnostics.classification_workflow import (
    FaultClassificationEvaluation,
    FaultClassificationPartition,
    FaultClassificationSplit,
    FaultClassificationWorkflow,
)
from torch_like.diagnostics.composition import DiagnosticBundleResult, DiagnosticPipeline, DiagnosticStage
from torch_like.diagnostics.decisions import DecisionMode, DiagnosticDecision
from torch_like.diagnostics.events import DiagnosticEvent, extract_anomaly_events
from torch_like.diagnostics.fault_sources import (
    FaultSourceEvidenceMapper,
    FaultSourceRulebook,
    register_fault_source_mapper,
)
from torch_like.diagnostics.features import Spectrum, fft_amplitude, time_domain_features, welch_psd
from torch_like.diagnostics.generalization import (
    FaultDomainMetrics,
    FaultDomainPartition,
    FaultGeneralizationEvaluation,
    FaultGeneralizationSplit,
    FaultGeneralizationWorkflow,
)
from torch_like.diagnostics.health import (
    GaussianHealthIndex,
    MahalanobisHealthIndex,
    register_health_index_estimators,
)
from torch_like.diagnostics.pca import PCASPEDetector, PCAT2Detector, register_pca_detectors
from torch_like.diagnostics.registry import (
    DiagnosticRegistry,
    create_diagnostic_algorithm,
    get_diagnostic_algorithm_info,
    list_diagnostic_algorithms,
    register_diagnostic_algorithm,
)
from torch_like.diagnostics.rul_interface import (
    RULForecast,
    RULRegressor,
    create_rul_regressor,
    list_rul_models,
)
from torch_like.diagnostics.score_projection import AnomalyScoreProjection, project_anomaly_scores
from torch_like.diagnostics.selection import (
    FaultCandidateRanking,
    FaultCandidateSpec,
    FaultSeedResult,
    FaultSelectionEvaluation,
    FaultSelectionReport,
    FaultSelectionWorkflow,
)
from torch_like.diagnostics.sklearn_adapter import (
    DEFAULT_WINDOW_FEATURES,
    SklearnWindowAnomalyDetector,
    WindowFeatureTransformer,
    register_sklearn_window_detectors,
)
from torch_like.diagnostics.statistical import (
    CUSUMDetector,
    EWMADetector,
    ZScoreDetector,
    register_statistical_detectors,
)
from torch_like.diagnostics.streaming import StreamingThreshold
from torch_like.diagnostics.torch_classifier import TorchWindowFaultClassifier, register_torch_fault_classifiers
from torch_like.diagnostics.types import (
    DiagnosticAlgorithmSpec,
    DiagnosticOutputKind,
    DiagnosticParameterSpec,
    DiagnosticResult,
    DiagnosticTask,
    SignalBatch,
)
from torch_like.diagnostics.uncertainty import (
    FaultClusterUncertainty,
    FaultMetricInterval,
    bootstrap_fault_generalization,
)
from torch_like.diagnostics.workflow import (
    AnomalyDiagnosticWorkflow,
    AnomalyEvaluation,
    AnomalyPartition,
    AnomalySplit,
)

__all__ = [
    "AnomalyScoreProjection",
    "project_anomaly_scores",
    "StreamingThreshold",
    "DecisionMode",
    "DiagnosticDecision",
    "DiagnosticBundleResult",
    "DiagnosticPipeline",
    "DiagnosticStage",
    "RULForecast",
    "RULRegressor",
    "create_rul_regressor",
    "list_rul_models",
    "GaussianHealthIndex",
    "MahalanobisHealthIndex",
    "register_health_index_estimators",
    "FaultClusterUncertainty",
    "FaultMetricInterval",
    "bootstrap_fault_generalization",
    "FaultCandidateRanking",
    "FaultCandidateSpec",
    "FaultSeedResult",
    "FaultSelectionEvaluation",
    "FaultSelectionReport",
    "FaultSelectionWorkflow",
    "TorchWindowFaultClassifier",
    "register_torch_fault_classifiers",
    "FaultDomainMetrics",
    "FaultDomainPartition",
    "FaultGeneralizationEvaluation",
    "FaultGeneralizationSplit",
    "FaultGeneralizationWorkflow",
    "FaultSourceEvidenceMapper",
    "FaultSourceRulebook",
    "register_fault_source_mapper",
    "FaultLabelSchema",
    "SklearnWindowFaultClassifier",
    "register_sklearn_fault_classifiers",
    "FaultClassificationEvaluation",
    "FaultClassificationPartition",
    "FaultClassificationSplit",
    "FaultClassificationWorkflow",
    "DiagnosticEvent",
    "extract_anomaly_events",
    "AnomalyDiagnosticWorkflow",
    "AnomalyEvaluation",
    "AnomalyPartition",
    "AnomalySplit",
    "DEFAULT_WINDOW_FEATURES",
    "SklearnWindowAnomalyDetector",
    "WindowFeatureTransformer",
    "register_sklearn_window_detectors",
    "CUSUMDetector",
    "PCASPEDetector",
    "PCAT2Detector",
    "Spectrum",
    "fft_amplitude",
    "time_domain_features",
    "welch_psd",
    "register_pca_detectors",
    "EWMADetector",
    "ZScoreDetector",
    "register_statistical_detectors",
    "DiagnosticAlgorithmSpec",
    "DiagnosticParameterSpec",
    "DiagnosticOutputKind",
    "DiagnosticRegistry",
    "DiagnosticResult",
    "DiagnosticTask",
    "SignalBatch",
    "create_diagnostic_algorithm",
    "get_diagnostic_algorithm_info",
    "list_diagnostic_algorithms",
    "register_diagnostic_algorithm",
]
