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
from torch_like.diagnostics.pca import PCASPEDetector, PCAT2Detector, register_pca_detectors
from torch_like.diagnostics.registry import (
    DiagnosticRegistry,
    create_diagnostic_algorithm,
    get_diagnostic_algorithm_info,
    list_diagnostic_algorithms,
    register_diagnostic_algorithm,
)
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
from torch_like.diagnostics.torch_classifier import TorchWindowFaultClassifier, register_torch_fault_classifiers
from torch_like.diagnostics.types import (
    DiagnosticAlgorithmSpec,
    DiagnosticOutputKind,
    DiagnosticResult,
    DiagnosticTask,
    SignalBatch,
)
from torch_like.diagnostics.workflow import (
    AnomalyDiagnosticWorkflow,
    AnomalyEvaluation,
    AnomalyPartition,
    AnomalySplit,
)

__all__ = [
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
