"""Industrial diagnostics task contracts and backend-preserving algorithm registry."""

from torch_like.diagnostics.features import Spectrum, fft_amplitude, time_domain_features, welch_psd
from torch_like.diagnostics.pca import PCASPEDetector, PCAT2Detector, register_pca_detectors
from torch_like.diagnostics.registry import (
    DiagnosticRegistry,
    create_diagnostic_algorithm,
    get_diagnostic_algorithm_info,
    list_diagnostic_algorithms,
    register_diagnostic_algorithm,
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
from torch_like.diagnostics.types import (
    DiagnosticAlgorithmSpec,
    DiagnosticOutputKind,
    DiagnosticResult,
    DiagnosticTask,
    SignalBatch,
)

__all__ = [
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
