"""Industrial diagnostics task contracts and backend-preserving algorithm registry."""

from torch_like.diagnostics.registry import (
    DiagnosticRegistry,
    create_diagnostic_algorithm,
    get_diagnostic_algorithm_info,
    list_diagnostic_algorithms,
    register_diagnostic_algorithm,
)
from torch_like.diagnostics.types import (
    DiagnosticAlgorithmSpec,
    DiagnosticOutputKind,
    DiagnosticResult,
    DiagnosticTask,
    SignalBatch,
)

__all__ = [
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
