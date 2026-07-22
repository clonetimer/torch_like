"""Internal helpers for consistent user-visible API deprecation warnings."""

from __future__ import annotations

import warnings


def warn_deprecated(
    symbol: str,
    *,
    deprecated_in: str,
    remove_in: str,
    replacement: str | None = None,
    reason: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit the framework-standard warning for one still-supported public symbol."""

    if not symbol or not deprecated_in or not remove_in:
        raise ValueError("symbol, deprecated_in, and remove_in must be non-empty")
    message = f"{symbol} is deprecated since torch_like {deprecated_in} and will be removed in {remove_in}."
    if replacement:
        message += f" Use {replacement} instead."
    if reason:
        message += f" {reason}"
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


__all__ = ["warn_deprecated"]
