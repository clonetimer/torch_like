"""Opt-in ctypes adapter for trusted v1 native anomaly-score plugins.

Does not import native libraries on package import. Only explicitly supplied
local libraries are loaded; DLL/SO loading executes arbitrary native code.
"""

from __future__ import annotations

import base64
import ctypes
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from torch_like.diagnostics.types import DiagnosticResult, SignalBatch

ABI_VERSION = 1


class _NativeInput(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("windows", ctypes.c_uint64),
        ("time_steps", ctypes.c_uint64),
        ("channels", ctypes.c_uint64),
        ("values", ctypes.POINTER(ctypes.c_double)),
    ]


class _NativeOutput(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("capacity", ctypes.c_uint64),
        ("scores", ctypes.POINTER(ctypes.c_double)),
    ]


class NativeAnomalyDetector:
    """A fitted Python-native bridge that preserves SignalBatch metadata.

    ABI v1 promises scores shaped (N,L,C); there is no implicit threshold,
    labels, RUL value, feature inference, interpolation, or native binary trust
    verification. Explicitly invoked local libraries must be trusted.
    """

    def __init__(
        self,
        library_path: str | Path,
        *,
        algorithm_name: str,
        config: Mapping[str, Any] | None = None,
        max_state_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if not isinstance(algorithm_name, str) or not algorithm_name.strip():
            raise ValueError("algorithm_name must be nonempty")
        if isinstance(max_state_bytes, bool) or not isinstance(max_state_bytes, int) or max_state_bytes < 1:
            raise ValueError("max_state_bytes must be positive integer")
        path = Path(library_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("library_path must point to a local shared library file")
        config_dict = dict(config or {})
        config_json = json.dumps(config_dict, sort_keys=True, allow_nan=False, ensure_ascii=False)
        self.library_path = path
        self.algorithm_name = algorithm_name
        self._config_json = config_json
        self._max_state_bytes = max_state_bytes
        self._library = ctypes.CDLL(str(path))
        self._bind()
        if self._library.tl_native_abi_version() != ABI_VERSION:
            raise ValueError("native plugin ABI version mismatch")
        handle = ctypes.c_void_p()
        self._check(self._library.tl_native_create(config_json.encode("utf-8"), ctypes.byref(handle)), "create")
        if not handle.value:
            raise RuntimeError("native create returned null handle")
        self._handle = handle
        self._reference: dict[str, Any] | None = None

    def _bind(self) -> None:
        signatures = {
            "tl_native_abi_version": (ctypes.c_uint32, ()),
            "tl_native_create": (ctypes.c_int32, (ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p))),
            "tl_native_fit": (ctypes.c_int32, (ctypes.c_void_p, ctypes.POINTER(_NativeInput))),
            "tl_native_score": (
                ctypes.c_int32,
                (ctypes.c_void_p, ctypes.POINTER(_NativeInput), ctypes.POINTER(_NativeOutput)),
            ),
            "tl_native_state_size": (ctypes.c_int32, (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64))),
            "tl_native_save_state": (
                ctypes.c_int32,
                (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64),
            ),
            "tl_native_load_state": (
                ctypes.c_int32,
                (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64),
            ),
            "tl_native_destroy": (None, (ctypes.c_void_p,)),
        }
        for name, (return_type, arguments) in signatures.items():
            try:
                function = getattr(self._library, name)
            except AttributeError as exc:
                raise ValueError(f"native plugin missing required symbol: {name}") from exc
            function.argtypes = arguments
            function.restype = return_type

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"native plugin {operation} failed (status={status})")

    @staticmethod
    def _data(batch: SignalBatch) -> tuple[np.ndarray, _NativeInput]:
        if not isinstance(batch, SignalBatch):
            raise TypeError("native plugin requires SignalBatch")
        if not np.isfinite(batch.values).all():
            raise ValueError("native plugin requires finite signals")
        values = np.ascontiguousarray(batch.values, dtype=np.float64)
        n, length, channels = values.shape
        native = _NativeInput(
            ABI_VERSION, 0, n, length, channels, values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return values, native

    def _active(self) -> None:
        if getattr(self, "_handle", None) is None:
            raise RuntimeError("native detector has been closed")

    @staticmethod
    def _metadata(batch: SignalBatch) -> dict[str, Any]:
        return {
            "channel_names": list(batch.channel_names),
            "units": None if batch.units is None else list(batch.units),
            "sample_rate_hz": batch.sample_rate_hz,
        }

    def fit(self, reference: SignalBatch) -> NativeAnomalyDetector:
        self._active()
        array, native = self._data(reference)
        self._check(self._library.tl_native_fit(self._handle, ctypes.byref(native)), "fit")
        self._reference = self._metadata(reference)
        del array
        return self

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        self._active()
        if self._reference is None:
            raise RuntimeError("fit or restore a trusted native state before scoring")
        if self._metadata(batch) != self._reference:
            raise ValueError("native inference metadata differ from the fitted reference")
        array, native = self._data(batch)
        scores = np.empty_like(array)
        output = _NativeOutput(ABI_VERSION, 0, scores.size, scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self._check(self._library.tl_native_score(self._handle, ctypes.byref(native), ctypes.byref(output)), "score")
        return DiagnosticResult("anomaly_detection", "anomaly_score", scores, self.algorithm_name)

    def export_state(self) -> dict[str, Any]:
        """JSON-safe host manifest; native bytes are opaque and plugin-defined."""
        self._active()
        if self._reference is None:
            raise RuntimeError("native detector must be fitted before export")
        count = ctypes.c_uint64()
        self._check(self._library.tl_native_state_size(self._handle, ctypes.byref(count)), "state_size")
        if count.value < 1 or count.value > self._max_state_bytes:
            raise ValueError("native plugin state size exceeds accepted bounds")
        data = (ctypes.c_uint8 * count.value)()
        self._check(self._library.tl_native_save_state(self._handle, data, count.value), "save_state")
        return {
            "abi_version": ABI_VERSION,
            "algorithm_name": self.algorithm_name,
            "config_json": self._config_json,
            "reference": dict(self._reference),
            "plugin_state_base64": base64.b64encode(bytes(data)).decode("ascii"),
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Restore caller-trusted state. Plugin must implement transactional load."""
        self._active()
        if not isinstance(state, Mapping) or set(state) != {
            "abi_version",
            "algorithm_name",
            "config_json",
            "reference",
            "plugin_state_base64",
        }:
            raise ValueError("invalid native checkpoint manifest")
        if type(state["abi_version"]) is not int or state["abi_version"] != ABI_VERSION:
            raise ValueError("native checkpoint ABI version mismatch")
        if state["algorithm_name"] != self.algorithm_name or state["config_json"] != self._config_json:
            raise ValueError("native checkpoint algorithm/configuration mismatch")
        meta = state["reference"]
        if not isinstance(meta, dict):
            raise ValueError("invalid native checkpoint reference metadata")
        if set(meta) != {"channel_names", "units", "sample_rate_hz"}:
            raise ValueError("invalid native checkpoint metadata fields")
        names = meta["channel_names"]
        units = meta["units"]
        rate = meta["sample_rate_hz"]
        if (
            not isinstance(names, list)
            or not names
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or (units is not None and (not isinstance(units, list) or len(units) != len(names)))
            or isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not np.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError("invalid native checkpoint channel metadata")
        try:
            binary = base64.b64decode(state["plugin_state_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid native checkpoint bytes") from exc
        if not binary or len(binary) > self._max_state_bytes:
            raise ValueError("native checkpoint bytes exceed accepted bounds")
        data = (ctypes.c_uint8 * len(binary)).from_buffer_copy(binary)
        self._check(self._library.tl_native_load_state(self._handle, data, len(binary)), "load_state")
        self._reference = {
            "channel_names": list(names),
            "units": None if units is None else list(units),
            "sample_rate_hz": float(rate),
        }

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            self._library.tl_native_destroy(handle)

    def __enter__(self) -> NativeAnomalyDetector:
        self._active()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ABI_VERSION", "NativeAnomalyDetector"]
