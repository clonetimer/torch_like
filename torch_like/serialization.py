"""Stable model persistence for torch_like and curated sklearn estimators."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch_like.contracts import BackendKind, get_model_spec
from torch_like.models.registry import get_model_class

_SCHEMA_VERSION = 1
_TYPE_TAG = "__torch_like_type__"
_MANIFEST = "model.json"
_WEIGHTS = "weights.pt"
_SKLEARN_MODEL = "estimator.joblib"


def _encode(value: Any, *, key_path: str = "config") -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _encode(value.value, key_path=key_path)
    if isinstance(value, tuple):
        return {_TYPE_TAG: "tuple", "items": [_encode(item, key_path=f"{key_path}[]") for item in value]}
    if isinstance(value, list):
        return [_encode(item, key_path=f"{key_path}[]") for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{key_path} contains a non-string mapping key")
        return {key: _encode(item, key_path=f"{key_path}.{key}") for key, item in value.items()}
    if isinstance(value, Path):
        return {_TYPE_TAG: "path", "value": str(value)}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return {
            _TYPE_TAG: "torch_tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "data": tensor.tolist(),
        }
    if isinstance(value, torch.dtype):
        return {_TYPE_TAG: "torch_dtype", "value": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {_TYPE_TAG: "torch_device", "value": str(value)}
    if isinstance(value, np.ndarray):
        return {_TYPE_TAG: "numpy_array", "dtype": str(value.dtype), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"{key_path} contains unsupported value {type(value).__module__}.{type(value).__qualname__}; "
        "use JSON-compatible values, tuples, pathlib.Path, NumPy arrays/scalars, or torch tensors/dtypes/devices"
    )


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_TYPE_TAG)
    if tag is None:
        return {key: _decode(item) for key, item in value.items()}
    if tag == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if tag == "path":
        return Path(value["value"])
    if tag == "torch_tensor":
        dtype = getattr(torch, value["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown torch dtype in saved config: {value['dtype']!r}")
        return torch.tensor(value["data"], dtype=dtype)
    if tag == "torch_dtype":
        dtype = getattr(torch, value["value"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown torch dtype in saved config: {value['value']!r}")
        return dtype
    if tag == "torch_device":
        return torch.device(value["value"])
    if tag == "numpy_array":
        return np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
    raise ValueError(f"Unknown torch_like config encoding tag: {tag!r}")


def _prepare_stage(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))


def _commit_stage(stage: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Model path already exists: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    os.replace(stage, target)


def save_model(model: Any, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
    """Persist a model into a self-describing directory.

    PyTorch-native models use a JSON constructor manifest plus ``state_dict``
    weights. Curated sklearn estimators use joblib while sharing the same
    manifest. Loading joblib/pickle artifacts is unsafe when the artifact is not
    trusted; only load model directories from trusted sources.
    """

    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Model path already exists: {target}")
    spec = get_model_spec(model)
    stage = _prepare_stage(target)
    try:
        from torch_like import __version__

        manifest: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "library": "torch_like",
            "library_version": __version__,
            "backend": spec.backend.value,
            "name": spec.name,
            "domain": spec.domain,
            "class_path": spec.class_path,
            "contract": spec.to_dict(),
        }
        if spec.backend == BackendKind.TORCH_LIKE:
            if not callable(getattr(model, "state_dict", None)) or not callable(getattr(model, "config", None)):
                raise TypeError("torch_like persistence requires model.state_dict() and model.config()")
            manifest["config"] = _encode(model.config())
            torch.save(model.state_dict(), stage / _WEIGHTS)
        elif spec.backend == BackendKind.SKLEARN:
            try:
                import joblib
            except ImportError as exc:  # pragma: no cover - sklearn normally installs joblib
                raise RuntimeError("joblib is required to save sklearn estimators") from exc
            joblib.dump(model, stage / _SKLEARN_MODEL)
        else:  # pragma: no cover - enum is closed today
            raise ValueError(f"Unsupported backend: {spec.backend}")
        (stage / _MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _commit_stage(stage, target, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def _import_class(class_path: str) -> type[Any]:
    module_name, separator, qualname = class_path.rpartition(".")
    if not separator or "<locals>" in qualname:
        raise ValueError(f"Saved class path is not importable: {class_path!r}")
    value: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not isinstance(value, type):
        raise TypeError(f"Saved class path did not resolve to a class: {class_path!r}")
    return value


def _torch_load_state(path: Path, map_location: str | torch.device | None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch releases
        return torch.load(path, map_location=map_location)


def load_model(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> Any:
    """Restore a model saved by :func:`save_model`."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / _MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing torch_like model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model schema_version={manifest.get('schema_version')!r}; expected {_SCHEMA_VERSION}"
        )
    if manifest.get("library") != "torch_like":
        raise ValueError(f"Not a torch_like model directory: {root}")

    backend = manifest.get("backend")
    if backend == BackendKind.TORCH_LIKE.value:
        name = str(manifest["name"])
        domain = str(manifest["domain"])
        config = _decode(manifest.get("config", {}))
        try:
            cls = get_model_class(name, domain=domain)
        except KeyError:
            cls = _import_class(str(manifest["class_path"]))
        model = cls(**config)
        state = _torch_load_state(root / _WEIGHTS, map_location)
        model.load_state_dict(state, strict=strict)
        if map_location is not None and callable(getattr(model, "to", None)):
            model.to(map_location)
        return model

    if backend == BackendKind.SKLEARN.value:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("joblib is required to load sklearn estimators") from exc
        return joblib.load(root / _SKLEARN_MODEL)

    raise ValueError(f"Unsupported saved backend: {backend!r}")


__all__ = ["save_model", "load_model"]
