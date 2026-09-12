"""Artifact persistence helpers shared by orchestration and backend runners.

Kept in its own module so both :mod:`torch_like.automl.task` (which writes run
metadata) and :mod:`torch_like.automl.runners` (which writes model artifacts)
can use one implementation without importing each other.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sklearn_artifact(path: Path, pipeline, candidate) -> None:
    """Persist a fitted sklearn pipeline next to an auditable manifest."""
    joblib.dump(pipeline, path)
    optional_versions = {}
    if candidate.name in {"LightGBMClassifier", "LightGBMRegressor"}:
        optional_versions["lightgbm"] = importlib.metadata.version("lightgbm")
    json_dump(
        path.parent / "artifact_manifest.json",
        {
            "backend": "sklearn",
            "format": "joblib",
            "trusted_local_only": True,
            "sha256": file_sha256(path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            **optional_versions,
        },
    )


__all__ = ["file_sha256", "json_dump", "write_sklearn_artifact"]
