"""Mask-conditional split conformal prediction for incomplete sensor modalities.

This module calibrates an existing classifier's probabilities separately for
each observed modality-availability mask. It does not implement a fusion
network, evidential learning, or end-to-end conformal training.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConformalPredictionSet:
    """Immutable class-inclusion matrix and the mask-specific quantile per sample."""

    included: np.ndarray
    modality_mask: np.ndarray
    quantiles: np.ndarray
    alpha: float

    def __post_init__(self) -> None:
        included = np.asarray(self.included, dtype=bool)
        masks = np.asarray(self.modality_mask, dtype=bool)
        quantiles = np.asarray(self.quantiles, dtype=np.float64)
        if included.ndim != 2 or min(included.shape) <= 0:
            raise ValueError("included must have nonempty shape (samples, classes)")
        if masks.ndim != 2 or masks.shape[0] != included.shape[0] or masks.shape[1] <= 0:
            raise ValueError("modality_mask must have shape (samples, modalities)")
        if quantiles.shape != (included.shape[0],) or not np.isfinite(quantiles).all():
            raise ValueError("quantiles must contain one finite value per sample")
        if np.any(quantiles < 0) or np.any(quantiles > 1):
            raise ValueError("quantiles must lie in [0, 1]")
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be finite and strictly between zero and one")
        if not masks.any(axis=1).all():
            raise ValueError("each prediction must have at least one available modality")

        included = np.array(included, copy=True)
        masks = np.array(masks, copy=True)
        quantiles = np.array(quantiles, copy=True)
        included.setflags(write=False)
        masks.setflags(write=False)
        quantiles.setflags(write=False)
        object.__setattr__(self, "included", included)
        object.__setattr__(self, "modality_mask", masks)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def set_sizes(self) -> np.ndarray:
        """Number of included classes for each sample; empty sets are preserved."""
        sizes = np.sum(self.included, axis=1)
        sizes.setflags(write=False)
        return sizes

    @property
    def class_indices(self) -> tuple[tuple[int, ...], ...]:
        """Return included class indices without forcing a top-1 fallback."""
        return tuple(tuple(np.flatnonzero(row).tolist()) for row in self.included)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "mask_conditional_split_conformal_lac",
            "alpha": self.alpha,
            "modality_mask": self.modality_mask.astype(int).tolist(),
            "quantiles": self.quantiles.tolist(),
            "class_indices": [list(row) for row in self.class_indices],
            "set_sizes": self.set_sizes.tolist(),
            "coverage_scope": (
                "finite-sample marginal coverage within each calibrated availability-mask group, "
                "assuming exchangeability within that group"
            ),
        }


class ModalityConditionedConformalClassifier:
    """Mondrian split conformal calibration grouped by exact modality mask.

    The input classifier is external: pass its class probabilities and the
    boolean availability mask for each calibration/test sample. A mask with no
    calibration group is rejected; the calibrator never silently falls back to
    a global threshold. With very small groups the finite-sample quantile can
    be 1.0, yielding an uninformative but valid all-class prediction set.
    """

    def __init__(self, *, alpha: float = 0.1, min_calibration_per_mask: int = 1) -> None:
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be finite and strictly between zero and one")
        if (
            isinstance(min_calibration_per_mask, bool)
            or not isinstance(min_calibration_per_mask, int)
            or min_calibration_per_mask < 1
        ):
            raise ValueError("min_calibration_per_mask must be a positive integer")
        self.alpha = float(alpha)
        self.min_calibration_per_mask = min_calibration_per_mask
        self._quantiles: dict[tuple[int, ...], float] = {}
        self._group_sizes: dict[tuple[int, ...], int] = {}
        self._num_modalities: int | None = None
        self._num_classes: int | None = None

    @staticmethod
    def _probabilities(values: Any, *, expected_classes: int | None = None) -> np.ndarray:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError("complex-valued probabilities require an explicit real-valued representation")
        probabilities = np.asarray(raw, dtype=np.float64)
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] <= 0
            or probabilities.shape[1] <= 1
            or (expected_classes is not None and probabilities.shape[1] != expected_classes)
        ):
            raise ValueError("probabilities must have shape (samples, at least two classes)")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("probabilities must be finite and lie in [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("each probability row must sum to one")
        return probabilities

    @staticmethod
    def _masks(values: Any, *, expected_modalities: int | None = None) -> np.ndarray:
        raw = np.asarray(values)
        if raw.ndim != 2 or raw.shape[0] <= 0 or raw.shape[1] <= 0:
            raise ValueError("modality_mask must have shape (samples, modalities)")
        if expected_modalities is not None and raw.shape[1] != expected_modalities:
            raise ValueError(f"expected {expected_modalities} modality columns")
        if raw.dtype != np.bool_:
            if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
                raise ValueError("modality_mask must contain only boolean or numeric 0/1 values")
            if not np.isin(raw, (0, 1)).all():
                raise ValueError("modality_mask must contain only boolean or numeric 0/1 values")
        masks = np.asarray(raw, dtype=bool)
        if not masks.any(axis=1).all():
            raise ValueError("each sample must have at least one available modality")
        return masks

    @staticmethod
    def _labels(values: Any, *, sample_count: int, class_count: int) -> np.ndarray:
        raw = np.asarray(values)
        if (
            raw.ndim != 1
            or raw.shape[0] != sample_count
            or not np.issubdtype(raw.dtype, np.integer)
            or np.issubdtype(raw.dtype, np.bool_)
        ):
            raise ValueError("labels must be an integer class index per calibration sample")
        labels = np.asarray(raw, dtype=np.int64)
        if np.any(labels < 0) or np.any(labels >= class_count):
            raise ValueError("labels contain a class index outside the probability columns")
        return labels

    @staticmethod
    def _key(mask: np.ndarray) -> tuple[int, ...]:
        return tuple(int(value) for value in mask)

    def fit(
        self,
        calibration_probabilities: Any,
        calibration_labels: Any,
        modality_mask: Any,
    ) -> "ModalityConditionedConformalClassifier":
        """Fit mask-specific nonconformity quantiles on a held-out calibration set."""
        probabilities = self._probabilities(calibration_probabilities)
        masks = self._masks(modality_mask, expected_modalities=None)
        if masks.shape[0] != probabilities.shape[0]:
            raise ValueError("probability and modality-mask sample counts differ")
        labels = self._labels(
            calibration_labels,
            sample_count=probabilities.shape[0],
            class_count=probabilities.shape[1],
        )

        grouped_scores: dict[tuple[int, ...], list[float]] = {}
        for index, mask in enumerate(masks):
            key = self._key(mask)
            grouped_scores.setdefault(key, []).append(float(1.0 - probabilities[index, labels[index]]))

        quantiles: dict[tuple[int, ...], float] = {}
        group_sizes: dict[tuple[int, ...], int] = {}
        for key, scores in grouped_scores.items():
            count = len(scores)
            if count < self.min_calibration_per_mask:
                raise ValueError(
                    f"modality mask {key} has {count} calibration samples; "
                    f"requires at least {self.min_calibration_per_mask}"
                )
            order = ceil((count + 1) * (1.0 - self.alpha) - 1e-12)
            if order > count:
                quantile = 1.0
            else:
                quantile = float(np.partition(np.asarray(scores, dtype=np.float64), order - 1)[order - 1])
            quantiles[key] = quantile
            group_sizes[key] = count

        self._quantiles = quantiles
        self._group_sizes = group_sizes
        self._num_modalities = masks.shape[1]
        self._num_classes = probabilities.shape[1]
        return self

    @property
    def group_quantiles_(self) -> dict[tuple[int, ...], float]:
        if not self._quantiles:
            raise RuntimeError("fit the conformal calibrator before inspecting group quantiles")
        return dict(self._quantiles)

    @property
    def group_sizes_(self) -> dict[tuple[int, ...], int]:
        if not self._group_sizes:
            raise RuntimeError("fit the conformal calibrator before inspecting group sizes")
        return dict(self._group_sizes)

    def predict_set(self, probabilities: Any, modality_mask: Any) -> ConformalPredictionSet:
        """Return class sets using only the quantile for each exact availability mask."""
        if not self._quantiles or self._num_modalities is None or self._num_classes is None:
            raise RuntimeError("fit the conformal calibrator on a held-out calibration set before prediction")
        values = self._probabilities(probabilities, expected_classes=self._num_classes)
        masks = self._masks(modality_mask, expected_modalities=self._num_modalities)
        if masks.shape[0] != values.shape[0]:
            raise ValueError("probability and modality-mask sample counts differ")

        row_quantiles = np.empty(values.shape[0], dtype=np.float64)
        for index, mask in enumerate(masks):
            key = self._key(mask)
            if key not in self._quantiles:
                raise ValueError(
                    f"no calibration group for modality mask {key}; "
                    "collect calibration samples for this exact availability pattern"
                )
            row_quantiles[index] = self._quantiles[key]

        included = (1.0 - values) <= row_quantiles[:, None]
        return ConformalPredictionSet(
            included=included,
            modality_mask=masks,
            quantiles=row_quantiles,
            alpha=self.alpha,
        )

    def predict(self, probabilities: Any, modality_mask: Any) -> ConformalPredictionSet:
        """Alias for predict_set."""
        return self.predict_set(probabilities, modality_mask)


__all__ = ["ConformalPredictionSet", "ModalityConditionedConformalClassifier"]
