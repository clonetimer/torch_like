"""AGIL-inspired graph multi-task fault diagnosis and RUL candidate.

Based on Qi et al., Reliability Engineering & System Safety 263 (2025) 111209.
The implementation exposes a training-set-fitted feature graph builder and a
native PyTorch multi-task model. It is not a reproduction of the authors'
bearing dataset, preprocessing scripts or reported benchmark scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.diagnostics.types import SignalBatch


AGIL_FEATURE_NAMES = tuple(f"F{index}" for index in range(1, 17))


def extract_agil_features(signals: SignalBatch) -> np.ndarray:
    """Extract the 16 time/frequency indicators defined by the AGIL paper.

    The current adapter expects one vibration channel per SignalBatch sample.
    It returns shape (samples, 16), ordered F1...F16. The graph builder fits
    normalization and Pearson adjacency on training samples only.
    """
    if not isinstance(signals, SignalBatch):
        raise TypeError("signals must be SignalBatch")
    if signals.shape[1] < 3:
        raise ValueError("AGIL feature extraction requires at least three time points")
    if signals.shape[2] != 1:
        raise ValueError("AGIL feature extraction currently requires exactly one vibration channel")
    x = signals.values[:, :, 0]
    if not np.isfinite(x).all():
        raise ValueError("AGIL feature extraction requires finite signals")

    sample_count, length = x.shape
    mean = np.mean(x, axis=1)
    centered = x - mean[:, None]
    variance = np.var(x, axis=1, ddof=1)
    std = np.sqrt(np.maximum(variance, 0.0))
    rms = np.sqrt(np.mean(x**2, axis=1))
    absolute_mean = np.mean(np.abs(x), axis=1)
    sqrt_absolute_mean = np.sqrt(absolute_mean)
    peak = np.max(x, axis=1)
    peak_to_peak = np.max(x, axis=1) - np.min(x, axis=1)

    safe_std = np.where(std > 1e-12, std, 1.0)
    skewness = np.sum(centered**3, axis=1) / ((length - 1) * safe_std**3)
    kurtosis = np.sum(centered**4, axis=1) / ((length - 1) * safe_std**4)
    skewness = np.where(std > 1e-12, skewness, 0.0)
    kurtosis = np.where(std > 1e-12, kurtosis, 0.0)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=np.abs(denominator) > 1e-12,
        )

    frequency = np.fft.rfftfreq(length, d=1.0 / signals.sample_rate_hz)
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    if length % 2 == 0:
        power[:, 1:-1] *= 2.0
    else:
        power[:, 1:] *= 2.0
    spectral_energy = power.sum(axis=1)
    f14 = ratio((power * frequency[None, :]).sum(axis=1), 2.0 * pi * spectral_energy)
    f15 = ratio((power * (frequency[None, :] ** 2)).sum(axis=1), 4.0 * pi**2 * spectral_energy)
    f16 = np.maximum(f15 - f14**2, 0.0)

    features = np.column_stack(
        (
            mean,                                      # F1: mean
            rms,                                       # F2: root mean square
            variance,                                  # F3: sample variance
            sqrt_absolute_mean,                       # F4: square root of mean absolute value
            absolute_mean,                            # F5: mean absolute value
            peak,                                      # F6: maximum
            peak_to_peak,                             # F7: peak-to-peak
            kurtosis,                                 # F8: fourth standardized moment
            skewness,                                 # F9: third standardized moment
            ratio(peak, rms),                         # F10: crest factor
            ratio(rms, absolute_mean),                # F11: shape factor
            ratio(peak, absolute_mean),               # F12: impulse factor
            ratio(peak, sqrt_absolute_mean),          # F13: margin factor
            f14,                                       # F14: normalized spectral first moment
            f15,                                       # F15: normalized spectral second moment
            f16,                                       # F16: spectral variance
        )
    )
    if features.shape != (sample_count, 16) or not np.isfinite(features).all():
        raise ValueError("AGIL feature extraction produced invalid values")
    return features


@dataclass(frozen=True)
class AGILGraphBatch:
    """Immutable sample graphs with 16 scalar feature nodes."""

    node_features: np.ndarray
    adjacency: np.ndarray
    feature_names: tuple[str, ...] = AGIL_FEATURE_NAMES

    def __post_init__(self) -> None:
        nodes = np.asarray(self.node_features, dtype=np.float64)
        adjacency = np.asarray(self.adjacency, dtype=np.float64)
        if nodes.ndim != 3 or nodes.shape[0] <= 0 or nodes.shape[1:] != (16, 1):
            raise ValueError("node_features must have shape (samples, 16, 1)")
        if adjacency.shape != (16, 16) or not np.isfinite(adjacency).all():
            raise ValueError("adjacency must be a finite 16 by 16 matrix")
        if not np.isfinite(nodes).all() or np.any(adjacency < 0):
            raise ValueError("AGIL graph inputs must be finite with nonnegative adjacency")
        if not np.allclose(adjacency, adjacency.T, atol=1e-8):
            raise ValueError("AGIL adjacency must be symmetric")
        if not np.all(np.diag(adjacency) > 0):
            raise ValueError("AGIL adjacency must contain self-loops")
        names = tuple(self.feature_names)
        if names != AGIL_FEATURE_NAMES:
            raise ValueError("feature_names must preserve the fixed F1...F16 order")
        nodes = np.array(nodes, copy=True)
        adjacency = np.array(adjacency, copy=True)
        nodes.setflags(write=False)
        adjacency.setflags(write=False)
        object.__setattr__(self, "node_features", nodes)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "feature_names", names)

    def to_torch(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return node features and the shared adjacency on the requested device."""
        return (
            torch.tensor(self.node_features, device=device, dtype=dtype),
            torch.tensor(self.adjacency, device=device, dtype=dtype),
        )


class AGILFeatureGraphBuilder:
    """Fit a training-set Pearson graph and transform vibration windows.

    The adjacency is learned only from the supplied training partition.
    Held-out windows reuse the frozen training graph and normalization
    statistics, preventing test-distribution leakage.
    """

    def __init__(self, *, correlation_threshold: float = 0.8, min_scale: float = 1e-8) -> None:
        if not np.isfinite(correlation_threshold) or not 0.0 < correlation_threshold < 1.0:
            raise ValueError("correlation_threshold must be finite and in (0, 1)")
        if not np.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive")
        self.correlation_threshold = float(correlation_threshold)
        self.min_scale = float(min_scale)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._adjacency: np.ndarray | None = None
        self._channel_names: tuple[str, ...] | None = None
        self._units: tuple[str, ...] | None = None
        self._sample_rate_hz: float | None = None

    @property
    def adjacency_(self) -> np.ndarray:
        if self._adjacency is None:
            raise RuntimeError("fit AGILFeatureGraphBuilder on training signals first")
        result = self._adjacency.copy()
        result.setflags(write=False)
        return result

    def fit(self, training_signals: SignalBatch) -> "AGILFeatureGraphBuilder":
        features = extract_agil_features(training_signals)
        if features.shape[0] < 3:
            raise ValueError("AGIL correlation graph fitting requires at least three training windows")
        mean = features.mean(axis=0)
        scale = np.maximum(features.std(axis=0, ddof=1), self.min_scale)
        standardized = (features - mean) / scale
        correlation = np.corrcoef(standardized, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        adjacency = (correlation > self.correlation_threshold).astype(np.float64)
        adjacency = np.maximum(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, 1.0)

        for value in (mean, scale, adjacency):
            value.setflags(write=False)
        self._mean = mean
        self._scale = scale
        self._adjacency = adjacency
        self._channel_names = training_signals.channel_names
        self._units = training_signals.units
        self._sample_rate_hz = training_signals.sample_rate_hz
        return self

    def transform(self, signals: SignalBatch) -> AGILGraphBatch:
        if self._mean is None or self._scale is None or self._adjacency is None:
            raise RuntimeError("fit AGILFeatureGraphBuilder on training signals before transform")
        if not isinstance(signals, SignalBatch):
            raise TypeError("signals must be SignalBatch")
        if (
            signals.channel_names != self._channel_names
            or signals.units != self._units
            or signals.sample_rate_hz != self._sample_rate_hz
        ):
            raise ValueError("signal channel names, units and sampling rate must match the fitted training data")
        features = extract_agil_features(signals)
        standardized = (features - self._mean) / self._scale
        if not np.isfinite(standardized).all():
            raise ValueError("AGIL feature standardization produced nonfinite values")
        return AGILGraphBatch(standardized[:, :, None], self._adjacency)

    def fit_transform(self, training_signals: SignalBatch) -> AGILGraphBatch:
        return self.fit(training_signals).transform(training_signals)


@dataclass(frozen=True)
class AGILMultiTaskOutput:
    fault_logits: torch.Tensor
    rul_values: torch.Tensor


class _AGILGINLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        aggregated = torch.bmm(adjacency, x)
        hidden = self.mlp((1.0 + self.eps) * x + aggregated)
        flat = hidden.reshape(-1, hidden.shape[-1])
        valid = node_mask.reshape(-1)
        indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        selected = flat.index_select(0, indices)
        if selected.shape[0] > 1:
            selected = self.norm(selected)
        else:
            selected = F.batch_norm(
                selected,
                self.norm.running_mean,
                self.norm.running_var,
                self.norm.weight,
                self.norm.bias,
                training=False,
                eps=self.norm.eps,
            )
        normalized = torch.zeros_like(flat).index_copy(0, indices, selected)
        return F.relu(normalized.reshape_as(hidden)) * node_mask.unsqueeze(-1)


class _AGILEdgePool(nn.Module):
    """Greedy disjoint edge contraction with learnable, row-normalized scores."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_scorer = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, padded_nodes, hidden_dim = x.shape
        pooled_x = []
        pooled_adjacency = []
        pooled_mask = []

        for batch_index in range(batch_size):
            valid_indices = torch.nonzero(node_mask[batch_index], as_tuple=False).reshape(-1)
            features = x[batch_index].index_select(0, valid_indices)
            graph = adjacency[batch_index].index_select(0, valid_indices).index_select(1, valid_indices)
            node_count = features.shape[0]

            if node_count <= 1:
                clusters = [(index,) for index in range(node_count)]
                edge_probabilities = None
            else:
                left = features.unsqueeze(1).expand(node_count, node_count, hidden_dim)
                right = features.unsqueeze(0).expand(node_count, node_count, hidden_dim)
                raw_scores = self.edge_scorer(torch.cat((left, right), dim=-1)).squeeze(-1)
                eye = torch.eye(node_count, device=x.device, dtype=torch.bool)
                edge_mask = (graph > 0) & ~eye
                has_neighbors = edge_mask.any(dim=-1)
                logits = raw_scores.masked_fill(~edge_mask, -1e9)
                logits = torch.where(has_neighbors.unsqueeze(-1), logits, torch.zeros_like(logits))
                edge_probabilities = torch.softmax(logits, dim=-1) * edge_mask.to(dtype=x.dtype)
                edge_probabilities = edge_probabilities / edge_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                upper = torch.triu_indices(node_count, node_count, offset=1, device=x.device)
                edge_pairs = [
                    (int(i), int(j))
                    for i, j in zip(upper[0].tolist(), upper[1].tolist(), strict=True)
                    if float(graph[i, j].item()) > 0.0
                ]
                edge_pairs.sort(
                    key=lambda pair: (
                        -float((0.5 + 0.5 * (edge_probabilities[pair[0], pair[1]] + edge_probabilities[pair[1], pair[0]])).item()),
                        pair[0],
                        pair[1],
                    )
                )
                matched: set[int] = set()
                clusters = []
                for i, j in edge_pairs:
                    if i not in matched and j not in matched:
                        clusters.append((i, j))
                        matched.add(i)
                        matched.add(j)
                clusters.extend((index,) for index in range(node_count) if index not in matched)
                clusters.sort(key=lambda group: min(group))

            merged_features = []
            for group in clusters:
                if len(group) == 2 and edge_probabilities is not None:
                    i, j = group
                    score = 0.5 + 0.5 * (edge_probabilities[i, j] + edge_probabilities[j, i])
                    merged_features.append(score * (features[i] + features[j]))
                else:
                    merged_features.append(features[group[0]])
            merged = torch.stack(merged_features, dim=0)

            new_count = len(clusters)
            rows = []
            for row_index, row_group in enumerate(clusters):
                row = []
                for col_index, col_group in enumerate(clusters):
                    if row_index == col_index:
                        value = graph.new_tensor(1.0)
                    else:
                        row_ids = torch.as_tensor(row_group, device=x.device, dtype=torch.long)
                        col_ids = torch.as_tensor(col_group, device=x.device, dtype=torch.long)
                        value = graph.index_select(0, row_ids).index_select(1, col_ids).sum()
                    row.append(value)
                rows.append(torch.stack(row))
            new_graph = torch.stack(rows, dim=0)
            new_graph = torch.maximum(new_graph, new_graph.T)

            padded_x = F.pad(merged, (0, 0, 0, padded_nodes - new_count))
            padded_graph = F.pad(new_graph, (0, padded_nodes - new_count, 0, padded_nodes - new_count))
            mask = torch.arange(padded_nodes, device=x.device) < new_count
            pooled_x.append(padded_x)
            pooled_adjacency.append(padded_graph)
            pooled_mask.append(mask)

        return torch.stack(pooled_x), torch.stack(pooled_adjacency), torch.stack(pooled_mask)


class _AGILTaskAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, layer_features: torch.Tensor) -> torch.Tensor:
        query = self.query(layer_features)
        key = self.key(layer_features)
        value = self.value(layer_features)
        scores = torch.matmul(query, key.transpose(1, 2)) / sqrt(query.shape[-1])
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, value).mean(dim=1)


class AGILMultiTask(nn.Module):
    """Three-layer GIN + EdgePool with task-specific attention and joint heads.

    Input node features have shape (batch, 16, 1), produced by
    AGILFeatureGraphBuilder. The model returns fault logits and nonnegative
    RUL estimates. Training uses weighted cross-entropy plus MSE.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        num_nodes: int = 16,
        input_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lambda_fault: float = 0.2,
        lambda_rul: float = 0.8,
    ) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        if num_nodes != 16 or input_dim != 1:
            raise ValueError("AGILMultiTask currently requires 16 scalar feature nodes")
        if type(hidden_dim) is not int or hidden_dim < 4:
            raise ValueError("hidden_dim must be an integer >= 4")
        if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not np.isfinite(lambda_fault) or lambda_fault <= 0:
            raise ValueError("lambda_fault must be finite and positive")
        if not np.isfinite(lambda_rul) or lambda_rul <= 0:
            raise ValueError("lambda_rul must be finite and positive")

        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda_fault = float(lambda_fault)
        self.lambda_rul = float(lambda_rul)

        self.gin_layers = nn.ModuleList(
            [
                _AGILGINLayer(input_dim, hidden_dim),
                _AGILGINLayer(hidden_dim, hidden_dim),
                _AGILGINLayer(hidden_dim, hidden_dim),
            ]
        )
        self.edge_pools = nn.ModuleList([_AGILEdgePool(hidden_dim) for _ in range(3)])
        self.fault_attention = _AGILTaskAttention(hidden_dim)
        self.rul_attention = _AGILTaskAttention(hidden_dim)
        self.fault_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.ReLU(),
        )

    def _prepare_graph(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(node_features, torch.Tensor) or node_features.ndim != 3:
            raise ValueError("node_features must have shape (batch, 16, 1)")
        if node_features.shape[1:] != (self.num_nodes, self.input_dim):
            raise ValueError("node_features must have shape (batch, 16, 1)")
        if not torch.isfinite(node_features).all().item():
            raise ValueError("node_features must be finite")
        parameter = next(self.parameters())
        x = node_features.to(device=parameter.device, dtype=parameter.dtype)

        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("adjacency must be a torch.Tensor")
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(x.shape[0], -1, -1)
        if adjacency.shape != (x.shape[0], self.num_nodes, self.num_nodes):
            raise ValueError("adjacency must have shape (16, 16) or (batch, 16, 16)")
        if adjacency.is_complex() or not torch.isfinite(adjacency).all().item() or torch.any(adjacency < 0).item():
            raise ValueError("adjacency must be real, finite and nonnegative")
        graph = adjacency.to(device=x.device, dtype=x.dtype)
        if not torch.allclose(graph, graph.transpose(1, 2), atol=1e-5, rtol=1e-5):
            raise ValueError("AGIL adjacency must be symmetric")
        graph = graph.clone()
        graph.diagonal(dim1=1, dim2=2).fill_(1.0)

        if node_mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        else:
            if not isinstance(node_mask, torch.Tensor) or node_mask.shape != x.shape[:2]:
                raise ValueError("node_mask must have shape (batch, 16)")
            if node_mask.dtype != torch.bool:
                if not torch.isfinite(node_mask).all().item() or not torch.all(
                    (node_mask == 0) | (node_mask == 1)
                ).item():
                    raise ValueError("node_mask must contain only boolean or numeric 0/1 values")
            mask = node_mask.to(device=x.device, dtype=torch.bool)
            if not mask.any(dim=1).all().item():
                raise ValueError("each graph must contain at least one valid node")
        return x, graph, mask

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> AGILMultiTaskOutput:
        x, graph, mask = self._prepare_graph(node_features, adjacency, node_mask)
        layer_readouts = []

        for gin, pool in zip(self.gin_layers, self.edge_pools, strict=True):
            x = gin(x, graph, mask)
            x, graph, mask = pool(x, graph, mask)
            weights = mask.to(dtype=x.dtype).unsqueeze(-1)
            readout = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            layer_readouts.append(readout)

        stacked = torch.stack(layer_readouts, dim=1)
        fault_features = self.fault_attention(stacked)
        rul_features = self.rul_attention(stacked)
        fault_logits = self.fault_head(fault_features)
        rul_values = self.rul_head(rul_features).squeeze(-1)
        return AGILMultiTaskOutput(fault_logits=fault_logits, rul_values=rul_values)

    def predict_proba(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fault-class probabilities from the diagnosis head."""
        return torch.softmax(self.forward(node_features, adjacency, node_mask).fault_logits, dim=-1)

    def predict_rul(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return nonnegative RUL estimates from the prognosis head."""
        return self.forward(node_features, adjacency, node_mask).rul_values

    def training_loss(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        fault_labels: torch.Tensor,
        rul_targets: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.forward(node_features, adjacency, node_mask)
        if not isinstance(fault_labels, torch.Tensor) or fault_labels.shape != (output.rul_values.shape[0],):
            raise ValueError("fault_labels must have shape (batch,)")
        if fault_labels.dtype not in {torch.int64, torch.long, torch.int32, torch.int16, torch.uint8}:
            raise ValueError("fault_labels must use an integer dtype")
        labels = fault_labels.to(device=output.fault_logits.device, dtype=torch.long)
        if torch.any(labels < 0).item() or torch.any(labels >= self.num_classes).item():
            raise ValueError("fault_labels contain a class index outside num_classes")
        if not isinstance(rul_targets, torch.Tensor) or rul_targets.shape != output.rul_values.shape:
            raise ValueError("rul_targets must have shape (batch,)")
        targets = rul_targets.to(device=output.rul_values.device, dtype=output.rul_values.dtype)
        if not torch.isfinite(targets).all().item() or torch.any(targets < 0).item():
            raise ValueError("rul_targets must be finite and nonnegative")

        fault_loss = F.cross_entropy(output.fault_logits, labels)
        rul_loss = F.mse_loss(output.rul_values, targets)
        return self.lambda_fault * fault_loss + self.lambda_rul * rul_loss


__all__ = [
    "AGIL_FEATURE_NAMES",
    "AGILFeatureGraphBuilder",
    "AGILGraphBatch",
    "AGILMultiTask",
    "AGILMultiTaskOutput",
    "extract_agil_features",
]
