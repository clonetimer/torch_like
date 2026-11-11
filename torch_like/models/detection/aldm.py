"""ALDM-style latent-distribution detector for multivariate time series.

This portable implementation keeps the paper's main design: sensor-graph
context, a latent temporal representation, adaptive metric soft-contrastive
learning, and conditional flow likelihood scoring. It deliberately separates
the unsupervised training objective from deterministic inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


_LOG_2PI = log(2.0 * pi)


@dataclass
class ALDMDetectorConfig(DetectionConfig):
    """Configuration for the portable ALDM anomaly detector."""

    metric_dim: int = 32
    metric_rank: int = 8
    flow_hidden_dim: int = 64
    flow_layers: int = 1
    contrastive_weight: float = 0.01
    instance_weight: float = 0.5
    temperature: float = 0.2
    metric_scale: float = 1.0
    augmentation_noise: float = 0.02
    augmentation_mask_probability: float = 0.05


class SensorGraphAttention(nn.Module):
    """Learn a dense sensor relation graph from window-level sensor states."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sensor_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(sensor_states)
        key = self.key(sensor_states)
        value = self.value(sensor_states)
        logits = torch.matmul(query, key.transpose(1, 2)) / sqrt(sensor_states.size(-1))
        adjacency = torch.softmax(logits, dim=-1)
        messages = torch.matmul(adjacency, value)
        updated = self.norm(sensor_states + self.dropout(self.output(messages)))
        return adjacency, updated


class AdaptiveSPDMetric(nn.Module):
    """Positive-definite low-rank Mahalanobis metric for temporal soft labels.

    W = U U^T + floor * I is strictly positive definite. The low-rank factor
    is a registered trainable parameter, so the metric is optimized together
    with the detector rather than being re-created inside the loss function.
    """

    def __init__(self, feature_dim: int, rank: int, floor: float = 1e-3) -> None:
        super().__init__()
        if feature_dim <= 0 or rank <= 0 or rank > feature_dim:
            raise ValueError("metric feature_dim and rank must be positive, with rank <= feature_dim")
        if floor <= 0:
            raise ValueError("metric floor must be positive")
        self.factor = nn.Parameter(torch.randn(feature_dim, rank) * 0.02)
        self.floor = float(floor)

    def pairwise_distance(self, values: torch.Tensor) -> torch.Tensor:
        """Return pairwise distances for a batch/time/feature tensor."""
        if values.ndim != 3 or values.size(-1) != self.factor.size(0):
            raise ValueError("metric input must have shape (batch, time, metric_dim)")
        projected = torch.matmul(values, self.factor)
        projected_gram = torch.matmul(projected, projected.transpose(1, 2))
        projected_norm = projected.square().sum(dim=-1, keepdim=True)
        projected_distance = projected_norm + projected_norm.transpose(1, 2) - 2.0 * projected_gram

        identity_gram = torch.matmul(values, values.transpose(1, 2))
        identity_norm = values.square().sum(dim=-1, keepdim=True)
        identity_distance = identity_norm + identity_norm.transpose(1, 2) - 2.0 * identity_gram

        squared = (projected_distance + self.floor * identity_distance).clamp_min(1e-12)
        return torch.sqrt(squared)


class ConditionalMAFBlock(nn.Module):
    """One-dimensional conditional affine autoregressive flow transform.

    The official ALDM implementation applies MAF to scalar sensor values
    (input_size=1) conditioned on graph features. In that one-dimensional
    setting the masked conditioner reduces to a conditional affine transform.
    """

    def __init__(self, context_dim: int, hidden_dim: int, scale_clip: float = 2.0) -> None:
        super().__init__()
        if context_dim <= 0 or hidden_dim <= 0 or scale_clip <= 0:
            raise ValueError("flow dimensions and scale_clip must be positive")
        self.conditioner = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.scale_clip = float(scale_clip)
        final = self.conditioner[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, raw_log_scale = self.conditioner(context).chunk(2, dim=-1)
        log_scale = torch.tanh(raw_log_scale) * self.scale_clip
        transformed = (values - shift) * torch.exp(-log_scale)
        return transformed, -log_scale


class ALDMDetector(DetectionModel):
    """Unsupervised ALDM-style detector for windows shaped (batch, length, channels).

    Training minimizes conditional negative log likelihood plus a hierarchical
    instance/temporal contrastive objective. Inference is deterministic and
    returns one negative log-likelihood anomaly score per window. No labels,
    test-set model selection, dataset-specific scaler, or decision threshold
    are inferred by this model.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        dropout: float = 0.1,
        threshold: float = 0.5,
        metric_dim: int = 32,
        metric_rank: int = 8,
        flow_hidden_dim: int = 64,
        flow_layers: int = 1,
        contrastive_weight: float = 0.01,
        instance_weight: float = 0.5,
        temperature: float = 0.2,
        metric_scale: float = 1.0,
        augmentation_noise: float = 0.02,
        augmentation_mask_probability: float = 0.05,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or seq_len < 2 or d_model <= 0:
            raise ValueError("input_dim and d_model must be positive and seq_len must be at least two")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if metric_dim <= 0 or metric_rank <= 0 or metric_rank > metric_dim:
            raise ValueError("metric_dim and metric_rank must be positive, with metric_rank <= metric_dim")
        if flow_hidden_dim <= 0 or flow_layers <= 0:
            raise ValueError("flow_hidden_dim and flow_layers must be positive")
        if contrastive_weight < 0:
            raise ValueError("contrastive_weight must be nonnegative")
        if not 0.0 <= instance_weight <= 1.0:
            raise ValueError("instance_weight must lie in [0, 1]")
        if temperature <= 0 or metric_scale <= 0:
            raise ValueError("temperature and metric_scale must be positive")
        if augmentation_noise < 0:
            raise ValueError("augmentation_noise must be nonnegative")
        if not 0.0 <= augmentation_mask_probability < 1.0:
            raise ValueError("augmentation_mask_probability must lie in [0, 1)")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.metric_dim = int(metric_dim)
        self.threshold = float(threshold)
        self.contrastive_weight = float(contrastive_weight)
        self.instance_weight = float(instance_weight)
        self.temperature = float(temperature)
        self.metric_scale = float(metric_scale)
        self.augmentation_noise = float(augmentation_noise)
        self.augmentation_mask_probability = float(augmentation_mask_probability)

        self._config = ALDMDetectorConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            threshold=threshold,
            metric_dim=metric_dim,
            metric_rank=metric_rank,
            flow_hidden_dim=flow_hidden_dim,
            flow_layers=flow_layers,
            contrastive_weight=contrastive_weight,
            instance_weight=instance_weight,
            temperature=temperature,
            metric_scale=metric_scale,
            augmentation_noise=augmentation_noise,
            augmentation_mask_probability=augmentation_mask_probability,
        )

        self.sensor_encoder = nn.LSTM(input_size=1, hidden_size=d_model, batch_first=True)
        self.sensor_graph = SensorGraphAttention(d_model, dropout)
        self.graph_projection = nn.Linear(d_model, d_model)
        self.temporal_transition = nn.Linear(d_model, d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.metric_projection = nn.Linear(input_dim * d_model, metric_dim)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(metric_dim, metric_dim),
            nn.GELU(),
            nn.Linear(metric_dim, metric_dim),
        )
        self.adaptive_metric = AdaptiveSPDMetric(metric_dim, metric_rank)
        self.flow = nn.ModuleList(
            [ConditionalMAFBlock(d_model, flow_hidden_dim) for _ in range(flow_layers)]
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("ALDMDetector input must be a torch tensor")
        if x.ndim != 3:
            raise ValueError("ALDMDetector expects x with shape (batch, seq_len, input_dim)")
        if x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(
                f"Expected input shape (batch, {self.seq_len}, {self.input_dim}), got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x) or not torch.isfinite(x).all():
            raise ValueError("ALDMDetector input must contain finite floating-point values")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sensor/time context, sensor adjacency, and temporal metric features."""
        self._validate_input(x)
        batch_size, seq_len, channels = x.shape
        sensor_series = x.transpose(1, 2).reshape(batch_size * channels, seq_len, 1)
        encoded, _ = self.sensor_encoder(sensor_series)
        temporal = encoded.reshape(batch_size, channels, seq_len, self.d_model)

        sensor_summary = temporal.mean(dim=2)
        adjacency, sensor_context = self.sensor_graph(sensor_summary)
        graph_messages = torch.einsum("bcs,bsld->bcld", adjacency, temporal)
        shifted = torch.cat(
            (
                torch.zeros_like(temporal[:, :, :1, :]),
                self.temporal_transition(temporal[:, :, :-1, :]),
            ),
            dim=2,
        )
        context = self.context_norm(
            temporal
            + self.graph_projection(graph_messages)
            + sensor_context.unsqueeze(2)
            + shifted
        )
        temporal_features = context.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, channels * self.d_model
        )
        metric_features = self.metric_projection(temporal_features)
        return context, adjacency, metric_features

    def _log_prob(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, adjacency, metric_features = self.encode(x)
        values = x.transpose(1, 2).unsqueeze(-1)
        log_det = torch.zeros_like(values)
        for layer in self.flow:
            values, contribution = layer(values, context)
            log_det = log_det + contribution
        base_log_prob = -0.5 * (values.square() + _LOG_2PI)
        element_log_prob = base_log_prob + log_det
        window_log_prob = element_log_prob.sum(dim=-1).mean(dim=(1, 2))
        return window_log_prob, adjacency, metric_features

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        augmented = x
        if self.augmentation_noise:
            augmented = augmented + torch.randn_like(augmented) * self.augmentation_noise
        if self.augmentation_mask_probability:
            keep = torch.rand_like(augmented) >= self.augmentation_mask_probability
            augmented = augmented * keep.to(dtype=augmented.dtype)
        return augmented

    def _instance_contrastive_loss(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.size(0) == 1:
            return first.new_zeros(())
        left = F.normalize(first.mean(dim=1), dim=-1)
        right = F.normalize(second.mean(dim=1), dim=-1)
        logits = torch.matmul(left, right.transpose(0, 1)) / self.temperature
        targets = torch.arange(first.size(0), device=first.device)
        return 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.transpose(0, 1), targets)
        )

    def _temporal_contrastive_loss(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        metric_features: torch.Tensor,
    ) -> torch.Tensor:
        if first.size(1) == 1:
            return first.new_zeros(())
        left = F.normalize(first, dim=-1)
        right = F.normalize(second, dim=-1)
        logits = torch.matmul(left, right.transpose(1, 2)) / self.temperature
        distances = self.adaptive_metric.pairwise_distance(metric_features)
        soft_targets = torch.softmax(-distances * self.metric_scale, dim=-1)
        return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def _hierarchical_contrastive_loss(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        metric_features: torch.Tensor,
    ) -> torch.Tensor:
        scale_losses = []
        while True:
            instance_loss = self._instance_contrastive_loss(first, second)
            temporal_loss = self._temporal_contrastive_loss(first, second, metric_features)
            scale_losses.append(
                self.instance_weight * instance_loss
                + (1.0 - self.instance_weight) * temporal_loss
            )
            if first.size(1) == 1:
                break
            first = F.max_pool1d(first.transpose(1, 2), kernel_size=2).transpose(1, 2)
            second = F.max_pool1d(second.transpose(1, 2), kernel_size=2).transpose(1, 2)
            metric_features = F.max_pool1d(
                metric_features.transpose(1, 2), kernel_size=2
            ).transpose(1, 2)
        return torch.stack(scale_losses).mean()

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        log_prob, _, _ = self._log_prob(x)
        return -log_prob

    def fit_step(self, batch: DetectionBatch) -> torch.Tensor:
        log_prob, _, _ = self._log_prob(batch.x)
        negative_log_likelihood = -log_prob.mean()
        if self.contrastive_weight == 0:
            return negative_log_likelihood

        first_view = self._augment(batch.x)
        second_view = self._augment(batch.x)
        _, _, first_metric = self.encode(first_view)
        _, _, second_metric = self.encode(second_view)
        first_embedding = self.contrastive_projection(first_metric)
        second_embedding = self.contrastive_projection(second_metric)
        contrastive = self._hierarchical_contrastive_loss(
            first_embedding,
            second_embedding,
            0.5 * (first_metric + second_metric),
        )
        return negative_log_likelihood + self.contrastive_weight * contrastive

    def sensor_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the learned sensor-attention matrix for inspection."""
        _, adjacency, _ = self.encode(x)
        return adjacency

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        scores = self.forward(batch.x)
        if reduction == "none":
            return scores
        if reduction == "sum":
            return scores.sum()
        return scores.mean()


__all__ = ["ALDMDetectorConfig", "ALDMDetector"]
