"""ALDM-inspired adaptive latent distribution anomaly detector.

This independent implementation retains the paper's graph-conditioned
latent-flow scoring and adaptive temporal contrastive metric. It corrects the
reference repository's detached, per-call SPD metric by registering a
positive-definite metric factor as a trainable module parameter. It does not
bundle datasets or reproduce the paper's reported benchmark scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch_like.diagnostics.types import DiagnosticResult, SignalBatch
from torch_like.models.detection.common import DetectionBatch, DetectionConfig, DetectionModel


@dataclass
class ALDMDetectorConfig(DetectionConfig):
    d_model: int = 64
    metric_dim: int = 32
    latent_dim: int = 64
    flow_layers: int = 2
    contrastive_weight: float = 0.01
    instance_weight: float = 0.5
    contrastive_temperature: float = 0.2
    metric_temperature: float = 0.5
    augmentation_noise: float = 0.02
    augmentation_mask_probability: float = 0.1
    metric_regularization: float = 1e-3


class ALDMDetector(DetectionModel):
    """Graph-conditioned latent density model with adaptive temporal contrast.

    Input windows use shape (batch, time, channels). The model learns a sensor
    graph from recurrent channel representations, maps the multivariate
    sequence into a temporal latent representation, and scores each
    sensor/time latent scalar with a conditional affine normalizing flow.

    Training is intended for normal-only windows. The contrastive objective
    uses two stochastic views and a hierarchical temporal soft-target loss.
    Soft targets are derived from a trainable SPD Mahalanobis metric over
    graph-conditioned temporal context. Inference is deterministic in eval
    mode and returns negative conditional log likelihood as the anomaly score.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        dropout: float = 0.1,
        threshold: float = 0.5,
        metric_dim: int = 32,
        latent_dim: int = 64,
        flow_layers: int = 2,
        contrastive_weight: float = 0.01,
        instance_weight: float = 0.5,
        contrastive_temperature: float = 0.2,
        metric_temperature: float = 0.5,
        augmentation_noise: float = 0.02,
        augmentation_mask_probability: float = 0.1,
        metric_regularization: float = 1e-3,
    ) -> None:
        super().__init__()
        integer_values = {
            "input_dim": input_dim,
            "seq_len": seq_len,
            "d_model": d_model,
            "metric_dim": metric_dim,
            "latent_dim": latent_dim,
            "flow_layers": flow_layers,
        }
        if any(type(value) is not int or value < 1 for value in integer_values.values()):
            raise ValueError("ALDM dimensions and flow_layers must be positive integers")
        if seq_len < 2:
            raise ValueError("ALDM requires seq_len >= 2 for temporal contrastive learning")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if not math.isfinite(contrastive_weight) or contrastive_weight < 0:
            raise ValueError("contrastive_weight must be finite and nonnegative")
        if not math.isfinite(instance_weight) or not 0.0 <= instance_weight <= 1.0:
            raise ValueError("instance_weight must be finite and in [0, 1]")
        if not math.isfinite(contrastive_temperature) or contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be finite and positive")
        if not math.isfinite(metric_temperature) or metric_temperature <= 0:
            raise ValueError("metric_temperature must be finite and positive")
        if not math.isfinite(augmentation_noise) or augmentation_noise < 0:
            raise ValueError("augmentation_noise must be finite and nonnegative")
        if not math.isfinite(augmentation_mask_probability) or not 0.0 <= augmentation_mask_probability < 1.0:
            raise ValueError("augmentation_mask_probability must be finite and in [0, 1)")
        if not math.isfinite(metric_regularization) or metric_regularization < 0:
            raise ValueError("metric_regularization must be finite and nonnegative")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.metric_dim = metric_dim
        self.latent_dim = latent_dim
        self.flow_layers = flow_layers
        self.contrastive_weight = float(contrastive_weight)
        self.instance_weight = float(instance_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.metric_temperature = float(metric_temperature)
        self.augmentation_noise = float(augmentation_noise)
        self.augmentation_mask_probability = float(augmentation_mask_probability)
        self.metric_regularization = float(metric_regularization)
        self._signal_metadata: tuple[tuple[str, ...], tuple[str, ...], float] | None = None

        self._config = ALDMDetectorConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            threshold=threshold,
            metric_dim=metric_dim,
            latent_dim=latent_dim,
            flow_layers=flow_layers,
            contrastive_weight=contrastive_weight,
            instance_weight=instance_weight,
            contrastive_temperature=contrastive_temperature,
            metric_temperature=metric_temperature,
            augmentation_noise=augmentation_noise,
            augmentation_mask_probability=augmentation_mask_probability,
            metric_regularization=metric_regularization,
        )

        self.sensor_encoder = nn.LSTM(input_size=1, hidden_size=d_model, batch_first=True)
        self.sensor_query = nn.Linear(d_model, d_model, bias=False)
        self.sensor_key = nn.Linear(d_model, d_model, bias=False)
        self.graph_message = nn.Linear(d_model, d_model)
        self.graph_norm = nn.LayerNorm(d_model)
        self.graph_dropout = nn.Dropout(dropout)

        self.metric_projection = nn.Sequential(nn.Linear(d_model, metric_dim), nn.Tanh())
        self.metric_factor = nn.Parameter(torch.eye(metric_dim))

        self.latent_encoder = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_to_sensors = nn.Linear(latent_dim, input_dim)

        self.flow_conditioners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 2),
                )
                for _ in range(flow_layers)
            ]
        )

    @staticmethod
    def _validate_uniform_sampling(batch: SignalBatch, name: str) -> None:
        if batch.timestamps_s is None:
            return
        expected_interval = 1.0 / batch.sample_rate_hz
        intervals = np.diff(batch.timestamps_s, axis=1)
        if not np.allclose(intervals, expected_interval, rtol=1e-4, atol=max(1e-9, expected_interval * 1e-6)):
            raise ValueError(f"{name} timestamps must be uniformly sampled at sample_rate_hz")

    def fit(
        self,
        reference: SignalBatch,
        *,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 42,
        gradient_clip_norm: float = 5.0,
        calibration_quantile: float = 0.995,
    ) -> "ALDMDetector":
        """Fit on healthy-only SignalBatch windows for AnomalyDiagnosticWorkflow.

        The initial native threshold is a quantile of the fitted reference
        scores. For stricter threshold calibration, call calibrate_threshold
        with a separate healthy calibration partition after fitting.
        """
        if not isinstance(reference, SignalBatch):
            raise TypeError("reference must be SignalBatch")
        if reference.shape[0] < 2:
            raise ValueError("ALDM fitting requires at least two healthy reference windows")
        if reference.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(
                f"reference must have shape (windows, {self.seq_len}, {self.input_dim})"
            )
        if not np.isfinite(reference.values).all():
            raise ValueError("reference windows must contain only finite values")
        self._validate_uniform_sampling(reference, "reference")
        integer_settings = {"epochs": epochs, "batch_size": batch_size, "seed": seed}
        if any(type(value) is not int or value < 0 for value in integer_settings.values()):
            raise ValueError("epochs, batch_size and seed must be nonnegative integers")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if not math.isfinite(calibration_quantile) or not 0.0 < calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be finite and strictly between zero and one")

        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        values = torch.as_tensor(np.array(reference.values, copy=True), device=device, dtype=dtype)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.train()
        for _ in range(epochs):
            order = torch.randperm(values.shape[0], generator=generator).tolist()
            for start in range(0, values.shape[0], batch_size):
                indices = order[start : start + batch_size]
                batch = self.make_detection_batch(values[indices])
                optimizer.zero_grad(set_to_none=True)
                loss = self.fit_step(batch)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("ALDM training produced a nonfinite loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
                if not torch.isfinite(gradient_norm).item():
                    raise FloatingPointError("ALDM training produced nonfinite gradients")
                optimizer.step()

        self._signal_metadata = (reference.channel_names, reference.units, reference.sample_rate_hz)
        self.eval()
        self.calibrate_threshold(
            self.make_detection_batch(values),
            quantile=calibration_quantile,
        )
        self.eval()
        return self

    def score(self, batch: SignalBatch) -> DiagnosticResult:
        """Return one deterministic anomaly score per SignalBatch window."""
        if not isinstance(batch, SignalBatch):
            raise TypeError("batch must be SignalBatch")
        if self._signal_metadata is None:
            raise RuntimeError("fit ALDMDetector on healthy SignalBatch windows before scoring")
        channel_names, units, sample_rate_hz = self._signal_metadata
        if batch.channel_names != channel_names:
            raise ValueError("channel names or channel ordering differ from the fitted reference")
        if batch.units != units:
            raise ValueError("channel units differ from the fitted reference")
        if batch.sample_rate_hz != sample_rate_hz:
            raise ValueError("sample_rate_hz differs from the fitted reference")
        self._validate_uniform_sampling(batch, "score")
        if batch.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError("score windows have a different sequence or channel shape")
        parameter = next(self.parameters())
        values = torch.as_tensor(np.array(batch.values, copy=True), device=parameter.device, dtype=parameter.dtype)
        was_training = self.training
        try:
            self.eval()
            with torch.inference_mode():
                scores = self(values)
        finally:
            self.train(was_training)
        return DiagnosticResult(
            task="anomaly_detection",
            output_kind="anomaly_score",
            values=scores.detach().cpu().numpy(),
            algorithm_name="ALDMDetector",
            threshold=self._config.threshold,
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("ALDMDetector expects x with shape (batch, seq_len, input_dim)")
        if x.shape[0] < 1:
            raise ValueError("ALDMDetector requires at least one input window")
        if x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(
                f"ALDMDetector expects x with shape (batch, {self.seq_len}, {self.input_dim}), got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x) or not torch.isfinite(x).all().item():
            raise ValueError("ALDMDetector requires finite floating-point input windows")

    def _encode_graph_context(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        sensor_series = x.transpose(1, 2).reshape(batch_size * self.input_dim, self.seq_len, 1)
        encoded, _ = self.sensor_encoder(sensor_series)
        hidden = encoded.reshape(batch_size, self.input_dim, self.seq_len, self.d_model)
        sensor_summary = hidden.mean(dim=2)
        query = self.sensor_query(sensor_summary)
        key = self.sensor_key(sensor_summary)
        logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.d_model)
        adjacency = torch.softmax(logits, dim=-1)
        messages = torch.einsum("bij,bjlh->bilh", adjacency, hidden)
        graph_context = self.graph_norm(hidden + self.graph_dropout(F.gelu(self.graph_message(messages))))
        return graph_context, adjacency

    def _encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.latent_encoder(x.transpose(1, 2)).transpose(1, 2)
        return self.latent_norm(latent)

    def _metric_matrix(self) -> torch.Tensor:
        diagonal = F.softplus(torch.diagonal(self.metric_factor)) + 1e-4
        lower = torch.tril(self.metric_factor, diagonal=-1) + torch.diag_embed(diagonal)
        matrix = lower @ lower.transpose(0, 1)
        return matrix * (self.metric_dim / torch.trace(matrix).clamp_min(1e-8))

    def adaptive_distances(self, context: torch.Tensor) -> torch.Tensor:
        """Return pairwise time distances under the trainable SPD metric."""
        if context.ndim != 3 or context.shape[-1] != self.metric_dim:
            raise ValueError(f"context must have shape (batch, time, {self.metric_dim})")
        difference = context.unsqueeze(2) - context.unsqueeze(1)
        metric = self._metric_matrix()
        projected = torch.matmul(difference, metric)
        squared = torch.sum(projected * difference, dim=-1).clamp_min(1e-8)
        return torch.sqrt(squared)

    def _flow_log_prob(self, latent: torch.Tensor, graph_context: torch.Tensor) -> torch.Tensor:
        values = self.latent_to_sensors(latent)
        conditions = graph_context.permute(0, 2, 1, 3)
        value = values.reshape(-1)
        condition = conditions.reshape(-1, self.d_model)
        log_determinant = torch.zeros_like(value)

        for conditioner in self.flow_conditioners:
            shift, raw_log_scale = conditioner(condition).unbind(dim=-1)
            log_scale = 2.0 * torch.tanh(raw_log_scale / 2.0)
            value = (value - shift) * torch.exp(-log_scale)
            log_determinant = log_determinant - log_scale

        base_log_prob = -0.5 * (value.square() + math.log(2.0 * math.pi))
        return (base_log_prob + log_determinant).reshape(latent.shape[0], self.seq_len, self.input_dim)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.augmentation_mask_probability > 0:
            keep = torch.rand_like(x) >= self.augmentation_mask_probability
            x = x * keep.to(dtype=x.dtype)
        if self.augmentation_noise > 0:
            x = x + torch.randn_like(x) * self.augmentation_noise
        return x

    @staticmethod
    def _pool_time(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)

    def _instance_contrastive_loss(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = first.shape
        if batch_size < 2:
            return first.new_zeros(())
        first = F.normalize(first, dim=-1)
        second = F.normalize(second, dim=-1)
        paired = torch.cat((first, second), dim=0).transpose(0, 1)
        logits = torch.matmul(paired, paired.transpose(1, 2)) / self.contrastive_temperature
        identity = torch.eye(2 * batch_size, device=first.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(identity, torch.finfo(logits.dtype).min)
        targets = (torch.arange(2 * batch_size, device=first.device) + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits.reshape(time_steps * 2 * batch_size, 2 * batch_size), targets.repeat(time_steps))

    def _temporal_contrastive_loss(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        metric_context: torch.Tensor,
    ) -> torch.Tensor:
        first = F.normalize(first, dim=-1)
        second = F.normalize(second, dim=-1)
        logits = torch.matmul(first, second.transpose(1, 2)) / self.contrastive_temperature
        distances = self.adaptive_distances(metric_context)
        targets = torch.softmax(-distances / self.metric_temperature, dim=-1)
        forward = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        reverse_targets = targets.transpose(1, 2)
        reverse = -(reverse_targets * F.log_softmax(logits.transpose(1, 2), dim=-1)).sum(dim=-1).mean()
        return 0.5 * (forward + reverse)

    def _hierarchical_contrastive_loss(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        metric_context: torch.Tensor,
    ) -> torch.Tensor:
        total = first.new_zeros(())
        levels = 0
        while first.shape[1] >= 2:
            temporal = self._temporal_contrastive_loss(first, second, metric_context)
            instance = self._instance_contrastive_loss(first, second)
            total = total + temporal + self.instance_weight * instance
            levels += 1
            if first.shape[1] < 4:
                break
            first = self._pool_time(first)
            second = self._pool_time(second)
            metric_context = self._pool_time(metric_context)
        return total / max(levels, 1)

    def _anomaly_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(x)
        graph_context, _ = self._encode_graph_context(x)
        latent = self._encode_latent(x)
        return self._flow_log_prob(latent, graph_context), graph_context

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark
        log_prob, _ = self._anomaly_components(x)
        return -log_prob.mean(dim=(1, 2))

    def fit_step(self, batch: DetectionBatch) -> torch.Tensor:
        if not isinstance(batch, DetectionBatch):
            raise TypeError("batch must be DetectionBatch")
        x = batch.x
        self._validate_input(x)
        log_prob, graph_context = self._anomaly_components(x)
        negative_log_likelihood = -log_prob.mean()

        first_view = self._encode_latent(self._augment(x))
        second_view = self._encode_latent(self._augment(x))
        metric_context = self.metric_projection(graph_context.mean(dim=1))
        contrastive_loss = self._hierarchical_contrastive_loss(first_view, second_view, metric_context)
        identity = torch.eye(self.metric_dim, device=x.device, dtype=x.dtype)
        metric_regularizer = (self._metric_matrix() - identity).square().mean()
        return (
            negative_log_likelihood
            + self.contrastive_weight * contrastive_loss
            + self.metric_regularization * metric_regularizer
        )

    def score_batch(self, batch: DetectionBatch, reduction: str = "mean") -> torch.Tensor:
        if not isinstance(batch, DetectionBatch):
            raise TypeError("batch must be DetectionBatch")
        log_prob, _ = self._anomaly_components(batch.x)
        scores = -log_prob
        if reduction == "none":
            return scores
        if reduction == "sum":
            return scores.sum(dim=-1)
        if reduction == "mean":
            return scores.mean(dim=(1, 2))
        raise ValueError("reduction must be 'none', 'sum' or 'mean'")

    @torch.no_grad()
    def calibrate_threshold(self, healthy_batch: DetectionBatch, quantile: float = 0.995) -> float:
        """Calibrate a window-level alarm threshold on independent healthy data."""
        if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be finite and strictly between zero and one")
        was_training = self.training
        try:
            self.eval()
            scores = self.score_batch(healthy_batch, reduction="mean")
            if scores.numel() == 0 or not torch.isfinite(scores).all().item():
                raise ValueError("healthy calibration scores must be nonempty and finite")
            threshold = float(torch.quantile(scores, quantile).item())
            self._config.threshold = threshold
            return threshold
        finally:
            self.train(was_training)


__all__ = ["ALDMDetectorConfig", "ALDMDetector"]
