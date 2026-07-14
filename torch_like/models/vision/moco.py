"""MoCo-style momentum contrast visual representation learner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch_like.models.vision.common import VisionBatch, VisionConfig, VisionModel
from torch_like.models.vision.simclr import SimCLR


@dataclass
class MoCoConfig(VisionConfig):
    projection_dim: int = 128
    queue_size: int = 1024
    momentum: float = 0.999
    temperature: float = 0.2
    encoder_channels: tuple[int, ...] = (32, 64, 128)


class MoCo(VisionModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 224,
        d_model: int = 128,
        dropout: float = 0.0,
        projection_dim: int = 128,
        queue_size: int = 1024,
        momentum: float = 0.999,
        temperature: float = 0.2,
        encoder_channels: tuple[int, ...] = (32, 64, 128),
    ):
        del num_classes
        super().__init__()
        if min(in_channels, image_size, d_model, projection_dim, queue_size) <= 0:
            raise ValueError("MoCo dimensions and queue_size must be positive")
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        self._config = MoCoConfig(
            in_channels=in_channels,
            num_classes=projection_dim,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            projection_dim=projection_dim,
            queue_size=queue_size,
            momentum=momentum,
            temperature=temperature,
            encoder_channels=tuple(encoder_channels),
        )
        self.query_encoder = SimCLR(
            in_channels=in_channels,
            image_size=image_size,
            d_model=d_model,
            dropout=dropout,
            projection_dim=projection_dim,
            temperature=temperature,
            encoder_channels=encoder_channels,
        )
        self.key_encoder = deepcopy(self.query_encoder)
        for param in self.key_encoder.parameters():
            param.requires_grad = False
        queue = F.normalize(torch.randn(projection_dim, queue_size), dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_key_encoder(self) -> None:
        for query_param, key_param in zip(self.query_encoder.parameters(), self.key_encoder.parameters(), strict=True):
            key_param.data.mul_(self.momentum).add_(query_param.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = F.normalize(keys.detach(), dim=-1)
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr.item())
        if batch_size >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size :].t())
            self.queue_ptr.zero_()
            return
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.t()
        else:
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].t()
            self.queue[:, : end - self.queue_size] = keys[first:].t()
        self.queue_ptr[0] = end % self.queue_size

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder.encode_image(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(x)

    @torch.no_grad()
    def encode_key(self, x: torch.Tensor) -> torch.Tensor:
        return self.key_encoder(x).detach()

    def contrastive_logits(self, query_view: torch.Tensor, key_view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_encoder(query_view)
        with torch.no_grad():
            self.update_key_encoder()
            k = self.encode_key(key_view)
        positive = torch.einsum("bd,bd->b", q, k).unsqueeze(1)
        negative = q @ self.queue.detach()
        logits = torch.cat([positive, negative], dim=1) / self.temperature
        labels = torch.zeros(query_view.size(0), dtype=torch.long, device=query_view.device)
        return logits, labels

    def contrastive_loss(self, query_view: torch.Tensor, key_view: torch.Tensor) -> torch.Tensor:
        logits, labels = self.contrastive_logits(query_view, key_view)
        with torch.no_grad():
            self.dequeue_and_enqueue(self.encode_key(key_view))
        return F.cross_entropy(logits, labels)

    def loss_step(self, batch: VisionBatch | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(batch, tuple):
            return self.contrastive_loss(batch[0], batch[1])
        if batch.y is None:
            raise ValueError("MoCo loss_step requires batch.y to contain the key augmented view")
        return self.contrastive_loss(batch.x, batch.y)
