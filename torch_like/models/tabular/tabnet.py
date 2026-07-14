"""TabNet-style attentive tabular model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.tabular.common import TabularBatch, TabularConfig, TabularModel


@dataclass
class TabNetConfig(TabularConfig):
    n_steps: int = 3
    gamma: float = 1.5
    sparsity_loss_weight: float = 1e-3


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    range_values = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = -1
    range_values = range_values.view(shape)
    support = 1 + range_values * sorted_logits > sorted_logits.cumsum(dim)
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (sorted_logits.mul(support).sum(dim=dim, keepdim=True) - 1) / support_size
    return torch.clamp(logits - tau, min=0.0)


class GLUBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2 * out_dim)
        self.norm = nn.LayerNorm(2 * out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.norm(self.linear(x)).chunk(2, dim=-1)
        return self.dropout(value * torch.sigmoid(gate))


class TabNet(TabularModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        dropout: float = 0.1,
        categories: tuple[int, ...] = (),
        cat_embedding_dim: int = 8,
        n_steps: int = 3,
        gamma: float = 1.5,
        sparsity_loss_weight: float = 1e-3,
    ):
        super().__init__()
        if min(input_dim, output_dim, d_model, n_steps) <= 0:
            raise ValueError("input_dim, output_dim, d_model, and n_steps must be positive")
        if gamma <= 1.0:
            raise ValueError("gamma must be greater than 1")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.categories = tuple(categories)
        self.cat_embedding_dim = cat_embedding_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.sparsity_loss_weight = sparsity_loss_weight
        self._config = TabNetConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            categories=self.categories,
            n_steps=n_steps,
            gamma=gamma,
            sparsity_loss_weight=sparsity_loss_weight,
        )
        self.cat_embeddings = nn.ModuleList([nn.Embedding(size, cat_embedding_dim) for size in self.categories])
        total_input_dim = input_dim + len(self.categories) * cat_embedding_dim
        self.input_norm = nn.LayerNorm(total_input_dim)
        self.feature_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    GLUBlock(total_input_dim, d_model, dropout),
                    GLUBlock(d_model, d_model, dropout),
                )
                for _ in range(n_steps)
            ]
        )
        self.attentive = nn.ModuleList([nn.Linear(d_model, total_input_dim) for _ in range(n_steps)])
        self.head = nn.Linear(d_model, output_dim)

    def _prepare_features(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        if x_cont.dim() != 2 or x_cont.size(-1) != self.input_dim:
            raise ValueError(f"TabNet expects x_cont with shape (batch, {self.input_dim})")
        features = [x_cont.float()]
        if self.categories:
            if x_cat is None or x_cat.shape != (x_cont.size(0), len(self.categories)):
                raise ValueError(f"x_cat must have shape (batch, {len(self.categories)})")
            embedded = [emb(x_cat[:, idx].long()) for idx, emb in enumerate(self.cat_embeddings)]
            features.extend(embedded)
        return self.input_norm(torch.cat(features, dim=-1))

    def forward_with_masks(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        features = self._prepare_features(x_cont, x_cat)
        prior = torch.ones_like(features)
        aggregate = features.new_zeros(features.size(0), self._config.d_model)
        masks: list[torch.Tensor] = []
        sparsity_loss = features.new_tensor(0.0)
        context = features.new_zeros(features.size(0), self._config.d_model)
        for block, attention in zip(self.feature_blocks, self.attentive, strict=True):
            mask = sparsemax(attention(context) * prior, dim=-1)
            masked_features = mask * features
            decision = block(masked_features)
            aggregate = aggregate + F.relu(decision)
            context = decision
            prior = prior * (self.gamma - mask)
            sparsity_loss = sparsity_loss + (mask * torch.log(mask + 1e-8)).sum(dim=-1).mean()
            masks.append(mask)
        return self.head(aggregate), masks, sparsity_loss / self.n_steps

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        logits, _, _ = self.forward_with_masks(x_cont, x_cat)
        return logits

    def explain(self, batch: TabularBatch) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        logits, masks, sparsity_loss = self.forward_with_masks(batch.x_cont, batch.x_cat)
        return {"logits": logits, "masks": masks, "sparsity_loss": sparsity_loss}

    def loss_step(self, batch: TabularBatch) -> torch.Tensor:
        if batch.y is None:
            raise ValueError("TabNet loss_step requires batch.y")
        logits, _, sparsity_loss = self.forward_with_masks(batch.x_cont, batch.x_cat)
        if logits.size(-1) > 1 and batch.y.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            task_loss = F.cross_entropy(logits, batch.y.long())
        else:
            target = batch.y.to(device=logits.device, dtype=logits.dtype).reshape_as(logits)
            task_loss = F.mse_loss(logits, target)
        return task_loss + self.sparsity_loss_weight * sparsity_loss
