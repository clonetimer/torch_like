"""NODE-style neural oblivious decision ensemble for tabular data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.tabular.common import TabularConfig, TabularModel


@dataclass
class NODEConfig(TabularConfig):
    num_trees: int = 16
    tree_depth: int = 4
    cat_embedding_dim: int = 8
    temperature: float = 1.0


def _leaf_codes(depth: int) -> torch.Tensor:
    leaves = torch.arange(2**depth)
    bits = [(leaves >> idx) & 1 for idx in range(depth)]
    return torch.stack(bits, dim=-1).float()


class ObliviousDecisionEnsemble(nn.Module):
    """Differentiable oblivious trees with shared split order per tree."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_trees: int,
        tree_depth: int,
        temperature: float,
    ):
        super().__init__()
        if min(input_dim, output_dim, num_trees, tree_depth) <= 0:
            raise ValueError("input_dim, output_dim, num_trees, and tree_depth must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.temperature = temperature

        self.feature_logits = nn.Parameter(torch.randn(num_trees, tree_depth, input_dim) * 0.02)
        self.thresholds = nn.Parameter(torch.zeros(num_trees, tree_depth))
        self.log_temperatures = nn.Parameter(torch.zeros(num_trees, tree_depth))
        self.leaf_responses = nn.Parameter(torch.randn(num_trees, 2**tree_depth, output_dim) * 0.02)
        self.register_buffer("leaf_codes", _leaf_codes(tree_depth), persistent=False)

    def split_weights(self) -> torch.Tensor:
        return F.softmax(self.feature_logits, dim=-1)

    def decision_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.split_weights()
        selected = torch.einsum("bf,tdf->btd", x, weights)
        scale = self.log_temperatures.exp().clamp_max(100.0) / self.temperature
        return torch.sigmoid((selected - self.thresholds.unsqueeze(0)) * scale.unsqueeze(0))

    def path_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        decisions = self.decision_probabilities(x)
        leaf_codes = self.leaf_codes.to(device=x.device, dtype=x.dtype)
        path = torch.where(
            leaf_codes.view(1, 1, -1, self.tree_depth) > 0.5, decisions.unsqueeze(2), 1.0 - decisions.unsqueeze(2)
        )
        return path.prod(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path_probs = self.path_probabilities(x)
        return torch.einsum("btl,tlo->bto", path_probs, self.leaf_responses)


class NODE(TabularModel):
    """Neural oblivious decision ensemble for classification and regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        dropout: float = 0.1,
        categories: tuple[int, ...] = (),
        cat_embedding_dim: int = 8,
        num_trees: int = 16,
        tree_depth: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        if min(input_dim, output_dim, d_model, num_trees, tree_depth) <= 0:
            raise ValueError("input_dim, output_dim, d_model, num_trees, and tree_depth must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.categories = tuple(categories)
        self.cat_embedding_dim = cat_embedding_dim
        self._config = NODEConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            categories=self.categories,
            num_trees=num_trees,
            tree_depth=tree_depth,
            cat_embedding_dim=cat_embedding_dim,
            temperature=temperature,
        )

        self.cat_embeddings = nn.ModuleList([nn.Embedding(size, cat_embedding_dim) for size in self.categories])
        total_input_dim = input_dim + len(self.categories) * cat_embedding_dim
        self.input_norm = nn.LayerNorm(total_input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ensemble = ObliviousDecisionEnsemble(
            input_dim=d_model,
            output_dim=output_dim,
            num_trees=num_trees,
            tree_depth=tree_depth,
            temperature=temperature,
        )

    def _prepare_features(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        if x_cont.dim() != 2 or x_cont.size(-1) != self.input_dim:
            raise ValueError(f"NODE expects x_cont with shape (batch, {self.input_dim})")
        features = [x_cont.float()]
        if self.categories:
            if x_cat is None or x_cat.shape != (x_cont.size(0), len(self.categories)):
                raise ValueError(f"x_cat must have shape (batch, {len(self.categories)})")
            embedded = [emb(x_cat[:, idx].long()) for idx, emb in enumerate(self.cat_embeddings)]
            features.extend(embedded)
        return self.input_norm(torch.cat(features, dim=-1))

    def forward_features(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        return self.input_projection(self._prepare_features(x_cont, x_cat))

    def tree_outputs(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        return self.ensemble(self.forward_features(x_cont, x_cat))

    def feature_importance(self) -> torch.Tensor:
        """Average differentiable split usage over trees and depths."""
        return self.ensemble.split_weights().mean(dim=(0, 1))

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        return self.tree_outputs(x_cont, x_cat).mean(dim=1)
