"""LightGCN graph collaborative filtering model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch_like.models.recommendation.common import RecommendationConfig, RecommendationModel


@dataclass
class LightGCNConfig(RecommendationConfig):
    num_users: int = 1
    num_items: int = 1
    embedding_dim: int = 64
    num_layers: int = 3


class LightGCN(RecommendationModel):
    """Light graph convolution for user-item collaborative filtering."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        output_dim: int = 1,
        d_model: int | None = None,
    ):
        super().__init__()
        if min(num_users, num_items, embedding_dim, output_dim) <= 0 or num_layers < 0:
            raise ValueError("LightGCN dimensions must be positive and num_layers must be non-negative")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self._config = LightGCNConfig(
            output_dim=output_dim,
            d_model=embedding_dim if d_model is None else d_model,
            dropout=dropout,
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

    def initial_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def propagate(self, adjacency: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.initial_embeddings()
        if adjacency is None or self.num_layers == 0:
            final = embeddings
        else:
            all_embeddings = [embeddings]
            current = embeddings
            for _ in range(self.num_layers):
                current = torch.sparse.mm(adjacency, current) if adjacency.is_sparse else adjacency @ current
                all_embeddings.append(current)
            final = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]

    def score_pairs(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_emb, item_emb = self.propagate(adjacency)
        scores = (user_emb[user_ids.long()] * item_emb[item_ids.long()]).sum(dim=-1, keepdim=True)
        return scores

    def forward(
        self,
        user_ids: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        x_cont: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_cat, x_cont
        if user_ids is None or item_ids is None:
            raise ValueError("LightGCN requires user_ids and item_ids")
        return self.score_pairs(user_ids, item_ids, adjacency=adjacency)

    def full_sort_scores(self, user_ids: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        user_emb, item_emb = self.propagate(adjacency)
        return user_emb[user_ids.long()] @ item_emb.t()

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        reg_weight: float = 1e-4,
    ) -> torch.Tensor:
        user_emb, item_emb = self.propagate(adjacency)
        users = user_emb[user_ids.long()]
        positives = item_emb[positive_item_ids.long()]
        negatives = item_emb[negative_item_ids.long()]
        pos_scores = (users * positives).sum(dim=-1)
        neg_scores = (users * negatives).sum(dim=-1)
        ranking_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = (
            self.user_embedding(user_ids.long()).pow(2).sum(dim=-1)
            + self.item_embedding(positive_item_ids.long()).pow(2).sum(dim=-1)
            + self.item_embedding(negative_item_ids.long()).pow(2).sum(dim=-1)
        ).mean()
        return ranking_loss + reg_weight * reg
