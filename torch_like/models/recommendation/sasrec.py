"""SASRec sequential recommendation model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch_like.models.recommendation.common import RecommendationConfig, RecommendationModel


@dataclass
class SASRecConfig(RecommendationConfig):
    num_items: int = 1
    max_seq_len: int = 50
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    padding_idx: int = 0


class SASRec(RecommendationModel):
    """Self-attentive sequential recommender for next-item prediction."""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
        output_dim: int = 1,
    ):
        super().__init__()
        if min(num_items, max_seq_len, d_model, n_heads, n_layers, d_ff, output_dim) <= 0:
            raise ValueError("SASRec dimensions must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self._config = SASRecConfig(
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            num_items=num_items,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            padding_idx=padding_idx,
        )
        self.item_embedding = torch.nn.Embedding(num_items + 1, d_model, padding_idx=padding_idx)
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def encode(self, item_sequences: torch.Tensor) -> torch.Tensor:
        if item_sequences.size(-1) > self.max_seq_len:
            raise ValueError("item_sequences length exceeds max_seq_len")
        item_sequences = item_sequences.long()
        batch, seq_len = item_sequences.shape
        positions = torch.arange(seq_len, device=item_sequences.device).unsqueeze(0).expand(batch, seq_len)
        hidden = self.item_embedding(item_sequences) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_sequences.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = item_sequences.eq(self.padding_idx)
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.norm(encoded)

    def sequence_representation(self, item_sequences: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(item_sequences)
        valid = item_sequences.ne(self.padding_idx)
        lengths = valid.sum(dim=-1).clamp_min(1) - 1
        batch_index = torch.arange(item_sequences.size(0), device=item_sequences.device)
        return encoded[batch_index, lengths]

    def next_item_logits(self, item_sequences: torch.Tensor) -> torch.Tensor:
        sequence_repr = self.sequence_representation(item_sequences)
        return sequence_repr @ self.item_embedding.weight[1:].t()

    def score_items(self, item_sequences: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        sequence_repr = self.sequence_representation(item_sequences)
        item_vectors = self.item_embedding(item_ids.long())
        return (sequence_repr * item_vectors).sum(dim=-1, keepdim=True)

    def forward(
        self,
        user_ids: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        x_cont: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del user_ids, x_cont
        if x_cat is None:
            raise ValueError("SASRec requires x_cat as item_sequences")
        if item_ids is None:
            return self.next_item_logits(x_cat)
        return self.score_items(x_cat, item_ids)

    def loss_step(self, batch) -> torch.Tensor:
        if batch.x_cat is None or batch.y is None:
            raise ValueError("SASRec loss_step requires batch.x_cat item sequences and batch.y next-item labels")
        logits = self.next_item_logits(batch.x_cat)
        labels = batch.y.long().view(-1) - 1
        if labels.min() < 0 or labels.max() >= self.num_items:
            raise ValueError("SASRec labels must be item ids in [1, num_items]")
        return F.cross_entropy(logits, labels)
