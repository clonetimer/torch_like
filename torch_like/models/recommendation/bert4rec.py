"""BERT4Rec-style bidirectional sequential recommendation model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch_like.models.recommendation.common import RecommendationConfig, RecommendationModel


@dataclass
class BERT4RecConfig(RecommendationConfig):
    num_items: int = 1
    max_seq_len: int = 50
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    padding_idx: int = 0
    mask_token_id: int = 2


class BERT4Rec(RecommendationModel):
    """Bidirectional Transformer recommender for masked item prediction."""

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
            raise ValueError("BERT4Rec dimensions must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.mask_token_id = num_items + 1
        self._config = BERT4RecConfig(
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            num_items=num_items,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            padding_idx=padding_idx,
            mask_token_id=self.mask_token_id,
        )
        self.item_embedding = torch.nn.Embedding(num_items + 2, d_model, padding_idx=padding_idx)
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

    def mask_items(self, item_sequences: torch.Tensor, mask_positions: torch.Tensor) -> torch.Tensor:
        """Replace selected positions with the learned mask token id."""
        if mask_positions.shape != item_sequences.shape:
            raise ValueError(f"mask_positions must have shape {tuple(item_sequences.shape)}")
        masked = item_sequences.long().clone()
        masked[mask_positions.to(device=item_sequences.device, dtype=torch.bool)] = self.mask_token_id
        return masked

    def encode(self, item_sequences: torch.Tensor) -> torch.Tensor:
        if item_sequences.dim() != 2:
            raise ValueError("BERT4Rec expects item_sequences with shape (batch, seq_len)")
        if item_sequences.size(-1) > self.max_seq_len:
            raise ValueError("item_sequences length exceeds max_seq_len")
        if item_sequences.max() > self.mask_token_id or item_sequences.min() < 0:
            raise ValueError("item ids must be in [0, num_items] or equal to mask_token_id")
        item_sequences = item_sequences.long()
        batch, seq_len = item_sequences.shape
        positions = torch.arange(seq_len, device=item_sequences.device).unsqueeze(0).expand(batch, seq_len)
        hidden = self.item_embedding(item_sequences) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        padding_mask = item_sequences.eq(self.padding_idx)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.norm(encoded)

    def sequence_logits(self, item_sequences: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(item_sequences)
        return encoded @ self.item_embedding.weight[1 : self.num_items + 1].t()

    def masked_item_logits(self, item_sequences: torch.Tensor, mask_positions: torch.Tensor) -> torch.Tensor:
        logits = self.sequence_logits(item_sequences)
        if mask_positions.dtype == torch.bool:
            if mask_positions.shape != item_sequences.shape:
                raise ValueError(f"mask_positions must have shape {tuple(item_sequences.shape)}")
            return logits[mask_positions]
        if mask_positions.dim() != 1 or mask_positions.size(0) != item_sequences.size(0):
            raise ValueError("integer mask_positions must have shape (batch,)")
        batch_index = torch.arange(item_sequences.size(0), device=item_sequences.device)
        return logits[batch_index, mask_positions.long()]

    def score_items(
        self,
        item_sequences: torch.Tensor,
        item_ids: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask_positions is None:
            valid = item_sequences.ne(self.padding_idx)
            mask_positions = valid.sum(dim=-1).clamp_min(1) - 1
        logits = self.masked_item_logits(item_sequences, mask_positions)
        item_ids = item_ids.long()
        if item_ids.dim() == 1:
            labels = item_ids - 1
            return logits.gather(1, labels.view(-1, 1))
        return logits.gather(1, item_ids - 1)

    def forward(
        self,
        user_ids: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        x_cont: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del user_ids, x_cont
        if x_cat is None:
            raise ValueError("BERT4Rec requires x_cat as item_sequences")
        if item_ids is None:
            return self.sequence_logits(x_cat)
        return self.score_items(x_cat, item_ids)

    def loss_step(self, batch) -> torch.Tensor:
        if batch.x_cat is None or batch.y is None:
            raise ValueError("BERT4Rec loss_step requires batch.x_cat item sequences and batch.y masked-item labels")
        logits = self.sequence_logits(batch.x_cat)
        labels = batch.y.long()
        if labels.shape != batch.x_cat.shape:
            raise ValueError("BERT4Rec labels must have shape (batch, seq_len); use 0 for ignored positions")
        active = labels.ne(self.padding_idx)
        if not torch.any(active):
            raise ValueError("BERT4Rec loss_step requires at least one non-zero masked label")
        shifted = labels[active] - 1
        if shifted.min() < 0 or shifted.max() >= self.num_items:
            raise ValueError("BERT4Rec labels must be item ids in [1, num_items] or 0 for ignored positions")
        return F.cross_entropy(logits[active], shifted)
