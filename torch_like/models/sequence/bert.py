"""BERT-style bidirectional Transformer encoder.

Framework-oriented implementation of "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding". It exposes masked-token logits and
pooled sequence representations through the existing sequence-domain contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


@dataclass
class BERTConfig(SequenceConfig):
    type_vocab_size: int = 2
    pad_token_id: int = 0
    layer_norm_eps: float = 1e-5


class BERTEmbeddings(nn.Module):
    def __init__(
        self, vocab_size: int, max_seq_len: int, type_vocab_size: int, d_model: int, dropout: float, pad_token_id: int
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.norm(x))


class BERT(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self._config = BERTConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            type_vocab_size=type_vocab_size,
            pad_token_id=pad_token_id,
            layer_norm_eps=layer_norm_eps,
        )

        self.embeddings = BERTEmbeddings(vocab_size, max_seq_len, type_vocab_size, d_model, dropout, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.mlm_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=layer_norm_eps),
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.lm_head.weight = self.embeddings.word_embeddings.weight

    @classmethod
    def bert_tiny(cls, **kwargs) -> "BERT":
        return cls(d_model=128, n_heads=2, n_layers=2, d_ff=512, **kwargs)

    @classmethod
    def bert_base(cls, **kwargs) -> "BERT":
        return cls(d_model=768, n_heads=12, n_layers=12, d_ff=3072, **kwargs)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("BERT expects input_ids with shape (batch, seq_len)")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"Expected sequence length <= {self.max_seq_len}, got {input_ids.size(1)}")
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        key_padding_mask = ~attention_mask.bool()
        x = self.embeddings(input_ids, position_ids, token_type_ids)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

    def pooled_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_features(input_ids, attention_mask, position_ids, token_type_ids)
        return self.pooler(hidden[:, 0, :])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del causal_mask
        hidden = self.forward_features(input_ids, attention_mask, position_ids)
        return self.lm_head(self.mlm_transform(hidden)) + self.lm_bias

    def next_token_logits(self, batch: SequenceBatch) -> torch.Tensor:
        # For BERT this returns masked-token logits at every position, not causal next-token logits.
        return self(batch.input_ids, attention_mask=batch.attention_mask, position_ids=batch.position_ids)

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for BERT loss_step")
        logits = self.next_token_logits(batch)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), batch.labels.reshape(-1), ignore_index=-100)

    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32) -> torch.Tensor:
        del batch, max_new_tokens
        raise NotImplementedError("BERT is bidirectional and does not support autoregressive generation")
