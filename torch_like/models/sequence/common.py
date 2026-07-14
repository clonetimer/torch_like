"""Common sequence domain configs and model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torch_like.models.base import BaseModel


@dataclass
class SequenceConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


@dataclass
class SequenceBatch:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    causal_mask: Optional[torch.Tensor] = None


def causal_sdpa_mask(
    attention_mask: torch.Tensor | None,
    causal_mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, bool]:
    """Convert framework masks to SDPA's ``True means allowed`` convention."""
    if attention_mask is None and causal_mask is None:
        return None, True

    if causal_mask is None:
        allowed = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    else:
        causal_mask = causal_mask.to(device=device, dtype=torch.bool)
        if causal_mask.dim() == 2:
            if causal_mask.shape != (seq_len, seq_len):
                raise ValueError(f"Expected causal_mask shape ({seq_len}, {seq_len}), got {tuple(causal_mask.shape)}")
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        elif causal_mask.dim() == 3:
            causal_mask = causal_mask.unsqueeze(1)
        if causal_mask.dim() != 4 or causal_mask.shape[-2:] != (seq_len, seq_len):
            raise ValueError(f"Expected causal_mask ending in ({seq_len}, {seq_len}), got {tuple(causal_mask.shape)}")
        allowed = ~causal_mask

    if attention_mask is not None:
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"Expected attention_mask shape ({batch_size}, {seq_len}), got {tuple(attention_mask.shape)}"
            )
        key_allowed = attention_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, 1, seq_len)
        allowed = allowed & key_allowed
    return allowed, False


@dataclass
class Seq2SeqBatch:
    input_ids: torch.Tensor
    decoder_input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    decoder_attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class SequenceModel(BaseModel):
    """Base class for causal sequence models."""

    def next_token_logits(self, batch: SequenceBatch) -> torch.Tensor:
        return self(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            causal_mask=batch.causal_mask,
        )

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for loss_step")
        raise NotImplementedError("Subclasses should implement causal language-model style losses")

    def causal_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute next-token loss using logits at t to predict labels at t + 1."""
        if logits.dim() != 3 or labels.dim() != 2:
            raise ValueError("causal_lm_loss expects logits (batch, seq_len, vocab) and labels (batch, seq_len)")
        if logits.shape[:2] != labels.shape:
            raise ValueError(f"Expected labels with shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}")
        if logits.size(1) < 2:
            raise ValueError("causal_lm_loss requires a sequence length of at least 2")
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement autoregressive generation if supported")

    def make_sequence_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> SequenceBatch:
        if input_ids.dim() != 2:
            raise ValueError("SequenceModel expects input_ids with shape (batch, seq_len)")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        if causal_mask is None:
            causal_mask = torch.triu(
                torch.ones(input_ids.size(1), input_ids.size(1), device=input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
        return SequenceBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            causal_mask=causal_mask,
        )


class Seq2SeqModel(BaseModel):
    """Base class for encoder-decoder sequence models."""

    def predict(self, batch: Seq2SeqBatch) -> torch.Tensor:
        return self(
            batch.input_ids,
            decoder_input_ids=batch.decoder_input_ids,
            attention_mask=batch.attention_mask,
            decoder_attention_mask=batch.decoder_attention_mask,
        )

    def loss_step(self, batch: Seq2SeqBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("Seq2SeqBatch.labels is required for loss_step")
        raise NotImplementedError("Subclasses should implement encoder-decoder losses")

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement autoregressive encoder-decoder generation")

    def make_seq2seq_batch(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Seq2SeqBatch:
        if input_ids.dim() != 2:
            raise ValueError("Seq2SeqModel expects input_ids with shape (batch, src_len)")
        if decoder_input_ids.dim() != 2:
            raise ValueError("Seq2SeqModel expects decoder_input_ids with shape (batch, tgt_len)")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.bool)
        return Seq2SeqBatch(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
