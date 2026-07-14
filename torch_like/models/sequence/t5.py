"""T5-style text-to-text Transformer.

Framework-oriented implementation of "Exploring the Limits of Transfer
Learning with a Unified Text-to-Text Transformer". T5 is represented as an
encoder-decoder model over token ids; tokenization and task prefixing remain
outside the model class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.sequence.common import Seq2SeqBatch, Seq2SeqModel, SequenceConfig


@dataclass
class T5Config(SequenceConfig):
    pad_token_id: int = 0
    decoder_start_token_id: int = 0
    tie_word_embeddings: bool = True


class T5Embeddings(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, dropout: float, pad_token_id: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return self.dropout(self.token_embedding(input_ids) + self.position_embedding(position_ids))


class T5(Seq2SeqModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        decoder_start_token_id: int = 0,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(vocab_size, max_seq_len, d_model, n_layers) <= 0:
            raise ValueError("vocab_size, max_seq_len, d_model, and n_layers must be positive")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self._config = T5Config(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
        )

        self.encoder_embeddings = T5Embeddings(vocab_size, max_seq_len, d_model, dropout, pad_token_id)
        self.decoder_embeddings = T5Embeddings(vocab_size, max_seq_len, d_model, dropout, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.decoder_embeddings.token_embedding.weight = self.encoder_embeddings.token_embedding.weight
            self.lm_head.weight = self.encoder_embeddings.token_embedding.weight
        self._init_weights()

    @classmethod
    def t5_small(cls, **kwargs) -> "T5":
        return cls(d_model=512, n_heads=8, n_layers=6, d_ff=2048, **kwargs)

    @classmethod
    def t5_tiny(cls, **kwargs) -> "T5":
        return cls(d_model=128, n_heads=4, n_layers=2, d_ff=512, **kwargs)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("T5 expects input_ids with shape (batch, src_len)")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"Expected source length <= {self.max_seq_len}, got {input_ids.size(1)}")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        hidden = self.encoder_embeddings(input_ids)
        return self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decoder_input_ids.dim() != 2:
            raise ValueError("T5 expects decoder_input_ids with shape (batch, tgt_len)")
        if decoder_input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"Expected target length <= {self.max_seq_len}, got {decoder_input_ids.size(1)}")
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id)
        tgt_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=decoder_input_ids.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.decoder_embeddings(decoder_input_ids)
        memory_key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        decoded = self.decoder(
            hidden,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~decoder_attention_mask.bool(),
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.final_norm(decoded)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(input_ids, attention_mask)
        hidden = self.decode(decoder_input_ids, memory, attention_mask, decoder_attention_mask)
        return self.lm_head(hidden)

    def loss_step(self, batch: Seq2SeqBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("Seq2SeqBatch.labels is required for T5 loss_step")
        logits = self.predict(batch)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), batch.labels.reshape(-1), ignore_index=-100)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        attention_mask: torch.Tensor | None = None,
        decoder_start_token_id: int | None = None,
    ) -> torch.Tensor:
        if decoder_start_token_id is None:
            decoder_start_token_id = self.decoder_start_token_id
        memory = self.encode(input_ids, attention_mask)
        generated = torch.full(
            (input_ids.size(0), 1),
            decoder_start_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        for _ in range(max_new_tokens):
            logits = self.lm_head(self.decode(generated, memory, attention_mask))
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
