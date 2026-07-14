"""Transformer-XL style causal language model with segment memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_like.models.sequence.common import SequenceBatch, SequenceConfig, SequenceModel


@dataclass
class TransformerXLConfig(SequenceConfig):
    mem_len: int = 128
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True


class TransformerXLBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, layer_norm_eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = x if memory is None else torch.cat([memory, x], dim=1)
        h = self.norm1(x)
        kv_norm = self.norm1(kv)
        out = self.attn(
            h, kv_norm, kv_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        x = x + out
        return x + self.ffn(self.norm2(x))


class TransformerXL(SequenceModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        mem_len: int = 128,
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff) <= 0:
            raise ValueError("vocab_size, max_seq_len, d_model, n_heads, n_layers, and d_ff must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if mem_len < 0:
            raise ValueError("mem_len must be non-negative")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.mem_len = mem_len
        self.n_layers = n_layers
        self._config = TransformerXLConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            mem_len=mem_len,
            layer_norm_eps=layer_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len + mem_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerXLBlock(d_model, n_heads, d_ff, dropout, layer_norm_eps) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def init_memory(self, batch_size: int, device: torch.device | str | None = None) -> list[torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = self.token_embedding.weight.device
        d_model = self.token_embedding.embedding_dim
        return [torch.empty(batch_size, 0, d_model, device=device) for _ in range(self.n_layers)]

    def _build_attn_mask(self, seq_len: int, mem_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(seq_len, mem_len + seq_len, dtype=torch.bool, device=device)
        mask[:, :mem_len] = False
        current = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        mask[:, mem_len:] = current
        return mask

    def forward_features(
        self,
        input_ids: torch.Tensor,
        memory: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if input_ids.dim() != 2:
            raise ValueError("TransformerXL expects input_ids with shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Expected seq_len <= {self.max_seq_len}, got {seq_len}")
        if memory is None:
            memory = self.init_memory(batch_size, input_ids.device)
        if len(memory) != self.n_layers:
            raise ValueError(f"memory must contain {self.n_layers} tensors")
        mem_len = memory[0].size(1) if memory else 0
        if mem_len + seq_len > self.position_embedding.num_embeddings:
            raise ValueError("memory length plus seq_len exceeds configured positional capacity")
        positions = torch.arange(mem_len, mem_len + seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.drop(self.token_embedding(input_ids) + self.position_embedding(positions))
        attn_mask = self._build_attn_mask(seq_len, mem_len, input_ids.device)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape")
            prefix = torch.ones(batch_size, mem_len, dtype=torch.bool, device=input_ids.device)
            key_padding_mask = ~torch.cat([prefix, attention_mask.bool()], dim=1)
        new_memory = []
        for layer_idx, block in enumerate(self.blocks):
            prev = memory[layer_idx]
            if prev.size(0) != batch_size or prev.size(-1) != hidden.size(-1):
                raise ValueError("memory tensors must match batch size and hidden width")
            hidden = block(hidden, prev if prev.numel() else None, attn_mask, key_padding_mask)
            combined = hidden if prev.numel() == 0 else torch.cat([prev, hidden], dim=1)
            new_memory.append(combined[:, -self.mem_len :].detach() if self.mem_len > 0 else combined[:, :0].detach())
        return self.norm(hidden), new_memory

    def forward_with_memory(
        self,
        input_ids: torch.Tensor,
        memory: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden, new_memory = self.forward_features(input_ids, memory=memory, attention_mask=attention_mask)
        return self.lm_head(hidden), new_memory

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_ids, causal_mask
        logits, _ = self.forward_with_memory(input_ids, attention_mask=attention_mask)
        return logits

    def loss_step(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.labels is None:
            raise ValueError("SequenceBatch.labels is required for TransformerXL loss_step")
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError("TransformerXL labels must match input_ids shape")
        return self.causal_lm_loss(self.next_token_logits(batch), batch.labels)

    @torch.no_grad()
    def generate(self, batch: SequenceBatch, max_new_tokens: int = 32) -> torch.Tensor:
        input_ids = batch.input_ids
        memory = self.init_memory(input_ids.size(0), input_ids.device)
        for start in range(0, input_ids.size(1), self.max_seq_len):
            _, memory = self.forward_with_memory(input_ids[:, start : start + self.max_seq_len], memory=memory)
        for _ in range(max_new_tokens):
            logits, memory = self.forward_with_memory(input_ids[:, -1:], memory=memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
