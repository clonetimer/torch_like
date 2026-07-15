"""Whisper-style encoder-decoder speech recognition model.

Framework-oriented implementation of "Robust Speech Recognition via
Large-Scale Weak Supervision": a log-mel audio encoder and an autoregressive
text decoder with cross-attention over audio features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.audio.common import AudioConfig, AudioModel


@dataclass
class WhisperConfig(AudioConfig):
    vocab_size: int = 51865
    max_audio_positions: int = 1500
    max_text_positions: int = 448
    n_heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    d_ff: int = 2048
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


class WhisperAudioEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        max_audio_positions: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_audio_positions = max_audio_positions
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_audio_positions, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Whisper expects log-mel input with shape (batch, n_mels, time) or (batch, time, n_mels)")
        if x.size(1) == self.input_dim:
            return x
        if x.size(-1) == self.input_dim:
            return x.transpose(1, 2)
        raise ValueError(f"Whisper expects one input axis to equal input_dim={self.input_dim}")

    def _padding_mask(self, lengths: torch.Tensor | None, max_len: int, device: torch.device) -> torch.Tensor | None:
        if lengths is None:
            return None
        encoded_lengths = torch.div(lengths.to(device=device) + 1, 2, rounding_mode="floor").clamp(max=max_len)
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        return positions >= encoded_lengths.unsqueeze(1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self._normalize_input(x).float()
        features = F.gelu(self.conv1(x))
        features = F.gelu(self.conv2(features)).transpose(1, 2)
        if features.size(1) > self.max_audio_positions:
            raise ValueError(
                f"Encoded audio length {features.size(1)} exceeds max_audio_positions={self.max_audio_positions}"
            )
        mask = self._padding_mask(lengths, features.size(1), features.device)
        features = features + self.pos_embed[:, : features.size(1), :].to(dtype=features.dtype)
        return self.norm(self.encoder(features, src_key_padding_mask=mask)), mask


class WhisperTextDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_text_positions: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        pad_token_id: int,
    ):
        super().__init__()
        self.max_text_positions = max_text_positions
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_text_positions, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embed.weight
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("decoder input_ids must have shape (batch, text_length)")
        if input_ids.size(1) > self.max_text_positions:
            raise ValueError(f"text_length exceeds max_text_positions={self.max_text_positions}")
        tokens = self.token_embed(input_ids.long())
        tokens = tokens + self.pos_embed[:, : input_ids.size(1), :].to(dtype=tokens.dtype)
        tgt_key_padding_mask = input_ids.eq(self.pad_token_id)
        decoded = self.decoder(
            tokens,
            memory,
            tgt_mask=self._causal_mask(input_ids.size(1), input_ids.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_projection(self.norm(decoded))


class Whisper(AudioModel):
    """Whisper-style ASR model with encoder, decoder, loss, and greedy generation."""

    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 51865,
        d_model: int = 256,
        dropout: float = 0.1,
        vocab_size: int | None = None,
        max_audio_positions: int = 1500,
        max_text_positions: int = 448,
        n_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        d_ff: int = 2048,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ):
        super().__init__()
        vocab_size = output_dim if vocab_size is None else vocab_size
        if output_dim != vocab_size:
            raise ValueError("output_dim and vocab_size must match for Whisper")
        if min(input_dim, output_dim, d_model, n_heads, encoder_layers, decoder_layers, d_ff) <= 0:
            raise ValueError("Whisper dimensions and layer counts must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not (0 <= pad_token_id < vocab_size and 0 <= bos_token_id < vocab_size and 0 <= eos_token_id < vocab_size):
            raise ValueError("special token ids must be in vocabulary range")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self._config = WhisperConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
            max_audio_positions=max_audio_positions,
            max_text_positions=max_text_positions,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            d_ff=d_ff,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        self.encoder = WhisperAudioEncoder(
            input_dim=input_dim,
            d_model=d_model,
            max_audio_positions=max_audio_positions,
            n_heads=n_heads,
            n_layers=encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = WhisperTextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_text_positions=max_text_positions,
            n_heads=n_heads,
            n_layers=decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

    def encode_audio(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        memory, _ = self.encoder(x, lengths=lengths)
        return memory

    def decode_tokens(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(input_ids, audio_features, memory_key_padding_mask=audio_padding_mask)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, audio_mask = self.encoder(x, lengths=lengths)
        if decoder_input_ids is None:
            decoder_input_ids = torch.full((x.size(0), 1), self.bos_token_id, dtype=torch.long, device=x.device)
        return self.decode_tokens(decoder_input_ids, memory, audio_padding_mask=audio_mask)

    def loss_step(self, batch) -> torch.Tensor:
        if batch.y is None:
            raise ValueError("Whisper loss_step requires batch.y token ids")
        if batch.y.dim() != 2 or batch.y.size(1) < 2:
            raise ValueError("Whisper batch.y must have shape (batch, text_length >= 2)")
        decoder_input_ids = batch.y[:, :-1].long()
        target_ids = batch.y[:, 1:].long()
        logits = self(batch.x, lengths=batch.lengths, decoder_input_ids=decoder_input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.pad_token_id,
        )

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        max_new_tokens: int = 16,
        decoder_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        memory, audio_mask = self.encoder(x, lengths=lengths)
        if decoder_input_ids is None:
            generated = torch.full((x.size(0), 1), self.bos_token_id, dtype=torch.long, device=x.device)
        else:
            generated = decoder_input_ids.to(device=x.device, dtype=torch.long)
        for _ in range(max_new_tokens):
            logits = self.decode_tokens(generated, memory, audio_padding_mask=audio_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1).eq(self.eos_token_id)):
                break
        return generated
