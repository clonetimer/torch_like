"""Portable EnCodec-style neural audio codec and residual vector quantizer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.audio.common import AudioBatch, AudioConfig, AudioModel


@dataclass
class EnCodecConfig(AudioConfig):
    sample_rate: int = 24000
    channels: tuple[int, ...] = (32, 64, 128)
    strides: tuple[int, ...] = (2, 4, 5)
    codebook_size: int = 1024
    num_codebooks: int = 8
    target_bandwidth: float = 6.0


class ResidualVectorQuantizer(nn.Module):
    """Stacked vector quantizers with residual refinement and straight-through gradients."""

    def __init__(self, dim: int, codebook_size: int = 1024, num_codebooks: int = 8):
        super().__init__()
        if min(dim, codebook_size, num_codebooks) <= 0:
            raise ValueError("RVQ dimensions must be positive")
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.codebooks = nn.Parameter(torch.empty(num_codebooks, codebook_size, dim))
        nn.init.uniform_(self.codebooks, -1 / codebook_size, 1 / codebook_size)

    def _nearest(self, residual, codebook):
        flat = residual.reshape(-1, self.dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        quantized = F.embedding(indices, codebook).view_as(residual)
        return quantized, indices.view(residual.shape[:-1])

    def encode(self, x: torch.Tensor, num_codebooks: int | None = None) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.dim:
            raise ValueError(f"Expected x with shape (batch, frames, {self.dim})")
        count = self.num_codebooks if num_codebooks is None else num_codebooks
        if not 1 <= count <= self.num_codebooks:
            raise ValueError(f"num_codebooks must be in [1, {self.num_codebooks}]")
        residual = x
        codes = []
        for index in range(count):
            quantized, code = self._nearest(residual, self.codebooks[index])
            residual = residual - quantized
            codes.append(code)
        return torch.stack(codes, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() != 3 or not 1 <= codes.size(1) <= self.num_codebooks:
            raise ValueError("Expected codes with shape (batch, codebooks, frames)")
        quantized = 0.0
        for index in range(codes.size(1)):
            quantized = quantized + F.embedding(codes[:, index].long(), self.codebooks[index])
        return quantized

    def forward(self, x: torch.Tensor, num_codebooks: int | None = None):
        codes = self.encode(x.detach(), num_codebooks)
        quantized = self.decode(codes)
        commitment = F.mse_loss(x, quantized.detach()) + F.mse_loss(quantized, x.detach())
        straight_through = x + (quantized - x).detach()
        return straight_through, codes, commitment


class _ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class EnCodec(AudioModel):
    """Random-initialized causal neural codec with bandwidth-scalable RVQ."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 128,
        dropout: float = 0.0,
        sample_rate: int = 24000,
        channels: tuple[int, ...] = (32, 64, 128),
        strides: tuple[int, ...] = (2, 4, 5),
        codebook_size: int = 1024,
        num_codebooks: int = 8,
        target_bandwidth: float = 6.0,
    ):
        super().__init__()
        del dropout
        if input_dim != output_dim:
            raise ValueError("EnCodec requires matching input_dim and output_dim")
        if len(channels) != len(strides) or not channels:
            raise ValueError("channels and strides must be non-empty and have equal length")
        if min(sample_rate, d_model, codebook_size, num_codebooks) <= 0:
            raise ValueError("EnCodec dimensions must be positive")
        if any(channel <= 0 for channel in channels) or any(stride <= 1 for stride in strides):
            raise ValueError("channels must be positive and strides must be greater than one")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.hop_length = math.prod(strides)
        self.target_bandwidth = float(target_bandwidth)
        self._config = EnCodecConfig(
            input_dim, output_dim, d_model, 0.0, sample_rate, channels, strides,
            codebook_size, num_codebooks, target_bandwidth,
        )

        encoder = []
        current = input_dim
        for index, (channel, stride) in enumerate(zip(channels, strides, strict=True)):
            encoder.extend([
                nn.Conv1d(current, channel, kernel_size=2 * stride, stride=stride, padding=stride // 2),
                _ResidualUnit(channel, 3**min(index, 2)),
            ])
            current = channel
        encoder.extend([nn.ELU(), nn.Conv1d(current, d_model, 3, padding=1)])
        self.encoder = nn.Sequential(*encoder)
        self.quantizer = ResidualVectorQuantizer(d_model, codebook_size, num_codebooks)

        decoder = [nn.Conv1d(d_model, channels[-1], 3, padding=1)]
        current = channels[-1]
        for index, stride in enumerate(reversed(strides)):
            out_channels = input_dim if index == len(channels) - 1 else channels[-index - 2]
            decoder.extend([
                _ResidualUnit(current, 3**min(index, 2)),
                nn.ELU(),
                nn.ConvTranspose1d(
                    current,
                    out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2,
                    output_padding=stride % 2,
                ),
            ])
            current = out_channels
        self.decoder = nn.Sequential(*decoder)
        self.set_target_bandwidth(target_bandwidth)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    def bandwidth_to_num_codebooks(self, bandwidth: float) -> int:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        bits_per_code = math.log2(self.quantizer.codebook_size)
        count = round(bandwidth * 1000 / (self.frame_rate * bits_per_code))
        return max(1, min(self.quantizer.num_codebooks, count))

    def set_target_bandwidth(self, bandwidth: float) -> None:
        self.active_codebooks = self.bandwidth_to_num_codebooks(float(bandwidth))
        self.target_bandwidth = float(bandwidth)

    def _prepare_waveform(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected waveform shape (batch, time) or (batch, {self.input_dim}, time)")
        length = x.size(-1)
        padded_length = math.ceil(length / self.hop_length) * self.hop_length
        return F.pad(x.float(), (0, padded_length - length)), length

    def encode(self, x, *, bandwidth: float | None = None):
        waveform, length = self._prepare_waveform(x)
        latent = self.encoder(waveform).transpose(1, 2)
        count = self.active_codebooks if bandwidth is None else self.bandwidth_to_num_codebooks(bandwidth)
        quantized, codes, commitment = self.quantizer(latent, count)
        return {"codes": codes, "quantized": quantized, "commitment_loss": commitment, "length": length}

    def _decode_latent(self, latent, length):
        latent = latent.transpose(1, 2)
        waveform = self.decoder(latent)
        if length is not None:
            waveform = waveform[..., :length]
        return waveform.squeeze(1) if self.input_dim == 1 else waveform

    def decode(self, codes, *, length: int | None = None):
        return self._decode_latent(self.quantizer.decode(codes), length)

    def forward(self, x, lengths=None, *, bandwidth: float | None = None):
        del lengths
        encoded = self.encode(x, bandwidth=bandwidth)
        return self._decode_latent(encoded["quantized"], encoded["length"])

    @staticmethod
    def _spectral_loss(prediction, target):
        if prediction.dim() == 3:
            prediction = prediction.mean(dim=1)
            target = target.mean(dim=1)
        losses = []
        for n_fft in (32, 64):
            if target.size(-1) < n_fft:
                continue
            window = torch.hann_window(n_fft, device=prediction.device, dtype=prediction.dtype)
            pred_stft = torch.stft(prediction, n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
            target_stft = torch.stft(target, n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
            losses.append(F.l1_loss(pred_stft.abs(), target_stft.abs()))
        return sum(losses) / len(losses) if losses else prediction.new_zeros(())

    def loss_step(self, batch: AudioBatch):
        target = batch.x.float()
        encoded = self.encode(target)
        reconstruction = self._decode_latent(encoded["quantized"], encoded["length"])
        waveform_loss = F.l1_loss(reconstruction, target)
        return waveform_loss + self._spectral_loss(reconstruction, target) + 0.25 * encoded["commitment_loss"]
