"""TimeGrad-style diffusion probabilistic forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.forecasting.common import (
    ProbabilisticForecastingBatch,
    ProbabilisticForecastingConfig,
    ProbabilisticForecastingModel,
)


class TimeGrad(ProbabilisticForecastingModel):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        distribution: str = "diffusion_gaussian",
        num_samples: int = 100,
        diffusion_steps: int = 8,
    ):
        super().__init__()
        del n_heads, d_layers, d_ff
        if distribution != "diffusion_gaussian":
            raise ValueError("TimeGrad currently supports distribution='diffusion_gaussian' only")

        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.distribution = distribution
        self._config = ProbabilisticForecastingConfig(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=8,
            e_layers=e_layers,
            d_layers=1,
            d_ff=128,
            dropout=dropout,
            distribution=distribution,
            num_samples=num_samples,
        )

        self.history_proj = nn.Linear(enc_in, d_model)
        self.history_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=e_layers,
            dropout=dropout if e_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder_proj = nn.Linear(dec_in, d_model)
        self.time_embedding = nn.Embedding(diffusion_steps, d_model)
        self.noise_predictor = nn.Sequential(
            nn.Linear(d_model * 3 + c_out, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_out),
        )

        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def _encode_history(self, x_enc: torch.Tensor) -> torch.Tensor:
        if x_enc.dim() != 3:
            raise ValueError("TimeGrad expects x_enc with shape (batch, seq_len, num_features)")
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x_enc.size(1)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} encoder features, got {x_enc.size(-1)}")
        history = self.history_proj(x_enc)
        _, hidden = self.history_encoder(history)
        return hidden[-1]

    def _decode_context(self, x_dec: torch.Tensor) -> torch.Tensor:
        if x_dec.dim() != 3:
            raise ValueError("TimeGrad expects x_dec with shape (batch, label_len + pred_len, num_features)")
        if x_dec.size(-1) != self.dec_in:
            raise ValueError(f"Expected {self.dec_in} decoder features, got {x_dec.size(-1)}")
        future = x_dec[:, -self.pred_len :, :]
        return self.decoder_proj(future)

    def _predict_noise(
        self,
        noisy_target: torch.Tensor,
        history_state: torch.Tensor,
        decoder_context: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        time_embed = self.time_embedding(step_ids).unsqueeze(1).expand(-1, self.pred_len, -1)
        history_embed = history_state.unsqueeze(1).expand(-1, self.pred_len, -1)
        features = torch.cat([noisy_target, history_embed, decoder_context, time_embed], dim=-1)
        return self.noise_predictor(features)

    def _reverse_diffusion(
        self,
        history_state: torch.Tensor,
        decoder_context: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        batch_size = history_state.size(0)
        samples = torch.randn(
            num_samples,
            batch_size,
            self.pred_len,
            self.c_out,
            device=history_state.device,
            dtype=history_state.dtype,
        )
        expanded_history = history_state.unsqueeze(0).expand(num_samples, -1, -1).reshape(num_samples * batch_size, -1)
        expanded_context = (
            decoder_context.unsqueeze(0)
            .expand(num_samples, -1, -1, -1)
            .reshape(num_samples * batch_size, self.pred_len, -1)
        )

        for step in reversed(range(self.diffusion_steps)):
            current = samples.reshape(num_samples * batch_size, self.pred_len, self.c_out)
            step_ids = torch.full(
                (num_samples * batch_size,),
                step,
                device=current.device,
                dtype=torch.long,
            )
            pred_noise = self._predict_noise(current, expanded_history, expanded_context, step_ids)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            mean = (current - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)
            if step > 0:
                mean = mean + torch.sqrt(beta) * torch.randn_like(mean)
            samples = mean.reshape(num_samples, batch_size, self.pred_len, self.c_out)
        return samples

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor | None,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
        history_state = self._encode_history(x_enc)
        decoder_context = self._decode_context(x_dec)
        return self._reverse_diffusion(history_state, decoder_context, num_samples=1).mean(dim=0)

    def predict_distribution(self, batch: ProbabilisticForecastingBatch):
        samples = self.sample(batch)
        mean = samples.mean(dim=0)
        scale = samples.std(dim=0, unbiased=False) + 1e-4
        return {"mean": mean, "scale": scale, "distribution": self.distribution}

    def sample(self, batch: ProbabilisticForecastingBatch, num_samples: int | None = None) -> torch.Tensor:
        history_state = self._encode_history(batch.x_enc)
        decoder_context = self._decode_context(batch.x_dec)
        sample_count = num_samples or self.num_samples
        return self._reverse_diffusion(history_state, decoder_context, num_samples=sample_count)

    def fit_step(self, batch: ProbabilisticForecastingBatch) -> torch.Tensor:
        if batch.y_true is None:
            raise ValueError("ProbabilisticForecastingBatch.y_true is required for fit_step")
        expected_shape = (batch.x_enc.size(0), self.pred_len, self.c_out)
        if batch.y_true.shape != expected_shape:
            raise ValueError(f"TimeGrad fit_step expects y_true with shape {expected_shape}, got {tuple(batch.y_true.shape)}")

        history_state = self._encode_history(batch.x_enc)
        decoder_context = self._decode_context(batch.x_dec)
        batch_size = batch.y_true.size(0)
        step_ids = torch.randint(0, self.diffusion_steps, (batch_size,), device=batch.y_true.device)
        alpha_bars = self.alpha_bars[step_ids].view(batch_size, 1, 1)
        noise = torch.randn_like(batch.y_true)
        noisy_target = torch.sqrt(alpha_bars) * batch.y_true + torch.sqrt(1.0 - alpha_bars) * noise
        pred_noise = self._predict_noise(noisy_target, history_state, decoder_context, step_ids)
        return F.mse_loss(pred_noise, noise)
