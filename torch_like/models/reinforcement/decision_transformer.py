"""Decision Transformer for offline trajectory-conditioned control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torch_like.models.reinforcement.common import ReinforcementBatch, ReinforcementConfig, ReinforcementModel


@dataclass
class DecisionTransformerConfig(ReinforcementConfig):
    max_seq_len: int = 20
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    max_timestep: int = 4096
    action_tanh: bool = True


class DecisionTransformer(ReinforcementModel):
    """Causal Transformer over return, state, and action tokens."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        d_model: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 20,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        max_timestep: int = 4096,
        action_tanh: bool = True,
    ):
        super().__init__()
        if min(state_dim, action_dim, d_model, max_seq_len, n_heads, n_layers, d_ff, max_timestep) <= 0:
            raise ValueError("DecisionTransformer dimensions and limits must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.max_timestep = max_timestep
        self.action_tanh = action_tanh
        self._config = DecisionTransformerConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            hidden_layers=hidden_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_timestep=max_timestep,
            action_tanh=action_tanh,
        )
        self.return_embed = torch.nn.Linear(1, d_model)
        self.state_embed = torch.nn.Linear(state_dim, d_model)
        self.action_embed = torch.nn.Linear(action_dim, d_model)
        self.timestep_embed = torch.nn.Embedding(max_timestep, d_model)
        self.embed_ln = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_ln = torch.nn.LayerNorm(d_model)
        self.predict_state_head = torch.nn.Linear(d_model, state_dim)
        self.predict_action_head = torch.nn.Linear(d_model, action_dim)
        self.predict_return_head = torch.nn.Linear(d_model, 1)

    def _normalize_returns(self, returns_to_go: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        returns = returns_to_go.to(device=states.device, dtype=states.dtype)
        if returns.dim() == 2:
            returns = returns.unsqueeze(-1)
        if returns.shape[-1] != 1:
            raise ValueError("returns_to_go must have shape (batch, time) or (batch, time, 1)")
        return returns

    def _default_timesteps(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        timesteps = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        return timesteps.clamp(max=self.max_timestep - 1)

    def _build_tokens(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if states.dim() != 3 or actions.dim() != 3:
            raise ValueError("states and actions must have shape (batch, time, feature)")
        if states.shape[:2] != actions.shape[:2]:
            raise ValueError("states and actions must share batch and time dimensions")
        if states.shape[-1] != self.state_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("states/actions feature dimensions do not match model config")
        batch_size, seq_len, _ = states.shape
        if seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        returns = self._normalize_returns(returns_to_go, states)
        if returns.shape[:2] != states.shape[:2]:
            raise ValueError("returns_to_go must share batch and time dimensions with states")
        if timesteps is None:
            timesteps = self._default_timesteps(batch_size, seq_len, states.device)
        timesteps = timesteps.to(device=states.device, dtype=torch.long).clamp(0, self.max_timestep - 1)
        time_embeddings = self.timestep_embed(timesteps)
        return_tokens = self.return_embed(returns) + time_embeddings
        state_tokens = self.state_embed(states) + time_embeddings
        action_tokens = self.action_embed(actions.to(dtype=states.dtype, device=states.device)) + time_embeddings
        tokens = torch.stack((return_tokens, state_tokens, action_tokens), dim=2)
        return self.embed_ln(tokens.reshape(batch_size, seq_len * 3, -1))

    def _causal_mask(self, token_count: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(token_count, token_count, device=device, dtype=torch.bool), diagonal=1)

    def _key_padding_mask(self, attention_mask: Optional[torch.Tensor], states: torch.Tensor) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.shape != states.shape[:2]:
            raise ValueError("attention_mask must have shape (batch, time)")
        token_mask = attention_mask.to(device=states.device).bool().repeat_interleave(3, dim=1)
        return ~token_mask

    def encode_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self._build_tokens(states, actions, returns_to_go, timesteps=timesteps)
        encoded = self.transformer(
            tokens,
            mask=self._causal_mask(tokens.size(1), tokens.device),
            src_key_padding_mask=self._key_padding_mask(attention_mask, states),
        )
        return self.final_ln(encoded)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_trajectory(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        batch_size, seq_len, _ = states.shape
        hidden = encoded.reshape(batch_size, seq_len, 3, -1)
        return_hidden = hidden[:, :, 0]
        state_hidden = hidden[:, :, 1]
        action_hidden = hidden[:, :, 2]
        action_preds = self.predict_action_head(state_hidden)
        if self.action_tanh:
            action_preds = torch.tanh(action_preds)
        return {
            "state_preds": self.predict_state_head(action_hidden),
            "action_preds": action_preds,
            "return_preds": self.predict_return_head(action_hidden),
            "hidden_states": hidden,
            "return_hidden": return_hidden,
            "state_hidden": state_hidden,
            "action_hidden": action_hidden,
        }

    def predict_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.forward(states, actions, returns_to_go, timesteps=timesteps, attention_mask=attention_mask)
        if attention_mask is None:
            return outputs["action_preds"][:, -1]
        lengths = attention_mask.to(device=states.device).long().sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(states.size(0), device=states.device)
        return outputs["action_preds"][batch_index, lengths]

    def action_loss(
        self,
        predictions: dict[str, torch.Tensor],
        target_actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_preds = predictions["action_preds"]
        targets = target_actions.to(device=action_preds.device, dtype=action_preds.dtype)
        losses = F.mse_loss(action_preds, targets, reduction="none").mean(dim=-1)
        if attention_mask is None:
            return losses.mean()
        mask = attention_mask.to(device=losses.device, dtype=losses.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (losses * mask).sum() / denom

    def loss_step(self, batch: ReinforcementBatch) -> torch.Tensor:
        if batch.actions is None or batch.returns_to_go is None:
            raise ValueError("DecisionTransformer loss_step requires actions and returns_to_go")
        predictions = self.forward(
            states=batch.states,
            actions=batch.actions,
            returns_to_go=batch.returns_to_go,
            timesteps=batch.timesteps,
            attention_mask=batch.attention_mask,
        )
        return self.action_loss(predictions, batch.actions, attention_mask=batch.attention_mask)
