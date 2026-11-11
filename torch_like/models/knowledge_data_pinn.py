"""Knowledge-data-driven physics-informed neural network for RUL regression.

Paper basis: Fang et al., Mechanical Systems and Signal Processing 232 (2025),
112736. This module implements the MLP-based M-PINN mechanism as a decreasing
RUL variant: a data network Psi predicts RUL, a positive-rate network Gamma
learns the unknown degradation/RUL-rate mechanism, and an autograd residual
couples the two networks. It is a paper-aligned subset, not a reproduction of
the authors' milling datasets or reported metrics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_like.models.rul import RULBatch, RULModel


def _make_tanh_mlp(
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for index in range(hidden_layers):
        next_width = max(hidden_dim // (2**index), 4)
        layers.extend((nn.Linear(width, next_width), nn.Tanh()))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = next_width
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class KnowledgeDataPINNRUL(RULModel):
    """MLP-based knowledge-data-driven PINN adapted to decreasing RUL.

    Parameters
    ----------
    enc_in:
        Number of normalized sensor/health features at each time step.
    seq_len:
        Number of time steps in each input window.
    hidden_dim:
        First hidden-layer width for both Psi and Gamma networks. Subsequent
        widths are halved (e.g. 32, 16, 8), following the paper's M-PINN.
    hidden_layers:
        Number of hidden layers in each network.
    physics_weight:
        Weight for the learned governing-rate residual loss. The paper uses 1.
    monotonicity_weight:
        Weight for the sequence monotonicity penalty. The paper uses 0.5.
    min_rate:
        Positive floor for Gamma's softplus output.
    dropout:
        Dropout probability in the two MLPs.

    Input shape is (batch, time, features). Optional x_mark must contain one
    strictly increasing time coordinate per step with shape (batch,time,1).
    When omitted, normalized coordinates in [0,1] are generated. Targets are
    the nonnegative RUL at the final time step of each window.
    """

    def __init__(
        self,
        enc_in: int = 16,
        seq_len: int = 60,
        hidden_dim: int = 32,
        hidden_layers: int = 3,
        physics_weight: float = 1.0,
        monotonicity_weight: float = 0.5,
        min_rate: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        integer_values = {
            "enc_in": enc_in,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
        }
        if any(type(value) is not int or value <= 0 for value in integer_values.values()):
            raise ValueError("enc_in, seq_len, hidden_dim and hidden_layers must be positive integers")
        if seq_len < 2:
            raise ValueError("seq_len must be at least two for physics and monotonicity losses")
        numeric_values = {
            "physics_weight": physics_weight,
            "monotonicity_weight": monotonicity_weight,
            "min_rate": min_rate,
            "dropout": dropout,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in numeric_values.values()
        ):
            raise ValueError("physics weights, min_rate and dropout must be finite real numbers")
        if physics_weight < 0:
            raise ValueError("physics_weight must be nonnegative")
        if monotonicity_weight < 0:
            raise ValueError("monotonicity_weight must be nonnegative")
        if min_rate <= 0:
            raise ValueError("min_rate must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.physics_weight = float(physics_weight)
        self.monotonicity_weight = float(monotonicity_weight)
        self.min_rate = float(min_rate)
        self.dropout_rate = float(dropout)

        # Psi(x,t): data-driven RUL trajectory.
        self.state_net = _make_tanh_mlp(enc_in + 1, hidden_dim, hidden_layers, dropout)
        # Gamma(x,t,u,du/dx,du/dt): learned positive degradation/RUL decay rate.
        gamma_input_dim = 2 * enc_in + 3
        self.rate_net = _make_tanh_mlp(gamma_input_dim, hidden_dim, hidden_layers, dropout)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("KnowledgeDataPINNRUL expects x with shape (batch, time, features)")
        if x.shape[0] < 1 or x.shape[1] != self.seq_len or x.shape[2] != self.enc_in:
            raise ValueError(
                f"KnowledgeDataPINNRUL expects (batch, {self.seq_len}, {self.enc_in}), got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x) or not torch.isfinite(x).all().item():
            raise ValueError("PINN inputs must be finite floating-point tensors")

        if x_mark is None:
            times = torch.linspace(0.0, 1.0, self.seq_len, device=x.device, dtype=x.dtype)
            times = times.view(1, self.seq_len, 1).expand(x.shape[0], -1, -1)
        else:
            if (
                not isinstance(x_mark, torch.Tensor)
                or x_mark.is_complex()
                or x_mark.ndim != 3
                or x_mark.shape != (x.shape[0], self.seq_len, 1)
            ):
                raise ValueError("x_mark must have shape (batch, time, 1) and contain the time coordinate")
            times = x_mark.to(device=x.device, dtype=x.dtype)
            if not torch.isfinite(times).all().item():
                raise ValueError("x_mark time coordinates must be finite")
            if torch.any(times[:, 1:, 0] <= times[:, :-1, 0]).item():
                raise ValueError("x_mark time coordinates must be strictly increasing in every window")
        return x, times

    def _state_trajectory(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        coordinates = torch.cat((x, times), dim=-1)
        return F.softplus(self.state_net(coordinates))

    def predict_trajectory(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return nonnegative RUL estimates at every input time step."""
        values, times = self._validate_inputs(x, x_mark)
        return self._state_trajectory(values, times)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        """Predict RUL at the final time step of each window."""
        return self.predict_trajectory(x, x_mark).squeeze(-1)[:, -1]

    def physics_residual(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the decreasing-RUL governing-rate residual at every time step.

        The paper's wear equation is du/dt = Gamma(...), with a positive
        wear rate. RUL is the remaining-life complement, so this adapter uses
        d(RUL)/dt + Gamma(...)=0 while retaining Gamma > 0 by construction.
        """
        values, times = self._validate_inputs(x, x_mark)
        with torch.enable_grad():
            coordinates = torch.cat((values, times), dim=-1).detach().requires_grad_(True)
            state = F.softplus(self.state_net(coordinates))
            derivatives = torch.autograd.grad(
                state.sum(),
                coordinates,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            feature_derivatives = derivatives[..., : self.enc_in]
            time_derivative = derivatives[..., self.enc_in : self.enc_in + 1]
            gamma_input = torch.cat(
                (coordinates, state, feature_derivatives, time_derivative),
                dim=-1,
            )
            positive_rate = F.softplus(self.rate_net(gamma_input)) + self.min_rate
            return time_derivative + positive_rate

    def loss_components(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        *,
        x_mark: torch.Tensor | None = None,
        loss: str = "mse",
    ) -> dict[str, torch.Tensor]:
        """Return data, physics, monotonicity and total training losses."""
        values, times = self._validate_inputs(x, x_mark)
        if (
            not isinstance(targets, torch.Tensor)
            or not torch.is_floating_point(targets)
            or targets.shape != (values.shape[0],)
        ):
            raise ValueError("targets must be floating-point with shape (batch,) and contain final-step RUL labels")
        labels = targets.to(device=values.device, dtype=values.dtype)
        if not torch.isfinite(labels).all().item() or torch.any(labels < 0).item():
            raise ValueError("RUL targets must be finite and nonnegative")
        if loss not in {"mse", "mae", "huber"}:
            raise ValueError("loss must be mse, mae or huber")

        coordinates = torch.cat((values, times), dim=-1).detach().requires_grad_(True)
        state = F.softplus(self.state_net(coordinates))
        derivatives = torch.autograd.grad(
            state.sum(),
            coordinates,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        feature_derivatives = derivatives[..., : self.enc_in]
        time_derivative = derivatives[..., self.enc_in : self.enc_in + 1]
        gamma_input = torch.cat((coordinates, state, feature_derivatives, time_derivative), dim=-1)
        positive_rate = F.softplus(self.rate_net(gamma_input)) + self.min_rate
        residual = time_derivative + positive_rate

        prediction = state[:, -1, 0]
        if loss == "mse":
            data_loss = F.mse_loss(prediction, labels)
        elif loss == "mae":
            data_loss = F.l1_loss(prediction, labels)
        else:
            data_loss = F.huber_loss(prediction, labels)

        # RUL must not increase as time advances within a window.
        monotonicity_loss = F.relu(state[:, 1:, 0] - state[:, :-1, 0]).mean()
        physics_loss = residual.square().mean()
        total = (
            data_loss
            + self.physics_weight * physics_loss
            + self.monotonicity_weight * monotonicity_loss
        )
        return {
            "total": total,
            "data": data_loss,
            "physics": physics_loss,
            "monotonicity": monotonicity_loss,
        }

    def training_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        *,
        x_mark: torch.Tensor | None = None,
        loss: str = "mse",
    ) -> torch.Tensor:
        """RUL data loss plus learned-rate PDE residual and monotonicity penalties."""
        return self.loss_components(x, targets, x_mark=x_mark, loss=loss)["total"]

    def loss_step(self, batch: RULBatch, loss: str = "mse") -> torch.Tensor:
        if batch.targets is None:
            raise ValueError("RULBatch.targets is required for loss_step")
        return self.training_loss(batch.x, batch.targets, x_mark=batch.x_mark, loss=loss)


__all__ = ["KnowledgeDataPINNRUL"]
