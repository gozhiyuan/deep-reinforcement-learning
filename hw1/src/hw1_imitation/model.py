"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # The MLP predicts an entire action chunk at once.
        # Its final layer is flat, so the raw output size is:
        #   chunk_size * action_dim
        # per batch element.
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Input shape:
        #   state: (batch_size, state_dim)
        #
        # Raw network output shape:
        #   pred: (batch_size, chunk_size * action_dim)
        #
        # We reshape it into one action vector per future timestep:
        #   (batch_size, chunk_size, action_dim)
        #
        # The -1 tells PyTorch to infer the batch dimension automatically.
        pred = self.net(state)
        return pred.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # Both tensors have shape:
        #   (batch_size, chunk_size, action_dim)
        # so we can apply elementwise MSE directly.
        pred_chunk = self.forward(state)
        return nn.functional.mse_loss(pred_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # num_steps is unused for the MSE policy. It is only present so this
        # method matches the shared policy interface used by evaluation code.
        del num_steps
        return self.forward(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # The flow network predicts a chunk-shaped velocity field from:
        #   - the current state
        #   - the current noisy action chunk
        #   - the flow timestep tau
        #
        # We keep the same MLP style as MSEPolicy, but widen the input to include
        # the flattened chunk and one scalar tau per batch element.
        layers: list[nn.Module] = []
        input_dim = state_dim + (chunk_size * action_dim) + 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        # Shapes:
        #   state:        (batch_size, state_dim)
        #   action_chunk: (batch_size, chunk_size, action_dim)
        #   tau:          (batch_size,) or (batch_size, 1)
        #
        # The network predicts a velocity tensor with the same chunk shape:
        #   (batch_size, chunk_size, action_dim)
        batch_size = state.shape[0]
        tau = tau.view(batch_size, 1)
        flat_chunk = action_chunk.view(batch_size, self.chunk_size * self.action_dim)
        net_input = torch.cat((state, flat_chunk, tau), dim=1)  # (batch_size, state_dim + chunk_size * action_dim + 1)
        pred = self.net(net_input)  # (batch_size, chunk_size * action_dim)
        return pred.view(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        # Sample one flow timestep and one Gaussian noise chunk per training example.
        tau = torch.rand(batch_size, device=state.device, dtype=state.dtype)  # Range: [0, 1) (uniformly random, continuous)
        noise = torch.randn_like(action_chunk) # Shape: (batch_size, chunk_size, action_dim) Approximately (-∞, ∞), but practically most values are in [-3, 3] (standard normal distribution: mean 0, standard deviation 1).

        # Interpolate between noise and the clean chunk:
        #   A_{t,tau} = tau * A_t + (1 - tau) * A_{t,0}
        # Shape: (batch_size, chunk_size, action_dim) A blend of the clean action chunk and noise, where tau controls the mix. When tau=0, it's all noise; when tau=1, it's all clean chunk. 
        # For values in between, it's a weighted average of the two. 
        # Broadcasting: tau_chunk (shape (batch_size, 1, 1)) broadcasts to match action_chunk (shape (batch_size, chunk_size, action_dim)
        tau_chunk = tau.view(batch_size, 1, 1)
        interpolated_chunk = tau_chunk * action_chunk + (1.0 - tau_chunk) * noise 

        # For the straight-line interpolation path, the target velocity is constant:
        #   dA_{t,tau}/dtau = A_t - A_{t,0}
        # Result: target_velocity differs across batch elements because both action_chunk and noise are different for each one.
        # Within a batch element: target_velocity is the same across all chunk_size and action_dim dimensions 
        # (it's a fixed tensor per batch element, as we discussed—constant velocity for the linear path).
        target_velocity = action_chunk - noise
        pred_velocity = self.forward(state, interpolated_chunk, tau)
        return nn.functional.mse_loss(pred_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        batch_size = state.shape[0]
        dt = 1.0 / num_steps

        # Start the ODE rollout from Gaussian noise with the final chunk shape.
        action_chunk = torch.randn(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
        )

        # Euler integration over flow time tau in [0, 1).
        # shape (batch_size,) where every element is the same scalar value step * dt.
        # step * dt computes the current flow time tau for that Euler step.
        # dt = 1.0 / num_steps is the step size along the interval [0, 1).
        # step goes from 0 to num_steps - 1.
        # So step * dt gives the current time point: 0, dt, 2*dt, ..., (num_steps-1)*dt
        for step in range(num_steps):
            tau = torch.full(
                (batch_size,),
                fill_value=step * dt,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self.forward(state, action_chunk, tau)
            action_chunk = action_chunk + dt * velocity

        return action_chunk


PolicyType = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
