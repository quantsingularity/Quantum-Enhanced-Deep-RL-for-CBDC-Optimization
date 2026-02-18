"""
Actor network for SAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):
    """
    Gaussian actor network with tanh squashing.
    Outputs mean and log_std for Gaussian policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ):
        """
        Initialize actor network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            log_std_min: Minimum log std
            log_std_max: Maximum log std
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Select activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        # Small initialization for output layer
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor.

        Args:
            state: State tensor (batch_size, state_dim)
            deterministic: Whether to sample deterministically
            with_logprob: Whether to compute log probability

        Returns:
            Tuple of (action, log_prob)
        """
        # Forward through hidden layers
        x = state
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # Get mean and log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Sample action
        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            # Reparameterization trick
            action = normal.rsample()

        # Apply tanh squashing
        action_squashed = torch.tanh(action)

        # Compute log probability if needed
        if with_logprob:
            # Log prob of Gaussian
            log_prob = Normal(mean, std).log_prob(action).sum(dim=-1, keepdim=True)

            # Correction for tanh squashing
            # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
            log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6).sum(
                dim=-1, keepdim=True
            )
        else:
            log_prob = None

        return action_squashed, log_prob

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> np.ndarray:
        """
        Get action from state (numpy interface).

        Args:
            state: State array
            deterministic: Whether to act deterministically

        Returns:
            Action array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.forward(state_tensor, deterministic=deterministic)
            return action.cpu().numpy()[0]
