"""
Classical critic network for SAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class Critic(nn.Module):
    """
    Twin Q-network critic for SAC.
    Implements two Q-networks to reduce overestimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ):
        """
        Initialize critic network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Select activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build Q1 network
        q1_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1_network = nn.ModuleList(q1_layers)

        # Build Q2 network
        q2_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2_network = nn.ModuleList(q2_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for network in [self.q1_network, self.q2_network]:
            for layer in network[:-1]:
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

            # Output layer
            nn.init.orthogonal_(network[-1].weight, gain=1.0)
            nn.init.constant_(network[-1].bias, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.

        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)

        Returns:
            Tuple of (q1_value, q2_value)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Q1 forward
        q1 = x
        for i, layer in enumerate(self.q1_network):
            q1 = layer(q1)
            if i < len(self.q1_network) - 1:
                q1 = self.activation(q1)

        # Q2 forward
        q2 = x
        for i, layer in enumerate(self.q2_network):
            q2 = layer(q2)
            if i < len(self.q2_network) - 1:
                q2 = self.activation(q2)

        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q1 value only."""
        x = torch.cat([state, action], dim=-1)

        for i, layer in enumerate(self.q1_network):
            x = layer(x)
            if i < len(self.q1_network) - 1:
                x = self.activation(x)

        return x

    def q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q2 value only."""
        x = torch.cat([state, action], dim=-1)

        for i, layer in enumerate(self.q2_network):
            x = layer(x)
            if i < len(self.q2_network) - 1:
                x = self.activation(x)

        return x
