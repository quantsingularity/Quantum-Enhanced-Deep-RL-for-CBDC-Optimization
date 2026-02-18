"""
Quantum-enhanced critic network for SAC.
Integrates VQC into critic architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from models.vqc import HybridQuantumClassical


class QuantumCritic(nn.Module):
    """
    Twin Q-network critic with quantum enhancement.
    Replaces classical MLP with hybrid quantum-classical architecture.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        n_qubits: int = 4,
        n_vqc_layers: int = 2,
        quantum_output_dim: int = 64,
        output_dims: Tuple[int, ...] = (128, 64),
        quantum_backend: str = "default.qubit",
        enable_zne: bool = False,
        activation: str = "relu",
    ):
        """
        Initialize quantum critic.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            embedding_dim: Classical embedding dimension
            n_qubits: Number of qubits
            n_vqc_layers: Number of VQC layers
            quantum_output_dim: Quantum layer output dimension
            output_dims: Post-quantum layer dimensions
            quantum_backend: Quantum backend
            enable_zne: Enable Zero Noise Extrapolation
            activation: Activation function
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits

        # Select activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Q1 network with quantum layer
        # Classical embedding
        self.q1_embedding = nn.Linear(state_dim + action_dim, embedding_dim)

        # Quantum layer
        self.q1_quantum = HybridQuantumClassical(
            input_dim=embedding_dim,
            n_qubits=n_qubits,
            n_vqc_layers=n_vqc_layers,
            output_dim=quantum_output_dim,
            quantum_backend=quantum_backend,
            enable_zne=enable_zne,
        )

        # Classical post-processing
        q1_post_layers = []
        prev_dim = quantum_output_dim

        for dim in output_dims:
            q1_post_layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim

        q1_post_layers.append(nn.Linear(prev_dim, 1))
        self.q1_post_network = nn.ModuleList(q1_post_layers)

        # Q2 network with quantum layer
        # Classical embedding
        self.q2_embedding = nn.Linear(state_dim + action_dim, embedding_dim)

        # Quantum layer
        self.q2_quantum = HybridQuantumClassical(
            input_dim=embedding_dim,
            n_qubits=n_qubits,
            n_vqc_layers=n_vqc_layers,
            output_dim=quantum_output_dim,
            quantum_backend=quantum_backend,
            enable_zne=enable_zne,
        )

        # Classical post-processing
        q2_post_layers = []
        prev_dim = quantum_output_dim

        for dim in output_dims:
            q2_post_layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim

        q2_post_layers.append(nn.Linear(prev_dim, 1))
        self.q2_post_network = nn.ModuleList(q2_post_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classical weights."""
        # Q1 embedding
        nn.init.orthogonal_(self.q1_embedding.weight, gain=np.sqrt(2))
        nn.init.constant_(self.q1_embedding.bias, 0.0)

        # Q1 post-processing
        for layer in self.q1_post_network[:-1]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.q1_post_network[-1].weight, gain=1.0)
        nn.init.constant_(self.q1_post_network[-1].bias, 0.0)

        # Q2 embedding
        nn.init.orthogonal_(self.q2_embedding.weight, gain=np.sqrt(2))
        nn.init.constant_(self.q2_embedding.bias, 0.0)

        # Q2 post-processing
        for layer in self.q2_post_network[:-1]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.q2_post_network[-1].weight, gain=1.0)
        nn.init.constant_(self.q2_post_network[-1].bias, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both quantum Q-networks.

        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)

        Returns:
            Tuple of (q1_value, q2_value)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Q1 forward
        q1_embed = self.activation(self.q1_embedding(x))
        q1_quantum = self.q1_quantum(q1_embed)

        q1 = q1_quantum
        for i, layer in enumerate(self.q1_post_network):
            q1 = layer(q1)
            if i < len(self.q1_post_network) - 1:
                q1 = self.activation(q1)

        # Q2 forward
        q2_embed = self.activation(self.q2_embedding(x))
        q2_quantum = self.q2_quantum(q2_embed)

        q2 = q2_quantum
        for i, layer in enumerate(self.q2_post_network):
            q2 = layer(q2)
            if i < len(self.q2_post_network) - 1:
                q2 = self.activation(q2)

        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q1 value only."""
        x = torch.cat([state, action], dim=-1)

        q1_embed = self.activation(self.q1_embedding(x))
        q1_quantum = self.q1_quantum(q1_embed)

        q1 = q1_quantum
        for i, layer in enumerate(self.q1_post_network):
            q1 = layer(q1)
            if i < len(self.q1_post_network) - 1:
                q1 = self.activation(q1)

        return q1

    def q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q2 value only."""
        x = torch.cat([state, action], dim=-1)

        q2_embed = self.activation(self.q2_embedding(x))
        q2_quantum = self.q2_quantum(q2_embed)

        q2 = q2_quantum
        for i, layer in enumerate(self.q2_post_network):
            q2 = layer(q2)
            if i < len(self.q2_post_network) - 1:
                q2 = self.activation(q2)

        return q2
