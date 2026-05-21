"""
Quantum-enhanced critic network for SAC.
Integrates HybridQuantumClassical into twin-Q architecture.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vqc import HybridQuantumClassical


class QuantumCritic(nn.Module):
    """
    Twin Q-network critic with hybrid quantum-classical architecture.
    Each Q-network: classical embedding → quantum layer → classical head.
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
        entanglement_type: str = "ring",
        enable_zne: bool = False,
        activation: str = "relu",
    ):
        """
        Initialize quantum critic.

        Args:
            state_dim: State space dimension.
            action_dim: Action space dimension.
            embedding_dim: Classical pre-embedding dimension.
            n_qubits: Number of qubits.
            n_vqc_layers: Number of VQC variational layers.
            quantum_output_dim: VQC output → classical head input dimension.
            output_dims: Hidden dimensions of the classical head.
            quantum_backend: PennyLane backend.
            entanglement_type: VQC entanglement topology.
            enable_zne: Enable Zero Noise Extrapolation.
            activation: Classical activation ('relu', 'tanh', 'elu').
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: '{activation}'")

        # ── Q1 ────────────────────────────────────────────────────────────────
        self.q1_embedding = nn.Linear(state_dim + action_dim, embedding_dim)
        self.q1_quantum = HybridQuantumClassical(
            input_dim=embedding_dim,
            n_qubits=n_qubits,
            n_vqc_layers=n_vqc_layers,
            output_dim=quantum_output_dim,
            quantum_backend=quantum_backend,
            entanglement_type=entanglement_type,
            enable_zne=enable_zne,
        )
        q1_post: list = []
        prev = quantum_output_dim
        for dim in output_dims:
            q1_post.append(nn.Linear(prev, dim))
            prev = dim
        q1_post.append(nn.Linear(prev, 1))
        self.q1_post_network = nn.ModuleList(q1_post)

        # ── Q2 ────────────────────────────────────────────────────────────────
        self.q2_embedding = nn.Linear(state_dim + action_dim, embedding_dim)
        self.q2_quantum = HybridQuantumClassical(
            input_dim=embedding_dim,
            n_qubits=n_qubits,
            n_vqc_layers=n_vqc_layers,
            output_dim=quantum_output_dim,
            quantum_backend=quantum_backend,
            entanglement_type=entanglement_type,
            enable_zne=enable_zne,
        )
        q2_post: list = []
        prev = quantum_output_dim
        for dim in output_dims:
            q2_post.append(nn.Linear(prev, dim))
            prev = dim
        q2_post.append(nn.Linear(prev, 1))
        self.q2_post_network = nn.ModuleList(q2_post)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for all classical linear layers."""
        for emb in (self.q1_embedding, self.q2_embedding):
            nn.init.orthogonal_(emb.weight, gain=np.sqrt(2))
            nn.init.constant_(emb.bias, 0.0)

        for net in (self.q1_post_network, self.q2_post_network):
            for layer in list(net)[:-1]:
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
            nn.init.orthogonal_(list(net)[-1].weight, gain=1.0)
            nn.init.constant_(list(net)[-1].bias, 0.0)

    def _q_forward(
        self,
        x: torch.Tensor,
        embedding: nn.Linear,
        quantum: HybridQuantumClassical,
        post_network: nn.ModuleList,
    ) -> torch.Tensor:
        """Shared forward logic for one Q-network."""
        h = self.activation(embedding(x))
        h = quantum(h)
        for i, layer in enumerate(post_network):
            h = layer(h)
            if i < len(post_network) - 1:
                h = self.activation(h)
        return h

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)

        Returns:
            (q1_value, q2_value) each of shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        q1 = self._q_forward(
            x, self.q1_embedding, self.q1_quantum, self.q1_post_network
        )
        q2 = self._q_forward(
            x, self.q2_embedding, self.q2_quantum, self.q2_post_network
        )
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q1 value only."""
        x = torch.cat([state, action], dim=-1)
        return self._q_forward(
            x, self.q1_embedding, self.q1_quantum, self.q1_post_network
        )

    def q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q2 value only."""
        x = torch.cat([state, action], dim=-1)
        return self._q_forward(
            x, self.q2_embedding, self.q2_quantum, self.q2_post_network
        )
