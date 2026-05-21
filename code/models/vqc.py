"""
Variational Quantum Circuit (VQC) for quantum-enhanced critic.
"""

from typing import List, Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


class VariationalQuantumCircuit(nn.Module):
    """
    Variational Quantum Circuit with parameterized layers.

    Architecture:
        1. Angle encoding (RY/RX/RZ rotations on inputs)
        2. Parameterized variational layers (RY + RZ per qubit, then entanglement)
        3. Measurement — Pauli-Z expectation on each qubit
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        rotation_type: str = "RY",
        entanglement_type: str = "ring",
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        shots: Optional[int] = None,
    ):
        """
        Initialize VQC.

        Args:
            n_qubits: Number of qubits.
            n_layers: Number of variational layers.
            rotation_type: Encoding rotation gate ('RY', 'RX', 'RZ').
            entanglement_type: Entanglement topology ('ring', 'full', 'linear').
            backend: PennyLane device backend.
            diff_method: Differentiation method for QNode.
            shots: Measurement shots (None = exact statevector simulation).
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.entanglement_type = entanglement_type

        self.dev = qml.device(backend, wires=n_qubits, shots=shots)

        # Each layer: 2 rotations (RY + RZ) per qubit
        self.n_params_per_layer = n_qubits * 2
        self.total_params = self.n_params_per_layer * n_layers

        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method=diff_method,
        )

    def _circuit(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Quantum circuit definition.

        Args:
            inputs: Encoded features, shape (n_qubits,), values in [-π, π].
            weights: Variational parameters, shape (n_layers, n_qubits, 2).

        Returns:
            List of Pauli-Z expectation tensors, one per qubit, values in [-1, 1].
        """
        # Angle encoding
        for i in range(self.n_qubits):
            angle = inputs[i]
            if self.rotation_type == "RY":
                qml.RY(angle, wires=i)
            elif self.rotation_type == "RX":
                qml.RX(angle, wires=i)
            else:  # "RZ"
                qml.RZ(angle, wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            self._apply_entanglement()

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def _apply_entanglement(self) -> None:
        """Apply CNOT entanglement gates according to topology."""
        if self.entanglement_type == "ring":
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        elif self.entanglement_type == "full":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
        elif self.entanglement_type == "linear":
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        else:
            raise ValueError(
                f"Unknown entanglement_type '{self.entanglement_type}'. "
                "Choose from: 'ring', 'full', 'linear'."
            )

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass through the VQC.

        Args:
            x: Input tensor (batch_size, n_qubits), values in [-π, π].
            weights: Variational weights (n_layers, n_qubits, 2).

        Returns:
            Quantum measurements (batch_size, n_qubits), values in [-1, 1].
        """
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            result = self.qnode(x[i], weights)
            outputs.append(
                torch.stack(
                    [
                        (
                            r
                            if isinstance(r, torch.Tensor)
                            else torch.tensor(float(r), dtype=torch.float32)
                        )
                        for r in result
                    ]
                )
            )
        return torch.stack(outputs)

    def init_weights(self, method: str = "uniform") -> torch.Tensor:
        """
        Initialize variational weights.

        Args:
            method: 'uniform' (random in [0, 2π]) or 'normal' (σ=0.1).

        Returns:
            Weight tensor of shape (n_layers, n_qubits, 2).
        """
        if method == "uniform":
            return (
                torch.rand(self.n_layers, self.n_qubits, 2, dtype=torch.float32)
                * 2
                * np.pi
            )
        if method == "normal":
            return (
                torch.randn(self.n_layers, self.n_qubits, 2, dtype=torch.float32) * 0.1
            )
        raise ValueError(
            f"Unknown weight init method '{method}'. Use 'uniform' or 'normal'."
        )


class ZeroNoiseExtrapolation:
    """Zero Noise Extrapolation (ZNE) for quantum error mitigation."""

    def __init__(self, scale_factors: Optional[List[float]] = None):
        """
        Initialize ZNE.

        Args:
            scale_factors: Noise scaling factors.  Defaults to [1.0, 1.5, 2.0].
        """
        self.scale_factors: List[float] = (
            scale_factors if scale_factors is not None else [1.0, 1.5, 2.0]
        )

    def extrapolate(
        self,
        vqc: VariationalQuantumCircuit,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply ZNE via Richardson (quadratic Lagrange) extrapolation to scale=0.

        Noise is approximated by scaling the variational weights.

        Args:
            vqc: VQC instance.
            inputs: Input tensor (batch_size, n_qubits).
            weights: VQC weights (n_layers, n_qubits, 2).

        Returns:
            Zero-noise extrapolated output.
        """
        results = [vqc(inputs, weights * scale) for scale in self.scale_factors]

        if len(self.scale_factors) >= 3:
            c = self.scale_factors
            y = results
            denom = (c[0] - c[1]) * (c[0] - c[2]) * (c[1] - c[2])
            w0 = c[1] * c[2] * (c[1] - c[2]) / denom
            w1 = -c[0] * c[2] * (c[0] - c[2]) / denom
            w2 = c[0] * c[1] * (c[0] - c[1]) / denom
            return w0 * y[0] + w1 * y[1] + w2 * y[2]

        # Fallback: simple mean over scale factors
        return torch.stack(results).mean(dim=0)


class HybridQuantumClassical(nn.Module):
    """Hybrid quantum-classical layer combining a VQC with classical linear layers."""

    def __init__(
        self,
        input_dim: int,
        n_qubits: int,
        n_vqc_layers: int,
        output_dim: int,
        quantum_backend: str = "default.qubit",
        entanglement_type: str = "ring",
        enable_zne: bool = False,
    ):
        """
        Initialize hybrid layer.

        Args:
            input_dim: Input feature dimension.
            n_qubits: Number of qubits.
            n_vqc_layers: Number of variational layers.
            output_dim: Classical post-processing output dimension.
            quantum_backend: PennyLane backend string.
            entanglement_type: Entanglement topology for the VQC.
            enable_zne: Enable Zero Noise Extrapolation.
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.enable_zne = enable_zne

        # Classical pre-processing: embed inputs into qubit angle space
        self.embedding = nn.Linear(input_dim, n_qubits)

        # Variational Quantum Circuit
        self.vqc = VariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_vqc_layers,
            backend=quantum_backend,
            entanglement_type=entanglement_type,
        )

        # VQC weights as learnable nn.Parameter
        self.vqc_weights = nn.Parameter(self.vqc.init_weights(method="uniform"))

        # Classical post-processing
        self.post_processing = nn.Linear(n_qubits, output_dim)

        if enable_zne:
            self.zne = ZeroNoiseExtrapolation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid layer.

        Args:
            x: Input tensor (batch_size, input_dim).

        Returns:
            Output tensor (batch_size, output_dim).
        """
        # Embed to qubit angle space in [-π, π]
        embedded = torch.tanh(self.embedding(x)) * np.pi

        if self.enable_zne:
            quantum_out = self.zne.extrapolate(self.vqc, embedded, self.vqc_weights)
        else:
            quantum_out = self.vqc(embedded, self.vqc_weights)

        return self.post_processing(quantum_out)
