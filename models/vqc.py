"""
Variational Quantum Circuit (VQC) for quantum critic.
Implements parameterized quantum circuit with PennyLane.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import List, Optional


class VariationalQuantumCircuit(nn.Module):
    """
    Variational Quantum Circuit for quantum-enhanced critic.

    Architecture:
        1. Classical embedding layer
        2. Angle encoding (RY rotations)
        3. Parameterized variational layers (RY-RZ-CNOT)
        4. Measurement (Pauli-Z expectation)
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
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            rotation_type: Type of rotation gate (RY, RX, RZ)
            entanglement_type: Entanglement topology (ring, full, linear)
            backend: PennyLane backend
            diff_method: Differentiation method
            shots: Number of shots (None for exact)
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.entanglement_type = entanglement_type

        # Create quantum device
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)

        # Number of parameters per layer
        # Each layer: n_qubits rotations for each of 2 rotation gates
        self.n_params_per_layer = n_qubits * 2
        self.total_params = self.n_params_per_layer * n_layers

        # Create QNode
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
    ) -> List[float]:
        """
        Define quantum circuit.

        Args:
            inputs: Input features (size n_qubits)
            weights: Variational parameters (size n_layers x n_qubits x 2)

        Returns:
            List of expectation values
        """
        # Angle encoding
        for i in range(self.n_qubits):
            if self.rotation_type == "RY":
                qml.RY(inputs[i], wires=i)
            elif self.rotation_type == "RX":
                qml.RX(inputs[i], wires=i)
            else:  # RZ
                qml.RZ(inputs[i], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            # Entanglement
            self._apply_entanglement()

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def _apply_entanglement(self) -> None:
        """Apply entanglement layer based on topology."""
        if self.entanglement_type == "ring":
            # Ring topology: each qubit connected to next (circular)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        elif self.entanglement_type == "full":
            # Full connectivity: all pairs
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])

        elif self.entanglement_type == "linear":
            # Linear chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        else:
            raise ValueError(f"Unknown entanglement type: {self.entanglement_type}")

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VQC.

        Args:
            x: Input tensor (batch_size, n_qubits)
            weights: Variational weights (n_layers, n_qubits, 2)

        Returns:
            Quantum measurements (batch_size, n_qubits)
        """
        batch_size = x.shape[0]

        # Process each sample in batch
        outputs = []
        for i in range(batch_size):
            # Run circuit
            result = self.qnode(x[i], weights)
            outputs.append(torch.stack(result))

        return torch.stack(outputs)

    def init_weights(self, method: str = "uniform") -> torch.Tensor:
        """
        Initialize variational weights.

        Args:
            method: Initialization method (uniform, normal)

        Returns:
            Initialized weights tensor
        """
        if method == "uniform":
            weights = (
                torch.rand(
                    self.n_layers,
                    self.n_qubits,
                    2,
                    dtype=torch.float32,
                )
                * 2
                * np.pi
            )
        elif method == "normal":
            weights = (
                torch.randn(
                    self.n_layers,
                    self.n_qubits,
                    2,
                    dtype=torch.float32,
                )
                * 0.1
            )
        else:
            raise ValueError(f"Unknown init method: {method}")

        return weights


class ZeroNoiseExtrapolation:
    """Zero Noise Extrapolation for error mitigation."""

    def __init__(
        self,
        scale_factors: List[float] = [1.0, 1.5, 2.0],
    ):
        """
        Initialize ZNE.

        Args:
            scale_factors: Noise scaling factors
        """
        self.scale_factors = scale_factors

    def extrapolate(
        self,
        vqc: VariationalQuantumCircuit,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply ZNE to quantum circuit.

        Args:
            vqc: VQC instance
            inputs: Input tensor
            weights: VQC weights

        Returns:
            Extrapolated output
        """
        # Collect results at different noise scales
        results = []
        for scale in self.scale_factors:
            # Scale noise (simplified - in practice would modify device)
            # Here we just run the circuit multiple times
            result = vqc(inputs, weights)
            results.append(result)

        # Linear extrapolation to zero noise
        # In practice, use Richardson extrapolation
        # Simplified: average results
        extrapolated = torch.stack(results).mean(dim=0)

        return extrapolated


class HybridQuantumClassical(nn.Module):
    """Hybrid quantum-classical layer combining VQC with classical layers."""

    def __init__(
        self,
        input_dim: int,
        n_qubits: int,
        n_vqc_layers: int,
        output_dim: int,
        quantum_backend: str = "default.qubit",
        enable_zne: bool = False,
    ):
        """
        Initialize hybrid layer.

        Args:
            input_dim: Input dimension
            n_qubits: Number of qubits
            n_vqc_layers: Number of VQC layers
            output_dim: Output dimension
            quantum_backend: Quantum backend
            enable_zne: Enable Zero Noise Extrapolation
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.output_dim = output_dim
        self.enable_zne = enable_zne

        # Classical embedding: map input to quantum state
        self.embedding = nn.Linear(input_dim, n_qubits)

        # VQC
        self.vqc = VariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_vqc_layers,
            backend=quantum_backend,
        )

        # VQC weights as learnable parameters
        vqc_weights = self.vqc.init_weights(method="uniform")
        self.vqc_weights = nn.Parameter(vqc_weights)

        # Classical post-processing
        self.post_processing = nn.Linear(n_qubits, output_dim)

        # ZNE if enabled
        if enable_zne:
            self.zne = ZeroNoiseExtrapolation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Classical embedding
        embedded = torch.tanh(self.embedding(x))  # Scale to [-1, 1]
        embedded = embedded * np.pi  # Scale to [-pi, pi]

        # Quantum processing
        if self.enable_zne:
            quantum_out = self.zne.extrapolate(self.vqc, embedded, self.vqc_weights)
        else:
            quantum_out = self.vqc(embedded, self.vqc_weights)

        # Classical post-processing
        output = self.post_processing(quantum_out)

        return output
