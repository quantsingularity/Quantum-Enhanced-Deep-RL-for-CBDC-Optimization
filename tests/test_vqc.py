"""
Unit tests for Variational Quantum Circuit.
"""

import pytest
import torch
import numpy as np
from models.vqc import VariationalQuantumCircuit, HybridQuantumClassical


def test_vqc_creation():
    """Test VQC initialization."""
    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)

    assert vqc.n_qubits == 4
    assert vqc.n_layers == 2
    assert vqc.total_params == 16  # 4 qubits * 2 rotations * 2 layers


def test_vqc_forward():
    """Test VQC forward pass."""
    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)

    batch_size = 2
    inputs = torch.randn(batch_size, 4) * np.pi
    weights = vqc.init_weights()

    outputs = vqc(inputs, weights)

    assert outputs.shape == (batch_size, 4)
    assert torch.all(outputs >= -1) and torch.all(outputs <= 1)  # Pauli-Z expectations


def test_vqc_gradient():
    """Test VQC gradient computation."""
    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)

    inputs = torch.randn(1, 4) * np.pi
    weights = vqc.init_weights()
    weights.requires_grad = True

    outputs = vqc(inputs, weights)
    loss = outputs.sum()
    loss.backward()

    assert weights.grad is not None
    assert weights.grad.shape == weights.shape


def test_hybrid_layer():
    """Test hybrid quantum-classical layer."""
    hybrid = HybridQuantumClassical(
        input_dim=8,
        n_qubits=4,
        n_vqc_layers=2,
        output_dim=16,
    )

    batch_size = 2
    inputs = torch.randn(batch_size, 8)

    outputs = hybrid(inputs)

    assert outputs.shape == (batch_size, 16)


def test_different_entanglement():
    """Test different entanglement topologies."""
    for entanglement in ["ring", "linear", "full"]:
        vqc = VariationalQuantumCircuit(
            n_qubits=4, n_layers=1, entanglement_type=entanglement
        )

        inputs = torch.randn(1, 4) * np.pi
        weights = vqc.init_weights()

        outputs = vqc(inputs, weights)

        assert outputs.shape == (1, 4)


def test_vqc_determinism():
    """Test VQC determinism."""
    torch.manual_seed(42)

    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)

    inputs = torch.randn(1, 4) * np.pi
    weights = vqc.init_weights()

    output1 = vqc(inputs, weights)
    output2 = vqc(inputs, weights)

    torch.testing.assert_close(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
