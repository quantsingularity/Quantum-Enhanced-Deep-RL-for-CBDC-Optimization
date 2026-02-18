"""
Unit tests for SAC agent.
"""

import pytest
import torch
import numpy as np
from models.sac_agent import SACAgent


def test_sac_creation():
    """Test SAC agent initialization."""
    agent = SACAgent(
        state_dim=8,
        action_dim=3,
        device="cpu",
        use_quantum_critic=False,
    )

    assert agent.state_dim == 8
    assert agent.action_dim == 3


def test_sac_select_action():
    """Test action selection."""
    agent = SACAgent(
        state_dim=8,
        action_dim=3,
        device="cpu",
    )

    state = np.random.randn(8)
    action = agent.select_action(state, deterministic=False)

    assert action.shape == (3,)
    assert np.all(action >= -1) and np.all(action <= 1)


def test_sac_update():
    """Test SAC update."""
    agent = SACAgent(
        state_dim=8,
        action_dim=3,
        device="cpu",
    )

    batch_size = 32
    states = torch.randn(batch_size, 8)
    actions = torch.randn(batch_size, 3)
    rewards = torch.randn(batch_size, 1)
    next_states = torch.randn(batch_size, 8)
    dones = torch.zeros(batch_size, 1)

    losses = agent.update(states, actions, rewards, next_states, dones)

    assert "critic_loss" in losses
    assert "actor_loss" in losses
    assert "alpha" in losses


def test_quantum_sac_creation():
    """Test quantum SAC creation."""
    quantum_config = {
        "embedding_dim": 128,
        "n_qubits": 4,
        "n_vqc_layers": 2,
        "quantum_output_dim": 64,
        "output_dims": [128, 64],
        "quantum_backend": "default.qubit",
        "enable_zne": False,
    }

    agent = SACAgent(
        state_dim=8,
        action_dim=3,
        device="cpu",
        use_quantum_critic=True,
        quantum_config=quantum_config,
    )

    assert agent.use_quantum_critic


def test_sac_save_load(tmp_path):
    """Test model save and load."""
    agent = SACAgent(
        state_dim=8,
        action_dim=3,
        device="cpu",
    )

    # Save
    save_path = tmp_path / "test_agent.pt"
    agent.save(str(save_path))

    assert save_path.exists()

    # Load
    loaded_agent = SACAgent.load(
        str(save_path),
        state_dim=8,
        action_dim=3,
        device="cpu",
    )

    # Test loaded agent
    state = np.random.randn(8)
    action = loaded_agent.select_action(state)

    assert action.shape == (3,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
