"""
Unit tests for CBDC environment.
"""

import pytest
import numpy as np
from env.cbdc_env import CBDCLiquidityEnv


def test_environment_creation():
    """Test environment initialization."""
    env = CBDCLiquidityEnv(seed=42)

    assert env.observation_space.shape == (8,)
    assert env.action_space.shape == (3,)

    env.close()


def test_environment_reset():
    """Test environment reset."""
    env = CBDCLiquidityEnv(seed=42)

    obs, info = env.reset()

    assert obs.shape == (8,)
    assert isinstance(info, dict)
    assert "step" in info
    assert info["step"] == 0

    env.close()


def test_environment_step():
    """Test environment step."""
    env = CBDCLiquidityEnv(seed=42)

    obs, _ = env.reset()
    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (8,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


def test_episode_completion():
    """Test full episode."""
    env = CBDCLiquidityEnv(seed=42, max_episode_steps=10)

    obs, _ = env.reset()
    done = False
    steps = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps <= 10

    env.close()


def test_lcr_computation():
    """Test LCR computation in environment."""
    env = CBDCLiquidityEnv(seed=42)

    obs, _ = env.reset()

    # Take action
    action = np.array([0.5, 0.5, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)

    assert "lcr" in info
    assert "lcr_compliant" in info

    env.close()


def test_deterministic_behavior():
    """Test environment determinism with same seed."""
    env1 = CBDCLiquidityEnv(seed=42)
    env2 = CBDCLiquidityEnv(seed=42)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()

    np.testing.assert_array_almost_equal(obs1, obs2)

    action = np.array([0.5, 0.5, 0.0])

    next_obs1, reward1, _, _, _ = env1.step(action)
    next_obs2, reward2, _, _, _ = env2.step(action)

    np.testing.assert_array_almost_equal(next_obs1, next_obs2)
    assert abs(reward1 - reward2) < 1e-6

    env1.close()
    env2.close()


def test_constraints():
    """Test constraint enforcement."""
    env = CBDCLiquidityEnv(seed=42)

    obs, _ = env.reset()

    # Test action clipping
    action = np.array([2.0, 2.0, 2.0])  # Out of bounds
    obs, reward, terminated, truncated, info = env.step(action)

    # Should not crash
    assert obs.shape == (8,)

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
