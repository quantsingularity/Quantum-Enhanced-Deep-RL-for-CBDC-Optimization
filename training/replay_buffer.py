"""
Experience replay buffer for SAC.
"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL.
    Stores transitions and samples random batches.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 1000000,
        device: str = "cpu",
    ):
        """
        Initialize replay buffer.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            max_size: Maximum buffer size
            device: Device for sampling
        """
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample random batch from buffer.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
