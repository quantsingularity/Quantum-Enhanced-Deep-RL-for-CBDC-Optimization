"""
Soft Actor-Critic (SAC) agent implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from copy import deepcopy

from models.actor import Actor
from models.critic_classical import Critic
from models.critic_quantum import QuantumCritic


class SACAgent:
    """
    Soft Actor-Critic agent with support for both classical and quantum critics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_hidden_dims: Tuple[int, ...] = (256, 256),
        critic_hidden_dims: Tuple[int, ...] = (256, 256),
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        device: str = "cpu",
        use_quantum_critic: bool = False,
        quantum_config: Optional[Dict] = None,
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            actor_hidden_dims: Actor hidden dimensions
            critic_hidden_dims: Critic hidden dimensions
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            lr_alpha: Temperature learning rate
            gamma: Discount factor
            tau: Polyak averaging coefficient
            alpha: Initial temperature
            auto_entropy_tuning: Enable automatic entropy tuning
            target_entropy: Target entropy (None for auto)
            device: Device (cpu/cuda)
            use_quantum_critic: Use quantum critic
            quantum_config: Quantum critic configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.auto_entropy_tuning = auto_entropy_tuning
        self.use_quantum_critic = use_quantum_critic

        # Initialize actor
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
        ).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize critic
        if use_quantum_critic:
            if quantum_config is None:
                quantum_config = {}

            self.critic = QuantumCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                embedding_dim=quantum_config.get("embedding_dim", 128),
                n_qubits=quantum_config.get("n_qubits", 4),
                n_vqc_layers=quantum_config.get("n_vqc_layers", 2),
                quantum_output_dim=quantum_config.get("quantum_output_dim", 64),
                output_dims=tuple(quantum_config.get("output_dims", [128, 64])),
                quantum_backend=quantum_config.get("quantum_backend", "default.qubit"),
                enable_zne=quantum_config.get("enable_zne", False),
            ).to(device)
        else:
            self.critic = Critic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=critic_hidden_dims,
            ).to(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Target critic
        self.critic_target = deepcopy(self.critic)

        # Freeze target network
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Entropy temperature
        if auto_entropy_tuning:
            if target_entropy is None:
                # Heuristic: -dim(A)
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy

            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha
            self.target_entropy = None
            self.log_alpha = None

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action from policy.

        Args:
            state: State array
            deterministic: Whether to act deterministically

        Returns:
            Action array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor(state_tensor, deterministic=deterministic)
            return action.cpu().numpy()[0]

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update SAC networks.

        Args:
            states: State batch
            actions: Action batch
            rewards: Reward batch
            next_states: Next state batch
            dones: Done mask batch

        Returns:
            Dictionary of losses
        """
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critic
        with torch.no_grad():
            # Sample action from current policy
            next_actions, next_log_probs = self.actor(next_states)

            # Compute target Q-value
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - self.alpha * next_log_probs

            # Bellman backup
            target_value = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss (MSE)
        critic_loss = nn.MSELoss()(current_q1, target_value) + nn.MSELoss()(
            current_q2, target_value
        )

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Gradient clipping for stability (especially for quantum)
        if self.use_quantum_critic:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)

        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss
        actor_loss = (self.alpha * log_probs - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # Update target networks (Polyak averaging)
        self._update_target_network()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "q_value": q_new.mean().item(),
            "log_prob": log_probs.mean().item(),
        }

    def _update_target_network(self):
        """Update target network using Polyak averaging."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """Save agent to file."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "alpha": self.alpha,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str, state_dim: int, action_dim: int, device: str = "cpu", **kwargs
    ):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=device)

        # Create agent
        agent = cls(state_dim=state_dim, action_dim=action_dim, device=device, **kwargs)

        # Load state dicts
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        agent.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )

        if agent.auto_entropy_tuning:
            agent.log_alpha = checkpoint["log_alpha"]
            agent.alpha = checkpoint["alpha"]

        return agent
