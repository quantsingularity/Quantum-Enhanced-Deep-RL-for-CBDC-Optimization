"""
Reward function for CBDC liquidity management.
"""

import numpy as np
from typing import Dict


class RewardFunction:
    """Reward function for liquidity management."""

    def __init__(
        self,
        cost_weight: float = 1.0,
        lcr_penalty_weight: float = 100.0,
        shortfall_penalty_weight: float = 500.0,
        stability_bonus_weight: float = 0.1,
    ):
        """
        Initialize reward function.

        Args:
            cost_weight: Weight for funding cost
            lcr_penalty_weight: Weight for LCR penalties
            shortfall_penalty_weight: Weight for shortfall penalties
            stability_bonus_weight: Weight for stability bonus
        """
        self.cost_weight = cost_weight
        self.lcr_penalty_weight = lcr_penalty_weight
        self.shortfall_penalty_weight = shortfall_penalty_weight
        self.stability_bonus_weight = stability_bonus_weight

    def compute_reward(
        self,
        funding_cost: float,
        lcr_penalty: float,
        shortfall_penalty: float,
        liquidity: float,
        target_liquidity: float,
    ) -> Dict[str, float]:
        """
        Compute reward components.

        Reward = -funding_cost - lcr_penalty - shortfall_penalty + stability_bonus

        Args:
            funding_cost: Cost of funding
            lcr_penalty: Penalty for LCR violation
            shortfall_penalty: Penalty for liquidity shortfall
            liquidity: Current liquidity
            target_liquidity: Target liquidity level

        Returns:
            Dictionary with reward components and total reward
        """
        # Negative cost (we want to minimize)
        cost_term = -self.cost_weight * funding_cost

        # Penalties
        lcr_term = -self.lcr_penalty_weight * lcr_penalty
        shortfall_term = -self.shortfall_penalty_weight * shortfall_penalty

        # Stability bonus for maintaining buffer
        if liquidity > 0:
            buffer_ratio = min(1.0, liquidity / target_liquidity)
            stability_bonus = self.stability_bonus_weight * buffer_ratio
        else:
            stability_bonus = 0.0

        # Total reward
        total_reward = cost_term + lcr_term + shortfall_term + stability_bonus

        return {
            "cost_term": cost_term,
            "lcr_term": lcr_term,
            "shortfall_term": shortfall_term,
            "stability_bonus": stability_bonus,
            "total_reward": total_reward,
            "funding_cost": funding_cost,
        }

    def normalize_reward(self, reward: float, clip: float = 10.0) -> float:
        """
        Normalize and clip reward.

        Args:
            reward: Raw reward
            clip: Clipping threshold

        Returns:
            Normalized reward
        """
        return np.clip(reward / 1000.0, -clip, clip)
