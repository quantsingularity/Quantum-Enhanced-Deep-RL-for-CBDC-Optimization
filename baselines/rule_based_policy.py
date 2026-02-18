"""
Rule-based baseline policy for liquidity management.
"""

import numpy as np

from env.cbdc_env import CBDCLiquidityEnv


class RuleBasedPolicy:
    """
    Simple rule-based policy for liquidity management.

    Strategy:
        - Borrow when LCR is below threshold
        - Borrow amount proportional to deficit
        - Use emergency funding when LCR critically low
        - Conservative reallocation
    """

    def __init__(
        self,
        lcr_target: float = 1.2,
        lcr_critical: float = 1.0,
        borrow_scale: float = 1.5,
        emergency_threshold: float = 0.95,
    ):
        """
        Initialize rule-based policy.

        Args:
            lcr_target: Target LCR level
            lcr_critical: Critical LCR level
            borrow_scale: Scaling factor for borrowing
            emergency_threshold: Threshold for emergency funding
        """
        self.lcr_target = lcr_target
        self.lcr_critical = lcr_critical
        self.borrow_scale = borrow_scale
        self.emergency_threshold = emergency_threshold

    def select_action(self, state: np.ndarray, env: CBDCLiquidityEnv) -> np.ndarray:
        """
        Select action based on rules.

        Args:
            state: Current state (normalized)
            env: Environment instance (for denormalization)

        Returns:
            Action array [borrow_normalized, reallocation_ratio, emergency_flag]
        """
        # Denormalize key state variables
        # State: [liquidity, liabilities, proj_inflow, proj_outflow,
        #         funding_rate, cbdc_shock, volatility, prev_action]

        scales = np.array([1e6, 1e6, 1e5, 1e5, 0.1, 1e6, 0.1, 1.0])
        state_denorm = state * scales

        liquidity = state_denorm[0]
        state_denorm[1]
        proj_inflow = state_denorm[2]
        proj_outflow = state_denorm[3]

        # Compute approximate LCR
        net_outflow = max(proj_outflow - proj_inflow, 1.0)
        current_lcr = liquidity / net_outflow if net_outflow > 0 else 10.0

        # Determine if emergency
        emergency_flag = 0.0
        if current_lcr < self.emergency_threshold:
            emergency_flag = 1.0

        # Compute borrow amount
        borrow_normalized = 0.0
        if current_lcr < self.lcr_target:
            # Need to borrow
            deficit = (self.lcr_target - current_lcr) * net_outflow
            borrow_amount = deficit * self.borrow_scale

            # Normalize to [0, 1]
            borrow_normalized = min(1.0, borrow_amount / env.max_borrow)

        # Reallocation strategy: moderate when LCR is low
        if current_lcr < self.lcr_critical:
            reallocation_ratio = 0.3  # Increase liquidity buffer
        else:
            reallocation_ratio = 0.1  # Minimal reallocation

        action = np.array(
            [borrow_normalized, reallocation_ratio, emergency_flag],
            dtype=np.float32,
        )

        return action


def evaluate_rule_based(
    env: CBDCLiquidityEnv,
    n_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """
    Evaluate rule-based policy.

    Args:
        env: Environment
        n_episodes: Number of episodes
        seed: Random seed

    Returns:
        Dictionary of evaluation metrics
    """
    policy = RuleBasedPolicy()

    episode_rewards = []
    episode_funding_costs = []
    episode_lcr_violations = []
    episode_lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        episode_reward = 0
        done = False

        while not done:
            action = policy.select_action(state, env)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_funding_costs.append(info["episode_funding_cost"])
        episode_lcr_violations.append(info["episode_lcr_violations"])
        episode_lengths.append(info["step"])

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_funding_cost": np.mean(episode_funding_costs),
        "std_funding_cost": np.std(episode_funding_costs),
        "lcr_violation_rate": np.mean([v > 0 for v in episode_lcr_violations]),
        "mean_lcr_violations": np.mean(episode_lcr_violations),
        "mean_episode_length": np.mean(episode_lengths),
    }


if __name__ == "__main__":
    import yaml

    # Load environment config
    with open("configs/environment.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    # Create environment
    env = CBDCLiquidityEnv(seed=42, **env_config)

    # Evaluate
    print("Evaluating rule-based policy...")
    metrics = evaluate_rule_based(env, n_episodes=100)

    print("\nRule-Based Policy Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    env.close()
