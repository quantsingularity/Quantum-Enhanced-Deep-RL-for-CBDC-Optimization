"""
CBDC Liquidity Management Environment.
Gymnasium-compatible environment for bank liquidity management under CBDC stress.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
import yaml

from env.liquidity_dynamics import LiquidityDynamics
from env.constraints import LiquidityConstraints
from env.reward import RewardFunction


class CBDCLiquidityEnv(gym.Env):
    """
    CBDC Liquidity Management Environment.

    State Space (8D):
        - Current liquidity buffer
        - Short-term liabilities
        - Projected inflows (30-day)
        - Projected outflows (30-day)
        - Interbank funding rate
        - CBDC demand shock
        - Market volatility proxy
        - Previous action (borrow amount normalized)

    Action Space (3D - Continuous):
        - Borrow amount [0, max_borrow]
        - Liquid asset reallocation ratio [0, 1]
        - Emergency funding decision [0, 1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: Optional[str] = None,
        initial_liquidity: float = 1000000.0,
        initial_liabilities: float = 5000000.0,
        initial_capital: float = 2000000.0,
        lcr_threshold: float = 1.0,
        max_borrow: float = 2000000.0,
        max_episode_steps: int = 252,
        normalize_obs: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize CBDC liquidity environment.

        Args:
            config_path: Path to environment config YAML
            initial_liquidity: Initial liquidity buffer
            initial_liabilities: Initial short-term liabilities
            initial_capital: Initial bank capital
            lcr_threshold: LCR threshold (1.0 = 100%)
            max_borrow: Maximum borrowing amount
            max_episode_steps: Maximum steps per episode
            normalize_obs: Whether to normalize observations
            seed: Random seed
            **kwargs: Additional config parameters
        """
        super().__init__()

        # Load config if provided
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                kwargs.update(config)

        # Initial conditions
        self.initial_liquidity = initial_liquidity
        self.initial_liabilities = initial_liabilities
        self.initial_capital = initial_capital
        self.lcr_threshold = lcr_threshold
        self.max_borrow = max_borrow
        self.max_episode_steps = max_episode_steps
        self.normalize_obs = normalize_obs

        # Environment parameters
        self.dt = kwargs.get("dt", 1.0)
        self.obs_clip = kwargs.get("obs_clip", 10.0)
        self.emergency_funding_cost_multiplier = kwargs.get(
            "emergency_funding_cost_multiplier", 2.0
        )

        # Initialize dynamics
        self.dynamics = LiquidityDynamics(
            inflow_mean=kwargs.get("inflow_mean", 0.0001),
            inflow_volatility=kwargs.get("inflow_volatility", 0.02),
            outflow_mean=kwargs.get("outflow_mean", 0.0001),
            outflow_volatility=kwargs.get("outflow_volatility", 0.02),
            cbdc_jump_probability=kwargs.get("cbdc_jump_probability", 0.05),
            cbdc_jump_mean=kwargs.get("cbdc_jump_mean", -0.1),
            cbdc_jump_std=kwargs.get("cbdc_jump_std", 0.05),
            funding_rate_mean=kwargs.get("base_funding_rate", 0.02),
            funding_rate_volatility=kwargs.get("funding_rate_volatility", 0.01),
            funding_rate_mean_reversion=kwargs.get("funding_rate_mean_reversion", 0.5),
            volatility_base=kwargs.get("volatility_base", 0.01),
            volatility_shock_multiplier=kwargs.get("volatility_shock_multiplier", 3.0),
            volatility_decay=kwargs.get("volatility_decay", 0.1),
            dt=self.dt,
            seed=seed,
        )

        # Initialize constraints
        self.constraints = LiquidityConstraints(
            lcr_threshold=lcr_threshold,
            capital_adequacy_ratio=kwargs.get("capital_adequacy_ratio", 0.08),
            max_borrow_fraction=kwargs.get("max_borrow_fraction", 0.5),
            lcr_penalty_weight=kwargs.get("lcr_penalty_weight", 100.0),
            shortfall_penalty_weight=kwargs.get("shortfall_penalty_weight", 500.0),
        )

        # Initialize reward function
        self.reward_fn = RewardFunction(
            cost_weight=kwargs.get("cost_weight", 1.0),
            lcr_penalty_weight=kwargs.get("lcr_penalty_weight", 100.0),
            shortfall_penalty_weight=kwargs.get("shortfall_penalty_weight", 500.0),
            stability_bonus_weight=kwargs.get("stability_bonus_weight", 0.1),
        )

        # Define spaces
        # State: [liquidity, liabilities, proj_inflow, proj_outflow,
        #         funding_rate, cbdc_shock, volatility, prev_action]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

        # Action: [borrow_amount, reallocation_ratio, emergency_funding]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # State variables
        self.liquidity = initial_liquidity
        self.liabilities = initial_liabilities
        self.capital = initial_capital
        self.prev_action = 0.0
        self.current_step = 0

        # Episode statistics
        self.episode_funding_cost = 0.0
        self.episode_lcr_violations = 0
        self.episode_liquidity_history = []

        # Normalization statistics (updated during training)
        self.obs_mean = np.zeros(8)
        self.obs_std = np.ones(8)

        # Seed
        self._seed = seed
        self.reset(seed=seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)

        # Reset dynamics
        self.dynamics.reset(seed=seed)

        # Reset state variables
        self.liquidity = self.initial_liquidity
        self.liabilities = self.initial_liabilities
        self.capital = self.initial_capital
        self.prev_action = 0.0
        self.current_step = 0

        # Reset episode statistics
        self.episode_funding_cost = 0.0
        self.episode_lcr_violations = 0
        self.episode_liquidity_history = [self.liquidity]

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: [borrow_amount_normalized, reallocation_ratio, emergency_flag]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Parse action
        borrow_normalized = np.clip(action[0], 0.0, 1.0)
        reallocation_ratio = np.clip(action[1], 0.0, 1.0)
        emergency_flag = np.clip(action[2], 0.0, 1.0)

        # Convert normalized borrow to actual amount
        borrow_amount = borrow_normalized * self.max_borrow

        # Check borrow limit
        _, borrow_amount = self.constraints.check_borrow_limit(
            borrow_amount, self.liabilities
        )

        # Compute funding cost
        base_cost = borrow_amount * self.dynamics.current_funding_rate * self.dt / 365.0
        emergency_cost = (
            borrow_amount
            * emergency_flag
            * self.dynamics.current_funding_rate
            * self.emergency_funding_cost_multiplier
            * self.dt
            / 365.0
        )
        funding_cost = base_cost + emergency_cost

        # Apply action: add borrowed funds
        self.liquidity += borrow_amount

        # Simulate liquidity dynamics
        inflow, outflow, cbdc_shock, funding_rate, volatility = self.dynamics.step(
            self.liquidity, self.liabilities
        )

        # Update liquidity
        self.liquidity += inflow - outflow + cbdc_shock

        # Apply reallocation (simplified - moves liquid assets)
        # Positive reallocation increases liquidity buffer
        reallocation_effect = reallocation_ratio * 0.1 * self.liquidity
        self.liquidity += reallocation_effect

        # Update liabilities (natural growth/decay)
        liability_change = self.liabilities * 0.0001 * self.dt
        self.liabilities += liability_change

        # Get projected flows
        proj_inflow, proj_outflow = self.dynamics.get_projected_flows(
            self.liquidity, horizon=30
        )
        net_outflow = max(proj_outflow - proj_inflow, 1.0)  # Avoid division by zero

        # Check constraints
        constraint_info = self.constraints.get_constraint_info(
            liquidity=self.liquidity,
            liabilities=self.liabilities,
            projected_outflows=net_outflow,
            capital=self.capital,
            risk_weighted_assets=self.liabilities * 0.5,  # Simplified RWA
        )

        # Compute reward
        reward_info = self.reward_fn.compute_reward(
            funding_cost=funding_cost,
            lcr_penalty=constraint_info["lcr_penalty"],
            shortfall_penalty=constraint_info["shortfall_penalty"],
            liquidity=self.liquidity,
            target_liquidity=self.initial_liquidity,
        )

        reward = reward_info["total_reward"]

        # Update episode statistics
        self.episode_funding_cost += funding_cost
        if not constraint_info["lcr_compliant"]:
            self.episode_lcr_violations += 1
        self.episode_liquidity_history.append(self.liquidity)

        # Store previous action
        self.prev_action = borrow_normalized

        # Check termination
        terminated = False
        if self.liquidity < 0:  # Bankruptcy
            terminated = True
            reward -= 1000.0  # Large penalty

        truncated = self.current_step >= self.max_episode_steps

        # Get next observation
        obs = self._get_observation()

        # Compile info
        info = self._get_info()
        info.update(
            {
                "funding_cost": funding_cost,
                "lcr": constraint_info["lcr"],
                "lcr_compliant": constraint_info["lcr_compliant"],
                "liquidity": self.liquidity,
                "reward_breakdown": reward_info,
                "constraint_info": constraint_info,
            }
        )

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Get projected flows
        proj_inflow, proj_outflow = self.dynamics.get_projected_flows(
            self.liquidity, horizon=30
        )

        obs = np.array(
            [
                self.liquidity,
                self.liabilities,
                proj_inflow,
                proj_outflow,
                self.dynamics.current_funding_rate,
                self.dynamics.cbdc_shock,
                self.dynamics.current_volatility,
                self.prev_action,
            ],
            dtype=np.float32,
        )

        # Normalize if enabled
        if self.normalize_obs:
            obs = self._normalize_obs(obs)

        return obs

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        # Simple normalization by scaling
        scales = np.array(
            [
                1e6,  # liquidity
                1e6,  # liabilities
                1e5,  # proj_inflow
                1e5,  # proj_outflow
                0.1,  # funding_rate
                1e6,  # cbdc_shock
                0.1,  # volatility
                1.0,  # prev_action
            ],
            dtype=np.float32,
        )

        normalized = obs / scales
        normalized = np.clip(normalized, -self.obs_clip, self.obs_clip)

        return normalized

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            "step": self.current_step,
            "episode_funding_cost": self.episode_funding_cost,
            "episode_lcr_violations": self.episode_lcr_violations,
            "liquidity": self.liquidity,
            "liabilities": self.liabilities,
            "funding_rate": self.dynamics.current_funding_rate,
            "volatility": self.dynamics.current_volatility,
        }

    def render(self, mode: str = "human") -> None:
        """Render environment state."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Liquidity: ${self.liquidity:,.2f}")
            print(f"Liabilities: ${self.liabilities:,.2f}")
            print(f"Funding Rate: {self.dynamics.current_funding_rate:.4f}")
            print(f"Volatility: {self.dynamics.current_volatility:.4f}")
            print(f"Episode Funding Cost: ${self.episode_funding_cost:,.2f}")
            print(f"LCR Violations: {self.episode_lcr_violations}")
            print("-" * 50)

    def close(self) -> None:
        """Clean up environment."""
