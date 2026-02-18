"""
Liquidity dynamics models for CBDC environment.
Implements stochastic processes for inflows, outflows, and CBDC shocks.
"""

import numpy as np
from typing import Tuple


class LiquidityDynamics:
    """Stochastic liquidity dynamics with CBDC shocks."""

    def __init__(
        self,
        inflow_mean: float = 0.0001,
        inflow_volatility: float = 0.02,
        outflow_mean: float = 0.0001,
        outflow_volatility: float = 0.02,
        cbdc_jump_probability: float = 0.05,
        cbdc_jump_mean: float = -0.1,
        cbdc_jump_std: float = 0.05,
        funding_rate_mean: float = 0.02,
        funding_rate_volatility: float = 0.01,
        funding_rate_mean_reversion: float = 0.5,
        volatility_base: float = 0.01,
        volatility_shock_multiplier: float = 3.0,
        volatility_decay: float = 0.1,
        dt: float = 1.0,
        seed: int = None,
    ):
        """
        Initialize liquidity dynamics.

        Args:
            inflow_mean: Mean drift for inflows (GBM)
            inflow_volatility: Volatility for inflows
            outflow_mean: Mean drift for outflows (GBM)
            outflow_volatility: Volatility for outflows
            cbdc_jump_probability: Daily probability of CBDC shock
            cbdc_jump_mean: Mean CBDC jump size
            cbdc_jump_std: Std of CBDC jump size
            funding_rate_mean: Long-term mean funding rate
            funding_rate_volatility: Funding rate volatility (OU)
            funding_rate_mean_reversion: Mean reversion speed
            volatility_base: Base market volatility
            volatility_shock_multiplier: Multiplier during shocks
            volatility_decay: Volatility decay rate
            dt: Time step
            seed: Random seed
        """
        self.inflow_mean = inflow_mean
        self.inflow_volatility = inflow_volatility
        self.outflow_mean = outflow_mean
        self.outflow_volatility = outflow_volatility
        self.cbdc_jump_probability = cbdc_jump_probability
        self.cbdc_jump_mean = cbdc_jump_mean
        self.cbdc_jump_std = cbdc_jump_std
        self.funding_rate_mean = funding_rate_mean
        self.funding_rate_volatility = funding_rate_volatility
        self.funding_rate_mean_reversion = funding_rate_mean_reversion
        self.volatility_base = volatility_base
        self.volatility_shock_multiplier = volatility_shock_multiplier
        self.volatility_decay = volatility_decay
        self.dt = dt

        self.rng = np.random.RandomState(seed)

        # State variables
        self.current_funding_rate = funding_rate_mean
        self.current_volatility = volatility_base
        self.cbdc_shock = 0.0

    def reset(self, seed: int = None) -> None:
        """Reset dynamics to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_funding_rate = self.funding_rate_mean
        self.current_volatility = self.volatility_base
        self.cbdc_shock = 0.0

    def step(
        self, current_liquidity: float, current_liabilities: float
    ) -> Tuple[float, float, float, float, float]:
        """
        Simulate one step of liquidity dynamics.

        Args:
            current_liquidity: Current liquidity buffer
            current_liabilities: Current liabilities

        Returns:
            Tuple of (inflow, outflow, cbdc_shock, funding_rate, volatility)
        """
        # Geometric Brownian Motion for inflows
        inflow = current_liquidity * (
            self.inflow_mean * self.dt
            + self.inflow_volatility * np.sqrt(self.dt) * self.rng.randn()
        )
        inflow = max(0, inflow)  # No negative inflows

        # Geometric Brownian Motion for outflows
        outflow = current_liquidity * (
            self.outflow_mean * self.dt
            + self.outflow_volatility * np.sqrt(self.dt) * self.rng.randn()
        )
        outflow = max(0, outflow)  # No negative outflows

        # Jump-diffusion for CBDC shock
        if self.rng.rand() < self.cbdc_jump_probability:
            jump_size = self.rng.normal(self.cbdc_jump_mean, self.cbdc_jump_std)
            self.cbdc_shock = jump_size * current_liabilities
            # Increase volatility after shock
            self.current_volatility = min(
                self.volatility_base * self.volatility_shock_multiplier,
                self.current_volatility + 0.02,
            )
        else:
            self.cbdc_shock = 0.0

        # Decay volatility
        self.current_volatility = self.volatility_base + (
            self.current_volatility - self.volatility_base
        ) * np.exp(-self.volatility_decay * self.dt)

        # Ornstein-Uhlenbeck process for funding rate
        dr = (
            self.funding_rate_mean_reversion
            * (self.funding_rate_mean - self.current_funding_rate)
            * self.dt
            + self.funding_rate_volatility * np.sqrt(self.dt) * self.rng.randn()
        )
        self.current_funding_rate += dr
        self.current_funding_rate = max(0.001, self.current_funding_rate)  # Floor

        return (
            inflow,
            outflow,
            self.cbdc_shock,
            self.current_funding_rate,
            self.current_volatility,
        )

    def get_projected_flows(
        self, current_liquidity: float, horizon: int = 30
    ) -> Tuple[float, float]:
        """
        Project expected inflows and outflows over horizon.

        Args:
            current_liquidity: Current liquidity
            horizon: Projection horizon in days

        Returns:
            Tuple of (projected_inflow, projected_outflow)
        """
        # Simple projection based on mean drift
        projected_inflow = current_liquidity * self.inflow_mean * horizon
        projected_outflow = current_liquidity * self.outflow_mean * horizon

        return projected_inflow, projected_outflow
