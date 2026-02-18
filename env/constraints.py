"""
Regulatory constraints for CBDC liquidity environment.
"""

import numpy as np
from typing import Dict, Tuple


class LiquidityConstraints:
    """Regulatory constraints including LCR and capital adequacy."""

    def __init__(
        self,
        lcr_threshold: float = 1.0,
        capital_adequacy_ratio: float = 0.08,
        max_borrow_fraction: float = 0.5,
        lcr_penalty_weight: float = 100.0,
        shortfall_penalty_weight: float = 500.0,
    ):
        """
        Initialize constraints.

        Args:
            lcr_threshold: Minimum LCR (1.0 = 100%)
            capital_adequacy_ratio: Minimum capital ratio
            max_borrow_fraction: Max borrowing as fraction of liabilities
            lcr_penalty_weight: Penalty weight for LCR violations
            shortfall_penalty_weight: Penalty weight for liquidity shortfalls
        """
        self.lcr_threshold = lcr_threshold
        self.capital_adequacy_ratio = capital_adequacy_ratio
        self.max_borrow_fraction = max_borrow_fraction
        self.lcr_penalty_weight = lcr_penalty_weight
        self.shortfall_penalty_weight = shortfall_penalty_weight

    def compute_lcr(
        self,
        high_quality_liquid_assets: float,
        net_cash_outflows: float,
    ) -> float:
        """
        Compute Liquidity Coverage Ratio.

        LCR = HQLA / Net Cash Outflows (30-day horizon)

        Args:
            high_quality_liquid_assets: HQLA amount
            net_cash_outflows: Expected net outflows over 30 days

        Returns:
            LCR value
        """
        if net_cash_outflows <= 0:
            return np.inf
        return high_quality_liquid_assets / net_cash_outflows

    def compute_capital_ratio(
        self,
        capital: float,
        risk_weighted_assets: float,
    ) -> float:
        """
        Compute capital adequacy ratio.

        Args:
            capital: Bank capital
            risk_weighted_assets: Risk-weighted assets

        Returns:
            Capital ratio
        """
        if risk_weighted_assets <= 0:
            return np.inf
        return capital / risk_weighted_assets

    def check_lcr_compliance(
        self,
        liquidity: float,
        projected_outflows: float,
    ) -> Tuple[bool, float]:
        """
        Check LCR compliance.

        Args:
            liquidity: Current liquidity buffer (HQLA)
            projected_outflows: Projected net cash outflows

        Returns:
            Tuple of (is_compliant, lcr_value)
        """
        lcr = self.compute_lcr(liquidity, projected_outflows)
        is_compliant = lcr >= self.lcr_threshold
        return is_compliant, lcr

    def compute_lcr_penalty(
        self,
        liquidity: float,
        projected_outflows: float,
    ) -> float:
        """
        Compute penalty for LCR violation.

        Args:
            liquidity: Current liquidity
            projected_outflows: Projected outflows

        Returns:
            Penalty value
        """
        is_compliant, lcr = self.check_lcr_compliance(liquidity, projected_outflows)

        if is_compliant:
            return 0.0

        # Penalty proportional to violation severity
        violation_amount = (self.lcr_threshold - lcr) * projected_outflows
        penalty = self.lcr_penalty_weight * max(0, violation_amount)

        return penalty

    def compute_shortfall_penalty(self, liquidity: float) -> float:
        """
        Compute penalty for liquidity shortfall.

        Args:
            liquidity: Current liquidity buffer

        Returns:
            Penalty value
        """
        if liquidity >= 0:
            return 0.0

        # Severe penalty for negative liquidity
        shortfall = abs(liquidity)
        penalty = self.shortfall_penalty_weight * shortfall

        return penalty

    def check_borrow_limit(
        self,
        borrow_amount: float,
        liabilities: float,
    ) -> Tuple[bool, float]:
        """
        Check if borrowing is within limits.

        Args:
            borrow_amount: Requested borrow amount
            liabilities: Current liabilities

        Returns:
            Tuple of (is_valid, capped_amount)
        """
        max_borrow = self.max_borrow_fraction * liabilities

        if borrow_amount <= max_borrow:
            return True, borrow_amount

        return False, max_borrow

    def check_capital_adequacy(
        self,
        capital: float,
        risk_weighted_assets: float,
    ) -> Tuple[bool, float]:
        """
        Check capital adequacy.

        Args:
            capital: Current capital
            risk_weighted_assets: RWA

        Returns:
            Tuple of (is_adequate, capital_ratio)
        """
        ratio = self.compute_capital_ratio(capital, risk_weighted_assets)
        is_adequate = ratio >= self.capital_adequacy_ratio
        return is_adequate, ratio

    def get_constraint_info(
        self,
        liquidity: float,
        liabilities: float,
        projected_outflows: float,
        capital: float,
        risk_weighted_assets: float,
    ) -> Dict[str, float]:
        """
        Get comprehensive constraint information.

        Args:
            liquidity: Current liquidity
            liabilities: Current liabilities
            projected_outflows: Projected outflows
            capital: Current capital
            risk_weighted_assets: RWA

        Returns:
            Dictionary with constraint metrics
        """
        lcr_compliant, lcr_value = self.check_lcr_compliance(
            liquidity, projected_outflows
        )
        capital_adequate, capital_ratio = self.check_capital_adequacy(
            capital, risk_weighted_assets
        )
        lcr_penalty = self.compute_lcr_penalty(liquidity, projected_outflows)
        shortfall_penalty = self.compute_shortfall_penalty(liquidity)

        return {
            "lcr": lcr_value,
            "lcr_compliant": float(lcr_compliant),
            "lcr_violation": max(0, self.lcr_threshold - lcr_value),
            "lcr_penalty": lcr_penalty,
            "capital_ratio": capital_ratio,
            "capital_adequate": float(capital_adequate),
            "liquidity_shortfall": max(0, -liquidity),
            "shortfall_penalty": shortfall_penalty,
            "total_penalty": lcr_penalty + shortfall_penalty,
        }
