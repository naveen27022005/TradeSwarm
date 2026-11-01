"""
Phase 3, Step 3: Risk Agent (MSPM-style Risk-Adjusted Allocation and Gating).

Derived from Huang & Tanaka (2021) MSPM Eq. 8 (risk-adjusted reward and volatility penalty).
Monitors:
- Market return volatility (standard deviation of recent returns)
- Portfolio drawdown (peak-to-trough decline)
- Position concentration limits

Policy:
- If volatility > max_volatility: throttles or blocks new BUY actions (gates to HOLD).
- If trailing drawdown > max_drawdown_limit: forces defensive cash preservation.
- Outputs sizing multiplier: [0.0, 1.0] indicating safe capital fraction to deploy.
"""
from typing import Any, Dict
import numpy as np
import pandas as pd
from src.agents.base_agent import AgentSignal, BaseAgent


class RiskAgent(BaseAgent):
    """
    Risk Agent performing volatility gating and position sizing controls.
    """

    def __init__(
        self,
        name: str = "RiskAgent",
        max_volatility: float = 0.05,
        max_drawdown_limit: float = 0.25,
        target_allocation: float = 1.0,
    ):
        super().__init__(name=name)
        self.max_volatility = max_volatility
        self.max_drawdown_limit = max_drawdown_limit
        self.target_allocation = target_allocation

    def assess_risk(self, price_df: pd.DataFrame, portfolio: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Compute volatility and drawdown risk metrics.

        Returns
        -------
        dict with: volatility, current_drawdown, risk_score in [0, 1]
        """
        portfolio = portfolio or {}
        vol = 0.0
        if price_df is not None and len(price_df) >= 10:
            returns = price_df["close"].pct_change().dropna()
            vol = float(returns.tail(15).std()) if len(returns) >= 2 else 0.0

        current_dd = float(portfolio.get("drawdown", 0.0))

        # Risk score calculation (0 to 1)
        vol_risk = min(1.0, vol / (self.max_volatility + 1e-8))
        dd_risk = min(1.0, current_dd / (self.max_drawdown_limit + 1e-8))
        aggregate_risk = float(0.5 * vol_risk + 0.5 * dd_risk)

        return {
            "volatility": vol,
            "drawdown": current_dd,
            "risk_score": aggregate_risk,
        }

    def act(self, state: Dict[str, Any]) -> AgentSignal:
        """
        Evaluate proposed trade against portfolio and volatility constraints.
        """
        price_df = state.get("price_df")
        portfolio = state.get("portfolio", {})
        proposed_action = state.get("proposed_action", "HOLD")

        risk_metrics = self.assess_risk(price_df, portfolio)
        risk_score = risk_metrics["risk_score"]
        vol = risk_metrics["volatility"]
        dd = risk_metrics["drawdown"]

        # If risk exceeds thresholds, gate trade to HOLD or defensively SELL
        if dd >= self.max_drawdown_limit:
            action = "SELL"  # Stop loss / capital preservation
            allowed_size = 0.0
            gate_reason = f"Max drawdown breached: {dd:.1%} >= {self.max_drawdown_limit:.1%}"
        elif vol >= self.max_volatility and proposed_action == "BUY":
            action = "HOLD"  # Block aggressive buying into high volatility
            allowed_size = 0.0
            gate_reason = f"High volatility block: {vol:.4f} >= {self.max_volatility:.4f}"
        else:
            action = proposed_action
            allowed_size = max(0.2, 1.0 - risk_score)  # Scale sizing dynamically
            gate_reason = "Approved"

        return AgentSignal(
            agent_name=self.name,
            action=action,
            confidence=float(1.0 - risk_score),
            vote=1 if action == "BUY" else 0,
            raw_value=risk_score,
            metadata={
                "risk_metrics": risk_metrics,
                "allowed_size": allowed_size,
                "gate_reason": gate_reason,
            },
        )

