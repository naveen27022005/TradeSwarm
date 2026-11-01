"""
Phase 2, Step 1: Market Agent (Base paper: Singh Table II).

Computes market technical indicators from price and volume:
- 5-period momentum: (Close_t - Close_{t-5}) / Close_{t-5}
- 10-period volatility: std(returns, 10 periods)
- Binary signal:
    1 (BUY)  if momentum > 0 and volatility < median_volatility
    0 (HOLD/SELL) otherwise
"""
from typing import Any, Dict
import numpy as np
import pandas as pd
from src.agents.base_agent import AgentSignal, BaseAgent


class MarketAgent(BaseAgent):
    """
    Market Agent computing momentum and volatility signals
    per Singh (2026) Table II, conforming to BaseAgent.
    """

    def __init__(
        self,
        name: str = "MarketAgent",
        momentum_window: int = 5,
        volatility_window: int = 10,
        momentum_threshold: float = 0.0,
    ):
        super().__init__(name=name)
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.momentum_threshold = momentum_threshold

    def act(self, state: Dict[str, Any]) -> AgentSignal:
        """Process state and emit standard AgentSignal."""
        price_df = state.get("price_df")
        if price_df is None or len(price_df) < max(self.momentum_window, self.volatility_window) + 1:
            return AgentSignal(
                agent_name=self.name,
                action="HOLD",
                confidence=0.5,
                vote=0,
                raw_value=0.0,
            )

        features = self.compute_features(price_df)
        latest = features.iloc[-1]
        mom = float(latest.get("momentum", 0.0))
        vol = float(latest.get("volatility", 0.0))

        vote = self.predict_signal(price_df)
        action = "BUY" if vote == 1 else "HOLD"
        confidence = float(np.clip(abs(mom) * 10, 0.5, 1.0)) if not pd.isna(mom) else 0.5

        return AgentSignal(
            agent_name=self.name,
            action=action,
            confidence=confidence,
            vote=vote,
            raw_value=mom if not pd.isna(mom) else 0.0,
            metadata={"momentum": mom, "volatility": vol},
        )

    def compute_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum, returns, and rolling volatility on a price DataFrame.

        Parameters
        ----------
        price_df : pd.DataFrame
            Must contain 'close' column.

        Returns
        -------
        pd.DataFrame with added columns: 'returns', 'momentum', 'volatility'
        """
        df = price_df.copy()
        if "close" not in df.columns:
            raise ValueError("price_df must contain 'close' column")

        # 1-period returns
        df["returns"] = df["close"].pct_change()

        # N-period momentum: (P_t - P_{t-N}) / P_{t-N}
        df["momentum"] = (
            df["close"] - df["close"].shift(self.momentum_window)
        ) / df["close"].shift(self.momentum_window)

        # N-period return volatility (std dev)
        df["volatility"] = df["returns"].rolling(window=self.volatility_window).std()

        return df

    def predict_signal(self, price_df: pd.DataFrame) -> int:
        """
        Produce a binary trade signal from the latest price action.

        Returns
        -------
        int: 1 for BUY, 0 for HOLD/SELL
        """
        if len(price_df) < max(self.momentum_window, self.volatility_window) + 1:
            return 0  # Not enough data, default to HOLD

        features = self.compute_features(price_df)
        latest = features.iloc[-1]

        mom = latest.get("momentum", 0.0)
        vol = latest.get("volatility", 0.0)

        if pd.isna(mom) or pd.isna(vol):
            return 0

        # Dynamic median volatility from available window
        recent_vols = features["volatility"].dropna()
        med_vol = recent_vols.median() if len(recent_vols) > 0 else 0.02

        # Buy condition: positive momentum with controlled volatility
        if mom > self.momentum_threshold and vol <= med_vol * 1.5:
            return 1
        return 0

    def batch_signals(self, price_df: pd.DataFrame) -> np.ndarray:
        """Compute signals for each timestep in a price series."""
        features = self.compute_features(price_df)
        signals = np.zeros(len(features), dtype=int)

        med_vol = features["volatility"].median()
        if pd.isna(med_vol):
            med_vol = 0.02

        mask = (features["momentum"] > self.momentum_threshold) & (
            features["volatility"] <= med_vol * 1.5
        )
        signals[mask.fillna(False)] = 1
        return signals


if __name__ == "__main__":
    # Self-test
    dates = pd.date_range("2021-01-01", periods=50, freq="1min")
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, 50)))
    test_df = pd.DataFrame({"timestamp": dates, "close": prices})

    agent = MarketAgent()
    sig = agent.predict_signal(test_df)
    signals = agent.batch_signals(test_df)
    print(f"MarketAgent self-test signal: {sig}")
    print(f"MarketAgent batch signals (BUY count): {np.sum(signals)}/{len(signals)}")
