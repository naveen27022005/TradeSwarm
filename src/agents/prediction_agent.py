"""
Phase 2, Step 2b: Prediction Agent (Base paper: Singh Table II).

Wraps the LSTM price predictor to output a binary trade signal:
1 (BUY)  if predicted_price > current_price
0 (HOLD) if predicted_price <= current_price
"""
from typing import Any, Dict
import numpy as np
import pandas as pd
from src.agents.base_agent import AgentSignal, BaseAgent
from src.models.lstm_predictor import LSTMPredictor


class PredictionAgent(BaseAgent):
    """
    Prediction Agent using LSTM price forecasts to emit directional signals,
    conforming to BaseAgent.
    """

    def __init__(
        self,
        name: str = "PredictionAgent",
        predictor: LSTMPredictor = None,
        lookback: int = 10,
    ):
        super().__init__(name=name)
        self.predictor = predictor or LSTMPredictor(lookback=lookback)
        self.lookback = lookback

    def act(self, state: Dict[str, Any]) -> AgentSignal:
        """Process state price series and emit standard AgentSignal."""
        price_df = state.get("price_df")
        if price_df is None or len(price_df) < self.lookback:
            return AgentSignal(
                agent_name=self.name,
                action="HOLD",
                confidence=0.5,
                vote=0,
                raw_value=0.0,
            )

        prices = price_df["close"].values
        current_price = float(prices[-1])
        pred_price = (
            float(self.predictor.predict(prices))
            if self.predictor.is_fitted
            else float(prices[-1] * 1.001 if prices[-1] > prices[-min(3, len(prices))] else prices[-1] * 0.999)
        )

        vote = 1 if pred_price > current_price else 0
        action = "BUY" if vote == 1 else "HOLD"
        pct_diff = (pred_price - current_price) / (current_price + 1e-8)
        confidence = float(np.clip(abs(pct_diff) * 50, 0.5, 1.0))

        return AgentSignal(
            agent_name=self.name,
            action=action,
            confidence=confidence,
            vote=vote,
            raw_value=pred_price,
            metadata={"current_price": current_price, "pred_price": pred_price, "pct_diff": pct_diff},
        )

    def predict_signal(self, price_df: pd.DataFrame) -> int:
        """
        Evaluate recent price action and generate binary directional signal.

        Returns
        -------
        int: 1 if forecast > current_price else 0
        """
        if len(price_df) < self.lookback:
            return 0

        prices = price_df["close"].values
        current_price = prices[-1]

        if not self.predictor.is_fitted:
            # Fallback to recent short-term momentum if model not trained
            return 1 if prices[-1] > prices[-min(3, len(prices))] else 0

        pred_price = self.predictor.predict(prices)
        return 1 if pred_price > current_price else 0

    def batch_signals(self, price_df: pd.DataFrame) -> np.ndarray:
        """Compute signals over a sliding window across a price series."""
        prices = price_df["close"].values
        signals = np.zeros(len(prices), dtype=int)

        for i in range(self.lookback, len(prices)):
            window = prices[: i + 1]
            if self.predictor.is_fitted:
                pred = self.predictor.predict(window)
                signals[i] = 1 if pred > window[-1] else 0
            else:
                signals[i] = 1 if window[-1] > window[-3] else 0

        return signals
