"""
Phase 5: Multi-Agent Orchestrator and Independent Override Loop.

Executes the two-track decision pipeline:
Track 1 (Trading Swarm):
    State -> [Market, Sentiment, Prediction] -> Majority Decision -> Risk Gate -> Proposed Trade

Track 2 (Independent Veto):
    State -> SentimentAnomalyAgent -> Anomaly Index -> is_veto

Override Loop:
    If VETO_ENABLED and is_veto:
        Final Action = Forced Reversal / Hedge
        Log Override Event
    Else:
        Final Action = Proposed Trade
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.agents.decision_agent import DecisionAgent
from src.agents.market_agent import MarketAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.risk_agent import RiskAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.sentiment_anomaly_agent import SentimentAnomalyAgent
from src.config import TradeSwarmConfig
from src.data.split_data import get_messages, get_price_df
from src.models.lstm_predictor import LSTMPredictor
from src.models.q_learning import QLearningAgent

logger = logging.getLogger(__name__)


@dataclass
class OverrideEvent:
    """Record of an independent sentiment-anomaly veto override."""

    event_id: str
    coin_symbol: str
    timestamp: str
    proposed_action: str
    forced_action: str
    anomaly_index: float
    threshold: float
    reason: str


class TradeSwarmOrchestrator:
    """
    Central Multi-Agent Orchestration Engine with Independent Veto Authority.
    """

    def __init__(self, config: Optional[TradeSwarmConfig] = None):
        self.config = config or TradeSwarmConfig()

        # Track 1: Trading Agents
        self.market_agent = MarketAgent(
            momentum_window=self.config.momentum_window,
            volatility_window=self.config.volatility_window,
        )
        self.sentiment_agent = SentimentAgent(
            threshold=self.config.sentiment_threshold, use_finbert=False
        )
        self.lstm_predictor = LSTMPredictor(
            lookback=self.config.lookback_window,
            hidden_dim=self.config.lstm_hidden_dim,
            lr=self.config.lstm_lr,
        )
        self.pred_agent = PredictionAgent(
            predictor=self.lstm_predictor, lookback=self.config.lookback_window
        )
        self.decision_agent = DecisionAgent()
        self.risk_agent = RiskAgent(
            max_volatility=self.config.max_volatility,
            max_drawdown_limit=self.config.max_drawdown_limit,
        )
        self.q_agent = QLearningAgent(
            alpha=self.config.q_alpha,
            gamma=self.config.q_gamma,
            epsilon=self.config.q_epsilon,
        )

        # Track 2: Core Novelty — Independent Sentiment-Anomaly Veto Agent
        self.anomaly_agent = SentimentAnomalyAgent(
            veto_threshold=self.config.veto_threshold,
            sentiment_weight=self.config.sentiment_weight,
            volume_weight=self.config.volume_weight,
            forced_action=self.config.forced_action,
        )

        # Audit logs
        self.override_history: List[OverrideEvent] = []

    def calibrate_and_train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """
        Train predictive agents and calibrate anomaly agent baselines.
        """
        logger.info("Calibrating Sentiment Anomaly Agent on normal trading windows...")
        self.anomaly_agent.calibrate(train_df)

        if val_df is not None:
            logger.info("Tuning veto threshold on validation set (Constraint #2)...")
            tuning = self.anomaly_agent.tune_threshold_on_validation(val_df)
            self.config.veto_threshold = tuning["optimal_threshold"]
            logger.info(f"Optimal Veto Threshold set to: {self.config.veto_threshold:.2f}")

        logger.info("Training LSTM Predictor...")
        all_prices = []
        for _, row in train_df.iterrows():
            pdf = get_price_df(row)
            if len(pdf) >= 30:
                all_prices.extend(pdf["close"].values[:60])
        price_arr = np.array(all_prices, dtype=np.float32)
        if len(price_arr) > 100:
            self.lstm_predictor.fit(
                price_arr[:2000], epochs=self.config.lstm_epochs, batch_size=64
            )

    def process_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one decision timestep.

        Parameters
        ----------
        state : dict containing:
            'coin_symbol': str
            'price_df': pd.DataFrame
            'messages': list of message dicts
            'portfolio': dict with capital, position, drawdown
            'event_id': str
            'timestamp': str

        Returns
        -------
        dict with intermediate agent signals, final action, and override status.
        """
        # --- Track 1: Trading Agents Consensus ---
        m_sig = self.market_agent.act(state)
        s_sig = self.sentiment_agent.act(state)
        p_sig = self.pred_agent.act(state)

        # Majority Vote
        trading_signals = [m_sig, s_sig, p_sig]
        vote_state = dict(state)
        vote_state["signals"] = trading_signals
        majority_sig = self.decision_agent.act(vote_state)
        proposed_action = majority_sig.action

        # Optional Q-refinement
        if self.config.use_q_refinement:
            state_idx = self.q_agent.state_to_index(m_sig.vote, s_sig.vote, p_sig.vote)
            proposed_action = self.q_agent.choose_action(state_idx, explore=False)

        # Risk Gating
        risk_state = dict(state)
        risk_state["proposed_action"] = proposed_action
        risk_sig = self.risk_agent.act(risk_state)
        gated_action = risk_sig.action

        # --- Track 2: Independent Sentiment-Anomaly Veto ---
        anomaly_sig = self.anomaly_agent.act(state)
        is_veto = anomaly_sig.metadata.get("is_veto", False)
        anomaly_index = anomaly_sig.raw_value

        # --- The Override Loop ---
        override_triggered = False
        override_record = None

        if self.config.veto_enabled and is_veto:
            override_triggered = True
            final_action = anomaly_sig.action  # FORCE_REVERSE or FORCE_HEDGE
            reason = (
                f"Sentiment Anomaly ({anomaly_index:.2f}) breached threshold "
                f"({self.config.veto_threshold:.2f}). Overrode {gated_action}."
            )
            override_record = OverrideEvent(
                event_id=state.get("event_id", "N/A"),
                coin_symbol=state.get("coin_symbol", "UNKNOWN"),
                timestamp=str(state.get("timestamp", "")),
                proposed_action=gated_action,
                forced_action=final_action,
                anomaly_index=float(anomaly_index),
                threshold=float(self.config.veto_threshold),
                reason=reason,
            )
            self.override_history.append(override_record)
        else:
            final_action = gated_action

        return {
            "final_action": final_action,
            "proposed_action": gated_action,
            "override_triggered": override_triggered,
            "override_record": override_record,
            "anomaly_index": anomaly_index,
            "signals": {
                "market": m_sig.to_dict(),
                "sentiment": s_sig.to_dict(),
                "prediction": p_sig.to_dict(),
                "decision": majority_sig.to_dict(),
                "risk": risk_sig.to_dict(),
                "anomaly": anomaly_sig.to_dict(),
            },
        }

    def run_simulation(self, dataset_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run end-to-end backtest simulation over an event dataset.
        """
        capital = self.config.initial_capital
        position = 0.0  # units of coin
        equity_curve = [capital]
        trade_log = []
        override_count = 0
        manipulation_avoided_count = 0

        for _, row in dataset_df.iterrows():
            pdf = get_price_df(row)
            if len(pdf) < 15:
                continue

            msgs = get_messages(row)
            prices = pdf["close"].values
            current_price = float(prices[-1])

            # Current drawdown relative to historical peak
            curr_peak = max(equity_curve)
            curr_dd = (curr_peak - equity_curve[-1]) / curr_peak if curr_peak > 0 else 0.0

            state = {
                "coin_symbol": row["coin_symbol"],
                "price_df": pdf,
                "messages": msgs,
                "portfolio": {
                    "capital": capital,
                    "position": position,
                    "drawdown": curr_dd,
                },
                "event_id": row["event_id"],
                "timestamp": row["pump_start_time"],
                "label": row["label"],
            }

            step_res = self.process_step(state)
            action = step_res["final_action"]

            if step_res["override_triggered"]:
                override_count += 1
                if row["label"] == "pump_and_dump_event":
                    manipulation_avoided_count += 1

            # Order Execution
            # BUY order: enter full or partial position
            if action == "BUY" and position == 0.0:
                cost = capital * (1.0 - self.config.transaction_fee)
                position = cost / current_price
                capital = 0.0
                trade_log.append({
                    "event_id": row["event_id"],
                    "action": "BUY",
                    "price": current_price,
                    "label": row["label"],
                })

            # Forced Reversal or regular SELL: close position immediately
            elif action in ("SELL", "FORCE_REVERSE", "FORCE_HEDGE") and position > 0.0:
                revenue = position * current_price * (1.0 - self.config.transaction_fee)
                capital = revenue
                position = 0.0
                trade_log.append({
                    "event_id": row["event_id"],
                    "action": action,
                    "price": current_price,
                    "label": row["label"],
                })

            current_equity = capital + (position * current_price)
            equity_curve.append(current_equity)

        # Performance summary
        final_equity = equity_curve[-1]
        ret_series = pd.Series(equity_curve).pct_change().dropna()
        total_ret_pct = ((final_equity - self.config.initial_capital) / self.config.initial_capital) * 100
        mean_r = ret_series.mean()
        std_r = ret_series.std()
        sharpe = (mean_r / (std_r + 1e-8)) * np.sqrt(252) if len(ret_series) > 1 else 0.0

        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdowns = (peak - equity_arr) / peak
        max_dd_pct = float(np.max(drawdowns)) * 100

        total_pump_events = len(dataset_df[dataset_df["label"] == "pump_and_dump_event"])
        survival_rate = (
            (manipulation_avoided_count / total_pump_events) * 100
            if total_pump_events > 0
            else 0.0
        )

        return {
            "veto_enabled": self.config.veto_enabled,
            "final_equity": float(final_equity),
            "total_return_pct": float(total_ret_pct),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_dd_pct),
            "override_count": override_count,
            "manipulation_avoided_count": manipulation_avoided_count,
            "manipulation_survival_rate_pct": float(survival_rate),
            "total_trades": len(trade_log),
            "equity_curve": equity_curve,
        }

