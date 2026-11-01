"""
TradeSwarm System Configuration.

Holds runtime switches, agent thresholds, and simulation parameters.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TradeSwarmConfig:
    """Master configuration dataclass for TradeSwarm."""

    # Core Novelty Switch
    veto_enabled: bool = True

    # Portfolio & Simulation
    initial_capital: float = 10000.0
    transaction_fee: float = 0.001  # 0.1% Binance spot fee
    position_sizing_pct: float = 1.0  # 100% of available capital

    # Market & Risk Thresholds
    momentum_window: int = 5
    volatility_window: int = 10
    max_volatility: float = 0.05
    max_drawdown_limit: float = 0.25

    # Sentiment & Anomaly Agent
    sentiment_threshold: float = 0.05
    veto_threshold: float = 1.80  # Calibrated on validation split
    sentiment_weight: float = 0.6
    volume_weight: float = 0.4
    forced_action: str = "FORCE_REVERSE"  # Exit or reverse position

    # LSTM Predictor
    lookback_window: int = 10
    lstm_hidden_dim: int = 64
    lstm_epochs: int = 25
    lstm_lr: float = 0.001

    # Q-Learning Refinement
    q_alpha: float = 0.1
    q_gamma: float = 0.9
    q_epsilon: float = 0.1
    use_q_refinement: bool = False  # Pure majority vote vs. Q-refined

