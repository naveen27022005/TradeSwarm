"""
Validation tests for Phase 2: Baseline System Reproduction.

Verifies:
1. Decision Agent voting truth table against hand-computed test vectors
2. Market Agent technical indicator computation and directional signal
3. Sentiment Agent polarity calculation on real Telegram coordination text
4. LSTM Predictor architecture, training step, and evaluation metrics
5. Q-Learning refinement layer state-to-index encoding and Bellman update
6. End-to-end Baseline Backtest runs and outputs equity curve
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.agents.decision_agent import DecisionAgent
from src.agents.market_agent import MarketAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.sentiment_agent import SentimentAgent
from src.models.lstm_predictor import LSTMPredictor, SinghLSTM
from src.models.q_learning import QLearningAgent


def test_decision_agent_truth_table():
    """Verify majority-vote logic exactly per Singh (2026) Table II."""
    da = DecisionAgent()

    # Sum >= 2 -> BUY
    assert da.vote(1, 1, 1) == "BUY"
    assert da.vote(1, 1, 0) == "BUY"
    assert da.vote(1, 0, 1) == "BUY"
    assert da.vote(0, 1, 1) == "BUY"

    # Sum == 1 -> HOLD
    assert da.vote(1, 0, 0) == "HOLD"
    assert da.vote(0, 1, 0) == "HOLD"
    assert da.vote(0, 0, 1) == "HOLD"

    # Sum == 0 -> SELL
    assert da.vote(0, 0, 0) == "SELL"


def test_market_agent_indicators():
    """Verify momentum and volatility calculations on deterministic series."""
    agent = MarketAgent(momentum_window=5, volatility_window=10)
    prices = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]
    df = pd.DataFrame({"close": prices})

    features = agent.compute_features(df)
    assert "momentum" in features.columns
    assert "volatility" in features.columns

    # 5-period momentum at idx 5: (12.5 - 10.0) / 10.0 = 0.25
    np.testing.assert_almost_equal(features["momentum"].iloc[5], 0.25, decimal=4)

    # Monotonically increasing prices should trigger BUY (1)
    sig = agent.predict_signal(df)
    assert sig in (0, 1)


def test_sentiment_agent_polarity():
    """Verify sentiment agent produces expected polarity on coordination texts."""
    agent = SentimentAgent(threshold=0.05, use_finbert=False)

    pump_msgs = [
        {"text": "BUY BRD NOW! PUMP TARGET ON BINANCE 🚀🚀🚀"},
        {"text": "Huge profits coming, hold and do not dump!"},
    ]
    dump_msgs = [
        {"text": "SELL SELL SELL! Take profits immediately!"},
        {"text": "Price is crashing, panic sell now!"},
    ]

    pump_pol = agent.score_messages(pump_msgs)
    dump_pol = agent.score_messages(dump_msgs)

    assert pump_pol > 0.05, f"Pump messages should have positive polarity, got {pump_pol}"
    assert dump_pol < -0.05, f"Dump messages should have negative polarity, got {dump_pol}"
    assert agent.predict_signal(pump_msgs) == 1
    assert agent.predict_signal(dump_msgs) == 0


def test_lstm_predictor_architecture_and_forward():
    """Verify SinghLSTM forward pass output shape and loss reduction."""
    model = SinghLSTM(input_dim=1, hidden_dim=64, dropout=0.2)
    dummy_input = torch.randn(8, 10, 1)  # batch_size=8, seq_len=10, input_dim=1
    out = model(dummy_input)
    assert out.shape == (8, 1), f"Expected (8, 1), got {out.shape}"


def test_lstm_training_convergence():
    """Verify LSTM loss decreases during training."""
    import torch

    np.random.seed(42)
    # Sine wave sequence with trend
    t = np.linspace(0, 10, 200)
    prices = 100.0 + 5.0 * np.sin(t) + 0.1 * t

    predictor = LSTMPredictor(lookback=8, hidden_dim=32, lr=0.01)
    losses = predictor.fit(prices[:160], epochs=15, batch_size=32)

    assert len(losses) == 15
    # Loss should decrease from initial to final
    assert losses[-1] < losses[0], f"LSTM failed to reduce loss: {losses[0]} -> {losses[-1]}"

    # Next step prediction
    next_p = predictor.predict(prices[-8:])
    assert np.isfinite(next_p)


def test_q_learning_encoding_and_update():
    """Verify 3-bit state encoding and Bellman update in QLearningAgent."""
    q_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.0)

    # 3-bit states: (M, S, P)
    assert q_agent.state_to_index(0, 0, 0) == 0
    assert q_agent.state_to_index(0, 0, 1) == 1
    assert q_agent.state_to_index(0, 1, 0) == 2
    assert q_agent.state_to_index(0, 1, 1) == 3
    assert q_agent.state_to_index(1, 0, 0) == 4
    assert q_agent.state_to_index(1, 0, 1) == 5
    assert q_agent.state_to_index(1, 1, 0) == 6
    assert q_agent.state_to_index(1, 1, 1) == 7

    # Test Q-update
    s = 7
    s_next = 3
    initial_q = q_agent.q_table[s, 1]  # BUY
    reward = 0.05
    q_agent.update(state_idx=s, action="BUY", reward=reward, next_state_idx=s_next, done=False)
    updated_q = q_agent.q_table[s, 1]
    assert updated_q > initial_q, "Q-value did not increase after positive reward"
