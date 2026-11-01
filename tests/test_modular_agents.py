"""
Validation tests for Phase 3: Modular Agent Refactor (MSPM-style).

Verifies:
1. BaseAgent interface polymorphism: all agents inherit from BaseAgent
2. Standardized AgentSignal schema output across all agents
3. Swappability: agents can be substituted seamlessly in state loops
4. Risk Agent gating: blocks BUY on high volatility and forces SELL on max drawdown
5. Numerical equivalence with Phase 2 baseline outputs
"""
import numpy as np
import pandas as pd
import pytest

from src.agents.base_agent import AgentSignal, BaseAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.market_agent import MarketAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.risk_agent import RiskAgent
from src.agents.sentiment_agent import SentimentAgent


@pytest.fixture
def sample_state():
    """Create sample state dictionary for agent evaluation."""
    dates = pd.date_range("2021-01-01", periods=30, freq="1min")
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.005, 30)))
    price_df = pd.DataFrame({"timestamp": dates, "close": prices})
    messages = [
        {"text": "BUY BRD NOW! Massive target on Binance! 🚀"},
        {"text": "Hold and do not dump, next leg up soon!"},
    ]
    portfolio = {"capital": 10000.0, "position": 0.0, "drawdown": 0.05}
    return {
        "price_df": price_df,
        "messages": messages,
        "portfolio": portfolio,
        "proposed_action": "BUY",
    }


def test_base_agent_inheritance():
    """Verify all agents inherit from BaseAgent."""
    agents = [
        MarketAgent(),
        SentimentAgent(),
        PredictionAgent(),
        DecisionAgent(),
        RiskAgent(),
    ]
    for agent in agents:
        assert isinstance(agent, BaseAgent), f"{agent.__class__.__name__} is not a BaseAgent"


def test_standardized_agent_signal_output(sample_state):
    """Verify each agent emits a valid AgentSignal."""
    market = MarketAgent()
    sentiment = SentimentAgent()
    prediction = PredictionAgent()

    sig_m = market.act(sample_state)
    sig_s = sentiment.act(sample_state)
    sig_p = prediction.act(sample_state)

    for sig in [sig_m, sig_s, sig_p]:
        assert isinstance(sig, AgentSignal)
        assert sig.action in ("BUY", "HOLD", "SELL")
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.vote in (0, 1)


def test_decision_agent_signal_aggregation(sample_state):
    """Verify DecisionAgent aggregates signals from modular upstream agents."""
    market = MarketAgent()
    sentiment = SentimentAgent()
    prediction = PredictionAgent()
    decision = DecisionAgent()

    signals = [
        market.act(sample_state),
        sentiment.act(sample_state),
        prediction.act(sample_state),
    ]

    state_with_signals = dict(sample_state)
    state_with_signals["signals"] = signals

    decision_signal = decision.act(state_with_signals)
    assert isinstance(decision_signal, AgentSignal)
    assert decision_signal.action in ("BUY", "HOLD", "SELL")
    assert "total_votes" in decision_signal.metadata


def test_risk_agent_volatility_gating():
    """Verify RiskAgent blocks BUY when volatility exceeds threshold."""
    risk_agent = RiskAgent(max_volatility=0.01)

    # High volatility price series
    t = np.linspace(0, 10, 30)
    prices = 100.0 + 20.0 * np.sin(t * 3.0)  # Violent swings
    price_df = pd.DataFrame({"close": prices})

    state = {
        "price_df": price_df,
        "proposed_action": "BUY",
        "portfolio": {"drawdown": 0.02},
    }

    signal = risk_agent.act(state)
    assert signal.action == "HOLD", "RiskAgent failed to throttle BUY during extreme volatility"
    assert signal.metadata["allowed_size"] == 0.0


def test_risk_agent_drawdown_stop_loss():
    """Verify RiskAgent forces SELL when portfolio drawdown breaches limit."""
    risk_agent = RiskAgent(max_drawdown_limit=0.20)

    # Moderate price series but 25% portfolio drawdown
    price_df = pd.DataFrame({"close": [100.0] * 30})
    state = {
        "price_df": price_df,
        "proposed_action": "BUY",
        "portfolio": {"drawdown": 0.25},
    }

    signal = risk_agent.act(state)
    assert signal.action == "SELL", "RiskAgent failed to trigger capital preservation on max drawdown"


def test_numerical_regression_with_phase2(sample_state):
    """
    Verify refactored DecisionAgent output is numerically identical
    to Phase 2 majority vote logic for all 8 combinatorial states.
    """
    decision = DecisionAgent()

    for m in (0, 1):
        for s in (0, 1):
            for p in (0, 1):
                # Phase 2 direct vote
                expected_action = DecisionAgent.vote(m, s, p)

                # Phase 3 modular act(state)
                signals = [
                    AgentSignal(agent_name="M", action="BUY" if m else "HOLD", vote=m),
                    AgentSignal(agent_name="S", action="BUY" if s else "HOLD", vote=s),
                    AgentSignal(agent_name="P", action="BUY" if p else "HOLD", vote=p),
                ]
                result_signal = decision.act({"signals": signals})

                assert result_signal.action == expected_action, (
                    f"Regression failure for inputs ({m}, {s}, {p}): "
                    f"expected {expected_action}, got {result_signal.action}"
                )

