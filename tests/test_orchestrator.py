"""
Validation tests for Phase 5: Multi-Agent Orchestrator and Independent Override Loop.

Verifies:
1. Orchestrator executes two-track decision pipeline (Trading Swarm vs Veto Agent)
2. VETO_ENABLED=False passes proposed trades through unchanged without overrides
3. VETO_ENABLED=True intercepts pump hype and triggers FORCE_REVERSE
4. Override audit log records complete event metadata
5. Non-vetoed timesteps produce identical decisions across both modes
"""
import numpy as np
import pandas as pd
import pytest

from src.config import TradeSwarmConfig
from src.orchestrator import TradeSwarmOrchestrator


@pytest.fixture
def pump_state():
    """State representing a coordinated pump event with hype text."""
    dates = pd.date_range("2021-01-01", periods=30, freq="1min")
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.005, 0.01, 30)))
    price_df = pd.DataFrame({"timestamp": dates, "close": prices})
    messages = [
        {"text": "BUY BRD NOW! PUMP STARTING ON BINANCE! 🚀🚀🚀"} for _ in range(25)
    ]
    return {
        "coin_symbol": "BRD",
        "price_df": price_df,
        "messages": messages,
        "portfolio": {"capital": 10000.0, "position": 100.0, "drawdown": 0.0},
        "event_id": "E_PUMP_01",
        "timestamp": "2021-01-01T12:00:00Z",
    }


@pytest.fixture
def normal_state():
    """State representing calm normal trading."""
    dates = pd.date_range("2021-01-01", periods=30, freq="1min")
    prices = [100.0 + i * 0.01 for i in range(30)]
    price_df = pd.DataFrame({"timestamp": dates, "close": prices})
    messages = [
        {"text": "Market is quiet today. Waiting for BTC breakout."},
        {"text": "Just DCA-ing as usual."},
    ]
    return {
        "coin_symbol": "BRD",
        "price_df": price_df,
        "messages": messages,
        "portfolio": {"capital": 10000.0, "position": 0.0, "drawdown": 0.0},
        "event_id": "E_NORM_01",
        "timestamp": "2021-01-01T12:00:00Z",
    }


def test_veto_disabled_pass_through(pump_state):
    """When VETO_ENABLED=False, veto should never fire even during intense hype."""
    cfg = TradeSwarmConfig(veto_enabled=False, veto_threshold=1.5)
    orchestrator = TradeSwarmOrchestrator(cfg)

    res = orchestrator.process_step(pump_state)
    assert res["override_triggered"] is False
    assert res["final_action"] == res["proposed_action"]
    assert len(orchestrator.override_history) == 0


def test_veto_enabled_override(pump_state):
    """When VETO_ENABLED=True, anomaly should override proposed BUY with FORCE_REVERSE."""
    cfg = TradeSwarmConfig(veto_enabled=True, veto_threshold=1.5, forced_action="FORCE_REVERSE")
    orchestrator = TradeSwarmOrchestrator(cfg)

    res = orchestrator.process_step(pump_state)
    assert res["override_triggered"] is True
    assert res["final_action"] == "FORCE_REVERSE"
    assert len(orchestrator.override_history) == 1

    entry = orchestrator.override_history[0]
    assert entry.event_id == "E_PUMP_01"
    assert entry.coin_symbol == "BRD"
    assert entry.forced_action == "FORCE_REVERSE"


def test_identical_decisions_on_normal_state(normal_state):
    """When no anomaly exists, veto-enabled and veto-disabled must produce identical outputs."""
    cfg_on = TradeSwarmConfig(veto_enabled=True, veto_threshold=2.0)
    cfg_off = TradeSwarmConfig(veto_enabled=False, veto_threshold=2.0)

    orch_on = TradeSwarmOrchestrator(cfg_on)
    orch_off = TradeSwarmOrchestrator(cfg_off)

    res_on = orch_on.process_step(normal_state)
    res_off = orch_off.process_step(normal_state)

    assert res_on["override_triggered"] is False
    assert res_off["override_triggered"] is False
    assert res_on["final_action"] == res_off["final_action"]

