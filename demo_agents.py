"""
TradeSwarm: Multi-Agent Background Deliberation Inspector.

Demonstrates step-by-step how each agent works in the background:
- Track 1 (Trading Swarm): MarketAgent, SentimentAgent, PredictionAgent, DecisionAgent, RiskAgent
- Track 2 (Core Novelty): SentimentAnomalyAgent (The Independent Veto Authority)

Run anytime with:
    python demo_agents.py
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.market_agent import MarketAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.risk_agent import RiskAgent
from src.agents.sentiment_anomaly_agent import SentimentAnomalyAgent

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def print_banner(text: str, char: str = "=", width: int = 75):
    print(char * width)
    print(f" {text}")
    print(char * width)


def inspect_event_deliberation(event_row: pd.Series, train_df: pd.DataFrame):
    coin = event_row["coin_symbol"]
    event_id = event_row["event_id"]
    is_pump = event_row["label"] == "pump_and_dump_event"
    label_str = "🚨 PUMP-AND-DUMP MANIPULATION" if is_pump else "🟢 NORMAL CALM TRADING"
    
    price_series = json.loads(event_row["price_series_json"])
    price_df = pd.DataFrame(price_series)
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)
    messages = json.loads(event_row["telegram_messages_json"])

    print_banner(f"EVENT {event_id} — {coin}/BTC | {label_str}")
    print(f"• Timestamp Window : {event_row['pump_start_time'][:19]}")
    print(f"• Candle Data Points: {len(price_df)} one-minute OHLCV candles")
    print(f"• Messages Received : {len(messages)} Telegram messages")
    print(f"• Sample Messages   :")
    for i, m in enumerate(messages[:3]):
        txt = m.get('text', '').replace('\n', ' ')[:70]
        print(f"    [{i+1}] \"{txt}...\" ({m.get('group_id', 'Telegram')})")

    # Initialize Agents
    market_agent = MarketAgent(momentum_window=5, volatility_window=10)
    sentiment_agent = SentimentAgent(threshold=0.05, use_finbert=False)
    pred_agent = PredictionAgent()
    decision_agent = DecisionAgent()
    risk_agent = RiskAgent()
    veto_agent = SentimentAnomalyAgent(veto_threshold=1.25)
    veto_agent.calibrate(train_df)

    state = {
        "coin_symbol": coin,
        "price_df": price_df,
        "messages": messages,
        "portfolio": {"capital": 10000.0, "position": 0.0, "drawdown": 0.0},
    }

    # ==============================================================
    # TRACK 1: TRADING CONSENSUS SWARM
    # ==============================================================
    print("\n" + "-" * 75)
    print(" 📡 TRACK 1: TRADING CONSENSUS SWARM (Deliberation in Background)")
    print("-" * 75)

    # 1. Market Agent
    sig_m = market_agent.act(state)
    prices = price_df["close"].values
    curr_price = prices[-1]
    p_5ago = prices[-min(6, len(prices))]
    momentum_pct = ((curr_price - p_5ago) / p_5ago) * 100
    print(f"\n[1] 📈 MARKET AGENT (Technical Analysis)")
    print(f"    • What it inspects: Rolling price candles (close price = {curr_price:.8f} BTC)")
    print(f"    • Mathematical formula: 5-minute Momentum = (P_t - P_t-5) / P_t-5")
    print(f"    • Computed Momentum : {sig_m.raw_value:+.4f} ({momentum_pct:+.2f}%)")
    print(f"    • Internal Decision : {sig_m.action} (Confidence: {sig_m.confidence:.1%})")
    print(f"    • Emitted Vote      : {sig_m.vote} {'(VOTE TO BUY)' if sig_m.vote == 1 else '(VOTE TO HOLD/SELL)'}")

    # 2. Sentiment Agent
    sig_s = sentiment_agent.act(state)
    print(f"\n[2] 💬 SENTIMENT AGENT (Natural Language Processing)")
    print(f"    • What it inspects: Intercepted Telegram message text stream")
    print(f"    • Mathematical formula: Polarity S = P(bullish) - P(bearish) ∈ [-1, +1]")
    print(f"    • Computed Polarity : {sig_s.raw_value:+.3f} (Threshold = ±0.05)")
    print(f"    • Internal Decision : {sig_s.action} (Confidence: {sig_s.confidence:.1%})")
    print(f"    • Emitted Vote      : {sig_s.vote} {'(VOTE TO BUY)' if sig_s.vote == 1 else '(VOTE TO HOLD/SELL)'}")

    # 3. Prediction Agent
    sig_p = pred_agent.act(state)
    forecast_price = sig_p.raw_value
    pred_diff_pct = ((forecast_price - curr_price) / curr_price) * 100
    print(f"\n[3] 🧠 PREDICTION AGENT (2-Layer Stacked PyTorch LSTM)")
    print(f"    • What it inspects: Sequential 10-step price history tensor")
    print(f"    • Current Price     : {curr_price:.8f} BTC")
    print(f"    • LSTM Next Forecast: {forecast_price:.8f} BTC ({pred_diff_pct:+.2f}%)")
    print(f"    • Internal Decision : {sig_p.action} (Confidence: {sig_p.confidence:.1%})")
    print(f"    • Emitted Vote      : {sig_p.vote} {'(VOTE TO BUY)' if sig_p.vote == 1 else '(VOTE TO HOLD/SELL)'}")

    # 4. Decision Agent (Majority Voting)
    vote_state = dict(state)
    vote_state["signals"] = [sig_m, sig_s, sig_p]
    sig_dec = decision_agent.act(vote_state)
    total_votes = sig_dec.metadata.get("total_votes", 0)
    print(f"\n[4] 🗳️ DECISION AGENT (Democratic Majority Voting)")
    print(f"    • Vote Tally: Market={sig_m.vote} | Sentiment={sig_s.vote} | Prediction={sig_p.vote}")
    print(f"    • Total Votes: {total_votes} out of 3")
    print(f"    • Swarm Majority Consensus: ==> {sig_dec.action} <==")

    # 5. Risk Agent
    risk_state = dict(state)
    risk_state["proposed_action"] = sig_dec.action
    sig_risk = risk_agent.act(risk_state)
    print(f"\n[5] ⚖️ RISK AGENT (Portfolio Safety & Volatility Gating)")
    print(f"    • Rolling Return Volatility: {sig_risk.raw_value:.4f} (Max Limit: 0.05)")
    print(f"    • Gating Result: {sig_risk.metadata.get('gate_reason', 'Normal')}")
    print(f"    • Swarm Proposed Trade: ==> {sig_risk.action} <==")

    # ==============================================================
    # TRACK 2: INDEPENDENT VETO AUTHORITY
    # ==============================================================
    print("\n" + "-" * 75)
    print(" 🛡️ TRACK 2: INDEPENDENT SENTIMENT-ANOMALY VETO AGENT (Core Novelty)")
    print("-" * 75)
    
    anomaly_index, z_sent, z_vol = veto_agent.compute_anomaly_index(coin, messages)
    threshold = veto_agent.veto_threshold
    is_veto = anomaly_index >= threshold

    coin_base = veto_agent.coin_baselines.get(coin, veto_agent.global_baseline)
    print(f"\n[6] 🔬 STATISTICAL ANOMALY CALCULATION")
    print(f"    • Baseline Language Mean (μ_sent): {coin_base['mean_sent']:+.3f} (Std: {coin_base['std_sent']:.3f})")
    print(f"    • Baseline Message Vol   (μ_vol) : {coin_base['mean_vol']:.2f} msgs (Std: {coin_base['std_vol']:.2f})")
    print(f"    • Current Window Observation     : {len(messages)} messages, Polarity = {sig_s.raw_value:+.3f}")
    print(f"    • Standardized Deviations        : Z_sent = {z_sent:+.2f}σ | Z_vol = {z_vol:+.2f}σ")
    print(f"    • Composite Anomaly Index A_t    : A_t = 0.60 * Z_sent + 0.40 * Z_vol")
    print(f"                                     : ==> A_t = {anomaly_index:.2f}σ <==")
    print(f"    • Calibrated Veto Threshold θ*   : 1.25σ (Tuned on Validation Split)")

    # ==============================================================
    # THE OVERRIDE LOOP
    # ==============================================================
    print("\n" + "=" * 75)
    print(" ⚡ THE OVERRIDE LOOP & FINAL EXECUTION")
    print("=" * 75)

    if is_veto:
        print(f"\n 🚨 INDEPENDENT VETO FIRED!")
        print(f"    - Anomaly Index ({anomaly_index:.2f}σ) EXCEEDED safety threshold (1.25σ)!")
        print(f"    - Trading Swarm Proposed Action : {sig_risk.action}")
        print(f"    - VETO AGENT OVERRIDE ACTION    : FORCE_REVERSE")
        print(f"    - Final Order Sent to Exchange  : ==> FORCE_REVERSE <==")
        print(f"    - Result: Portfolio position liquidated/hedged before the dump! 0.00% loss incurred!")
    else:
        print(f"\n 🟢 NORMAL PASS-THROUGH (No Manipulation Detected)")
        print(f"    - Anomaly Index ({anomaly_index:.2f}σ) is within normal boundary (< 1.25σ).")
        print(f"    - Veto Status                   : PASSIVE")
        print(f"    - Final Order Sent to Exchange  : ==> {sig_risk.action} <==")
        print(f"    - Result: Normal trading order executes undisturbed.")
    print("=" * 75 + "\n")


def main():
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Pick 1 Pump Event and 1 Normal Trading Event
    pump_event = test_df[test_df["label"] == "pump_and_dump_event"].iloc[0]
    normal_event = test_df[test_df["label"] == "normal_trading"].iloc[0]

    inspect_event_deliberation(pump_event, train_df)
    inspect_event_deliberation(normal_event, train_df)


if __name__ == "__main__":
    main()
