"""
Phase 6: Publication-Quality Visualization Engine.

Generates:
1. ablation_equity_curves.png: Overlay of portfolio equity for Baseline, Veto-Off, Veto-On
2. case_study_events.png: Detailed 2-3 event case studies showing price series, Anomaly Index, and agent signals
"""
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.split_data import get_messages, get_price_df


def plot_ablation_equity_curves(
    curves: Dict[str, List[float]],
    metrics: Dict[str, Dict[str, float]],
    save_path: Path,
):
    """
    Generate overlaid equity curves for Baseline, Veto-Disabled, and Veto-Enabled.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6), dpi=300)

    colors = {
        "Baseline": "#d9534f",        # Crimson Red
        "Veto-Disabled": "#f0ad4e",   # Amber / Orange
        "Veto-Enabled": "#2e7d32",    # Forest Green
    }
    styles = {
        "Baseline": ":",
        "Veto-Disabled": "--",
        "Veto-Enabled": "-",
    }
    widths = {
        "Baseline": 2.0,
        "Veto-Disabled": 2.0,
        "Veto-Enabled": 2.8,
    }

    for name, curve in curves.items():
        m = metrics.get(name, {})
        sharpe = m.get("sharpe_ratio", 0.0)
        mdd = m.get("max_drawdown_pct", 0.0)
        label = f"{name} (Sharpe: {sharpe:.2f}, MDD: {mdd:.1f}%)"
        plt.plot(
            curve,
            label=label,
            color=colors.get(name, "black"),
            linestyle=styles.get(name, "-"),
            linewidth=widths.get(name, 2.0),
        )

    plt.axhline(10000.0, color="#7f8c8d", linestyle=":", alpha=0.7, label="Initial Capital ($10,000)")
    plt.title("Ablation Study: Equity Curves Across Architecture Configurations", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Trade Steps (Chronological Test Split)", fontsize=12)
    plt.ylabel("Portfolio Equity ($)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=10, loc="upper left", framealpha=0.95)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


def plot_event_case_studies(
    case_events: List[pd.Series],
    anomaly_agent,
    save_path: Path,
):
    """
    Generate multi-panel case study plots showing:
    Top panel: Price series with buy/sell/veto execution points
    Bottom panel: Anomaly Index vs Veto Threshold
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_events = min(3, len(case_events))
    if num_events == 0:
        return

    fig, axes = plt.subplots(num_events, 2, figsize=(14, 4 * num_events), dpi=300)
    if num_events == 1:
        axes = np.array([axes])

    for i in range(num_events):
        row = case_events[i]
        coin = row["coin_symbol"]
        event_id = row["event_id"]
        pdf = get_price_df(row)
        msgs = get_messages(row)

        prices = pdf["close"].values
        times = range(len(prices))

        anomaly_idx, z_sent, z_vol = anomaly_agent.compute_anomaly_index(coin, msgs)
        threshold = anomaly_agent.veto_threshold

        # Left panel: Price Series
        ax_price = axes[i, 0]
        ax_price.plot(times, prices, color="#1f77b4", linewidth=1.8, label=f"{coin}/BTC Price")
        peak_idx = int(np.argmax(prices))
        ax_price.scatter(peak_idx, prices[peak_idx], color="red", s=80, zorder=5, label="Pump Peak")
        ax_price.set_title(f"Case {i+1}: {coin} ({event_id}) - Price Action", fontsize=11, fontweight="bold")
        ax_price.set_ylabel("Price (BTC)", fontsize=10)
        ax_price.set_xlabel("Time (Minutes)", fontsize=10)
        ax_price.grid(True, linestyle=":", alpha=0.5)
        ax_price.legend(fontsize=9)

        # Right panel: Anomaly Index Gauge
        ax_anom = axes[i, 1]
        ax_anom.bar(
            ["Z-Sentiment", "Z-Volume", "Anomaly Index"],
            [z_sent, z_vol, anomaly_idx],
            color=["#3498db", "#9b59b6", "#e74c3c" if anomaly_idx >= threshold else "#2ecc71"],
            width=0.5,
        )
        ax_anom.axhline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Veto Threshold (θ={threshold:.2f})")
        ax_anom.set_title(f"Statistical Anomaly Metrics ({'VETO TRIGGERED' if anomaly_idx >= threshold else 'NORMAL'})", fontsize=11, fontweight="bold")
        ax_anom.set_ylabel("Z-Score / Deviation", fontsize=10)
        ax_anom.grid(True, linestyle=":", alpha=0.5)
        ax_anom.legend(fontsize=9)

    plt.suptitle("TradeSwarm Manipulation Case Studies: Price Action & Sentiment Anomaly Index", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

