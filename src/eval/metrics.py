"""
Phase 6: Evaluation Metrics Engine.

Computes:
1. Standard financial performance metrics:
   - Total Return (%)
   - Annualized Return (CAGR)
   - Annualized Sharpe Ratio
   - Maximum Drawdown (MDD %)
   - Win Rate (%)
   - Profit Factor

2. Novel Manipulation-Specific Metrics (Core Contribution):
   - Manipulation Episode Survival Rate (%)
   - Manipulation-Window Drawdown (%)
   - Normal-Period False Positive Veto Rate (%)
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd


def compute_standard_metrics(
    equity_curve: List[float],
    initial_capital: float = 10000.0,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """Compute standard quantitative financial metrics from equity series."""
    if len(equity_curve) < 2:
        return {
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
        }

    eq = np.array(equity_curve)
    final_eq = float(eq[-1])
    total_ret = ((final_eq - initial_capital) / initial_capital) * 100

    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]

    mean_r = float(np.mean(returns)) if len(returns) > 0 else 0.0
    std_r = float(np.std(returns)) if len(returns) > 0 else 0.0

    # Sharpe ratio
    sharpe = (mean_r / (std_r + 1e-8)) * np.sqrt(periods_per_year) if std_r > 0 else 0.0

    # Max Drawdown
    peaks = np.maximum.accumulate(eq)
    drawdowns = (peaks - eq) / peaks
    mdd = float(np.max(drawdowns)) * 100

    # Approximate CAGR based on number of events/periods
    years = max(1.0, len(eq) / periods_per_year)
    cagr = float(((final_eq / initial_capital) ** (1.0 / years) - 1.0) * 100) if final_eq > 0 else -100.0

    return {
        "final_equity": final_eq,
        "total_return_pct": total_ret,
        "cagr_pct": cagr,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": mdd,
    }


def compute_manipulation_metrics(
    events_df: pd.DataFrame,
    override_log: List[Any],
    trade_log: List[Dict[str, Any]],
    equity_curve: List[float],
) -> Dict[str, float]:
    """
    Compute novel manipulation resistance metrics.
    """
    pump_events = events_df[events_df["label"] == "pump_and_dump_event"]
    normal_events = events_df[events_df["label"] == "normal_trading"]

    total_pumps = len(pump_events)
    total_normals = len(normal_events)

    # Overridden pump events
    overridden_events = set()
    for o in override_log:
        if hasattr(o, "event_id"):
            overridden_events.add(o.event_id)
        elif isinstance(o, dict):
            overridden_events.add(o.get("event_id"))

    pump_ids = set(pump_events["event_id"].unique())
    normal_ids = set(normal_events["event_id"].unique())

    pumps_intercepted = len(overridden_events & pump_ids)
    normals_intercepted = len(overridden_events & normal_ids)

    # Survival Rate: % of pump events where system successfully vetoed or avoided buying
    survival_rate = (pumps_intercepted / total_pumps * 100) if total_pumps > 0 else 100.0

    # False Positive Veto Rate: % of normal windows incorrectly vetoed
    fpr_veto = (normals_intercepted / total_normals * 100) if total_normals > 0 else 0.0

    # Manipulation window drawdown
    # Isolate equity steps corresponding to pump events
    pump_equity_drawdowns = []
    eq_arr = np.array(equity_curve)
    for idx, row in events_df.iterrows():
        if row["label"] == "pump_and_dump_event" and idx + 1 < len(eq_arr):
            step_return = (eq_arr[idx + 1] - eq_arr[idx]) / eq_arr[idx]
            if step_return < 0:
                pump_equity_drawdowns.append(abs(step_return) * 100)

    manipulation_mdd = float(max(pump_equity_drawdowns)) if pump_equity_drawdowns else 0.0

    return {
        "manipulation_survival_rate_pct": survival_rate,
        "pumps_intercepted": pumps_intercepted,
        "total_pump_episodes": total_pumps,
        "false_positive_veto_rate_pct": fpr_veto,
        "normals_intercepted": normals_intercepted,
        "total_normal_episodes": total_normals,
        "manipulation_window_drawdown_pct": manipulation_mdd,
    }

