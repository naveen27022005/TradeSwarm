"""
Phase 6: Statistical Significance Testing.

Applies non-parametric tests (Mann-Whitney U test) following MSPM methodology
to verify whether Veto-Enabled statistically outperforms Baseline and Veto-Disabled
configurations on manipulation event returns and drawdowns.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats


def run_mann_whitney_u_test(
    sample_a: List[float],
    sample_b: List[float],
    alternative: str = "greater",
) -> Dict[str, float]:
    """
    Execute Mann-Whitney U test comparing distribution A (Veto-Enabled)
    vs. distribution B (Baseline or Veto-Disabled).

    Null hypothesis H0: Distribution A is stochastically equal to B.
    Alternative H1: Distribution A is stochastically greater than B.

    Returns
    -------
    dict with u_statistic, p_value, is_significant (p < 0.05)
    """
    a = np.array(sample_a, dtype=float)
    b = np.array(sample_b, dtype=float)

    # Filter out NaNs/Infs
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 3 or len(b) < 3:
        return {"u_stat": 0.0, "p_value": 1.0, "is_significant": False}

    u_stat, p_val = stats.mannwhitneyu(a, b, alternative=alternative)

    return {
        "u_stat": float(u_stat),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
    }


def compare_configurations_statistically(
    returns_baseline: List[float],
    returns_veto_off: List[float],
    returns_veto_on: List[float],
) -> Dict[str, Any]:
    """
    Run paired statistical comparisons across all 3 configurations.
    """
    # 1. Veto-On vs. Baseline
    test_vs_baseline = run_mann_whitney_u_test(returns_veto_on, returns_baseline)

    # 2. Veto-On vs. Veto-Off
    test_vs_veto_off = run_mann_whitney_u_test(returns_veto_on, returns_veto_off)

    return {
        "veto_on_vs_baseline": test_vs_baseline,
        "veto_on_vs_veto_off": test_vs_veto_off,
    }
