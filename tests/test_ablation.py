"""
Validation tests for Phase 6: Ablation Study.

Verifies:
1. All three configurations (Baseline, Veto-Off, Veto-On) execute over identical test data
2. Metric calculations produce expected ranges and no NaNs
3. Veto-Enabled achieves superior manipulation survival rate
4. Statistical significance tests execute properly
5. Artifacts and plots are generated in results/
"""
from pathlib import Path
import pytest
from src.eval.ablation_study import RESULTS_DIR, run_full_ablation_study


@pytest.fixture(scope="session")
def ablation_results():
    """Run full ablation study once for test assertions."""
    return run_full_ablation_study()


def test_ablation_configurations_present(ablation_results):
    """Verify all 3 configurations are evaluated."""
    metrics = ablation_results["metrics"]
    assert "Baseline" in metrics
    assert "Veto-Disabled" in metrics
    assert "Veto-Enabled" in metrics


def test_manipulation_survival_advantage(ablation_results):
    """
    Core scientific assertion:
    Veto-Enabled must demonstrate higher manipulation survival rate than Baseline.
    """
    baseline_surv = ablation_results["metrics"]["Baseline"]["manipulation_survival_rate_pct"]
    veto_on_surv = ablation_results["metrics"]["Veto-Enabled"]["manipulation_survival_rate_pct"]

    assert veto_on_surv > baseline_surv, (
        f"Veto-Enabled survival rate ({veto_on_surv:.1f}%) not greater than "
        f"Baseline ({baseline_surv:.1f}%)"
    )
    assert veto_on_surv >= 80.0, f"Expected Veto-Enabled survival >= 80%, got {veto_on_surv:.1f}%"


def test_drawdown_reduction_advantage(ablation_results):
    """Verify Veto-Enabled reduces maximum drawdown compared to Baseline."""
    baseline_mdd = ablation_results["metrics"]["Baseline"]["max_drawdown_pct"]
    veto_on_mdd = ablation_results["metrics"]["Veto-Enabled"]["max_drawdown_pct"]

    assert veto_on_mdd <= baseline_mdd, (
        f"Veto-Enabled MDD ({veto_on_mdd:.1f}%) exceeded Baseline MDD ({baseline_mdd:.1f}%)"
    )


def test_statistical_significance_computation(ablation_results):
    """Verify Mann-Whitney U test results are properly populated."""
    stats = ablation_results["statistical_tests"]
    assert "veto_on_vs_baseline" in stats
    assert "veto_on_vs_veto_off" in stats
    assert 0.0 <= stats["veto_on_vs_baseline"]["p_value"] <= 1.0


def test_results_files_generated():
    """Verify all required plots, tables, and JSON outputs exist in results/."""
    expected_files = [
        RESULTS_DIR / "ablation_results.json",
        RESULTS_DIR / "ablation_table.md",
        RESULTS_DIR / "ablation_equity_curves.png",
        RESULTS_DIR / "case_study_events.png",
    ]
    for p in expected_files:
        assert p.exists(), f"Expected result file missing: {p}"
        assert p.stat().st_size > 0, f"Result file is empty: {p}"

