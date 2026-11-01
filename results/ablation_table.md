# Empirical Ablation Study Results

## Performance Comparison on Held-Out Test Split (Chronological)

| Metric | Baseline (Singh Control) | Veto-Disabled (MSPM Gating) | Veto-Enabled (Novel Architecture) | Delta (Veto-On vs. Baseline) |
|---|---|---|---|---|
| **Total Return (%)** | -62.28% | 1999.24% | **65.82%** | **+128.09%** |
| **Annualized Sharpe Ratio** | 4.82 | 3.45 | **2.21** | **-2.61** |
| **Maximum Drawdown (MDD)** | 99.77% | 80.10% | **78.35%** | **-21.42% (Risk Cut)** |
| **Manipulation Survival Rate** | 0.0% | 0.0% | **95.7%** | **+95.7%** |
| **Drawdown During Pumps Only** | 86.44% | 80.10% | **2.00%** | **-84.44%** |
| **False-Positive Veto Rate** | 0.0% | 0.0% | **4.55%** | Controlled (<10%) |

## Statistical Significance (Mann-Whitney U Test)
- **Veto-Enabled vs. Baseline**: $p = 0.2114$ (p >= 0.05)
- **Veto-Enabled vs. Veto-Disabled**: $p = 0.5629$
