# TradeSwarm: Review 2 & Final Viva Demo Script

> **Project Title:** A Multi-Agent Reinforcement Learning Architecture with Independent Sentiment-Anomaly Veto for Pump-and-Dump-Resistant Trading  
> **Student:** Final Year B.Tech / B.E. Computer Science  
> **Guide / Examiners:** Project Review Committee  

---

## 🎯 1. Review Presentation Roadmap (5-Minute Pitch)

### Slide 1: Title & The Core Problem
* **Say:** *"Good morning respected committee members. Our project addresses a critical security vulnerability in automated cryptocurrency trading: organized pump-and-dump manipulation orchestrated through Telegram channels."*
* **The Problem:** Existing multi-agent trading models (like our primary base paper, Singh 2026) fuse sentiment into a democratic majority vote or a shared RL state.
* **The Flaw:** When an organized manipulation occurs, thousands of Telegram hype messages flood in simultaneously with price spikes. The Market Agent sees momentum, the Sentiment Agent sees bullish text, and the Prediction Agent sees rising prices. The system unanimously votes **BUY**—right at the peak—and gets completely wiped out when the organizers dump.

### Slide 2: Our Novelty — Independent Veto Authority, Not a Vote
* **Say:** *"Our core scientific contribution is simple yet transformative: **We do not let the sentiment detector vote.**"*
* Instead of fusing sentiment, we build an **Independent Sentiment-Anomaly Veto Agent**:
  1. It is trained and calibrated **only on calm, normal trading periods** to learn baseline language and volume distributions.
  2. It computes a statistical **Anomaly Index** (z-score deviation).
  3. When an unnatural surge occurs, it exercises **executive veto power**—superseding the trading agents to force an immediate exit or reversal (`FORCE_REVERSE`), regardless of how confident the trading agents are.

### Slide 3: Empirical Validation & Datasets
* **Say:** *"We did not use synthetic or toy data. We evaluated our system on 1,520 real pump-and-dump events across 228 cryptocurrency pairs on Binance (SYM/BTC) from two IEEE and Elsevier datasets (La Morgia 2020 and Ardia 2024)."*
* Data split strictly chronologically (70% train, 15% validation, 15% test) with zero look-ahead data leakage.

### Slide 4: Core Empirical Breakthrough (The Ablation Results)
* **Show the Ablation Comparison Table:**
  - **Manipulation Survival Rate:** Baseline = **0.0%** | TradeSwarm Veto-Enabled = **100.0%**
  - **Drawdown During Pumps Only:** Baseline suffered a **40.72% crash** | TradeSwarm suffered **0.00%**
  - **Overall Max Drawdown:** Slashed from **57.83% down to 26.20%** (a 31.63% risk reduction!)
  - **False Positive Rate:** Only **9.09%** on calm periods.

---

## 🖥️ 2. Live Dashboard Demo Walkthrough (Screen-by-Screen)

Launch the dashboard:
```bash
streamlit run app/dashboard.py
```

### Screen 1: Architecture Overview
* Point out the **Two-Track Pipeline**:
  - Track 1: Trading Agents (Market, Sentiment, Prediction) -> Decision Agent -> Risk Agent.
  - Track 2: Independent Veto Agent running in parallel.
  - Show how the Override Loop intercepts the trade when $\mathcal{A}_t \ge \theta^*$.

### Screen 2: Real-Time Event Simulation
1. Select a real pump-and-dump event from the dropdown (e.g. `NAV`, `PPT`, or `BRD`).
2. Toggle **Veto Enabled: OFF**:
   - Show how the Majority Vote says **BUY** because of the momentum and hype text.
   - Show how the portfolio enters a position right before the crash, suffering a massive drawdown.
3. Toggle **Veto Enabled: ON**:
   - Watch the **Anomaly Index Gauge** spike into the Red Zone ($> 1.25$).
   - Watch the alert trigger: `OVERRIDE TRIGGERED: Trade reversed to FORCE_REVERSE`.
   - Show how the portfolio exits safely with **0.00% drawdown**!

### Screen 3: Ablation Study Comparison
* Show the interactive equity curve plot comparing **Baseline (Red)**, **Veto-Disabled (Yellow)**, and **Veto-Enabled (Green)**.
* Show the statistical significance metrics ($p$-values and survival rate bars).

---

## ❓ 3. Likely Viva Questions & High-Scoring Model Answers

#### Q1: "Why did you use FinBERT instead of regular BERT or VADER?"
> **Answer:** *"FinBERT is pre-trained specifically on financial corpora (Financial PhraseBank and SEC filings), so it understands financial semantics. More importantly, we explicitly fixed a documented flaw in our primary base paper (Singh 2026), which erroneously trained its RL layer on randomized synthetic sentiment. TradeSwarm feeds real domain-specific polarity into the agent pipeline from day one."*

#### Q2: "How did you choose the Veto Threshold ($\theta = 1.25$)? Did you overfit on the test set?"
> **Answer:** *"No, sir/ma'am. We strictly enforced Constraint #2 of our specification: **The threshold was calibrated exclusively on the chronological validation split** using an F1-score grid sweep. It was never tuned on the test set. The 100% survival rate on the test set is a true out-of-sample result."*

#### Q3: "What prevents the Veto Agent from constantly blocking valid trades (false alarms)?"
> **Answer:** *"Because the baseline distribution ($\mu, \sigma$) is fitted per-coin over calm trading windows. High volume with normal market vocabulary doesn't produce an extreme z-score. An anomaly requires a simultaneous surge in message arrival frequency and abnormal coordination urgency. Our empirical test results show a False Positive Rate of only 9.09% (over 90% True Negative specificity)."*

#### Q4: "What is the difference between your Risk Agent and your Veto Agent?"
> **Answer:** *"The Risk Agent (based on MSPM Eq. 8) operates on internal portfolio and market state—it throttles position sizing based on historical return volatility and trailing drawdown. However, the Veto Agent operates on external communication signals (Telegram message coordination text and velocity). The Risk Agent manages continuous exposure, while the Veto Agent executes emergency discrete overrides against manipulation."*

