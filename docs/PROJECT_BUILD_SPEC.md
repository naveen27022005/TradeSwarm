# A Multi-Agent Reinforcement Learning Architecture with Independent Sentiment-Anomaly Veto for Pump-and-Dump-Resistant Trading

## Build Specification for Agentic AI Implementation

**Document purpose:** This is a step-by-step implementation specification intended to be handed to an agentic AI coding tool (e.g. Claude Code, an autonomous dev agent, or similar). It defines every phase of the build, the exact actions required in each phase, and the expected outcome of each action.

**Operating protocol for the AI agent building this project — read before starting any phase:**

1. Before writing or modifying any code, state in plain language: (a) which phase and step this is, (b) exactly what files/components will be created or changed, (c) what the expected outcome/behavior will be once the step is complete, and (d) how success will be verified.
2. Do not proceed to implementation until this explanation has been presented.
3. After implementing a step, verify the stated outcome actually occurred (run the code, check the output, inspect the file) before marking the step complete and moving to the next one.
4. If a step's outcome does not match what was predicted, stop, explain the discrepancy, and propose a fix before continuing.
5. Never skip ahead to a later phase while an earlier phase has unresolved validation failures.
6. Keep a running `PROGRESS_LOG.md` file, appending one entry per completed step: step name, action taken, outcome observed, validation result.

---

## 1. Project Overview

**Title:** A Multi-Agent Reinforcement Learning Architecture with Independent Sentiment-Anomaly Veto for Pump-and-Dump-Resistant Trading

**One-sentence summary:** A team of reinforcement-learning trading agents makes buy/sell/hold decisions from price action, while a separately-trained Sentiment-Anomaly Agent monitors real coordination text from cryptocurrency pump-and-dump groups and holds independent veto authority to force a hedge or reversal the moment it detects statistically anomalous sentiment — regardless of what the trading agents conclude.

### 1.1 Problem Statement

Existing multi-agent trading systems treat sentiment as one input among many, fused into either:
- a majority vote alongside other agents (sentiment can be outvoted by two other agents even during a real manipulation event), or
- a shared state vector fed into an RL policy (sentiment can be down-weighted or ignored by a network optimizing purely for profit).

Neither design can guarantee a response to manipulation, because the sentiment signal has no authority independent of the profit-seeking objective that manipulation is designed to exploit.

### 1.2 Proposed Solution

Give the sentiment/hype detector **veto power**, not a vote. A dedicated agent:
1. Is trained and evaluated independently of the trading agents' reward signal.
2. Continuously scores real manipulation-coordination text (not generic news) against a rolling per-asset baseline.
3. Computes a Sentiment-Anomaly Index (a statistical deviation measure, not raw polarity).
4. When the index crosses a threshold, forces a hedge or position reversal — overriding the trading agents' recommendation entirely.

### 1.3 Novelty Claim (for the paper's contribution statement)

> Prior multi-agent RL trading systems (including our primary base paper) fuse sentiment into a shared decision space, where a sufficiently coordinated manipulation can still influence the aggregate decision. This work gives sentiment-anomaly detection independent veto authority — a separately-trained agent that can override the trading agents regardless of their confidence — closing the specific gap that allows pump-and-dump manipulation to succeed against sentiment-informed but sentiment-fused systems.

---

## 2. Base Papers and How Each Maps to This Implementation

| Paper | Role in this project |
|---|---|
| **Khushvir Singh, "Agentic AI for Intelligent Stock Market Prediction: A Multi-Agent Framework Combining LSTM, Transformer, XGBoost, FinBERT, and Reinforcement Learning"** (Research Square, 2026) | **Primary base paper.** Its 4-agent design (Market Agent, Sentiment Agent, Prediction Agent, Decision Agent via majority vote) plus Q-learning refinement layer is the **baseline/control system** this project must reproduce first, then extend. Its LSTM configuration (2×LSTM(128), dropout 0.2, 600 epochs, Adam lr=0.001) is reused directly for the Prediction Agent. Its documented flaw — training the RL layer on **randomized sentiment instead of real FinBERT output** — must NOT be repeated; real sentiment scores must flow into training from the start. |
| **Huang & Tanaka, "MSPM: A Modularized and Scalable Multi-Agent Reinforcement Learning-Based System for Financial Portfolio Management"** (arXiv:2102.03502) | **Secondary base paper.** Provides: (a) the modular Evolving-Agent-Module / Strategic-Agent-Module (EAM/SAM) design pattern to follow for clean agent separation; (b) the "news buzz" feature (normalized change in news volume) as a starting template for the anomaly-index calculation; (c) the ablation methodology (EAM-enabled vs. EAM-disabled back-testing) that this project's veto-enabled vs. veto-disabled evaluation directly mirrors. |

---

## 3. Datasets

### 3.1 Primary dataset — real, labeled pump-and-dump events with real market data

**Source:** La Morgia et al., "Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations" (ICCCN 2020)
**Repository:** `https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset`

Contents to use:
- `pump_telegram.csv` — real, documented pump-and-dump events with timestamps and target coin, organized by real Telegram groups
- `group.csv` — metadata mapping each event to its organizing Telegram group
- Provided script to pull the actual historical Binance trading data (SYM/BTC pairs) for each labeled event window
- `labeled features` folder — pre-engineered features usable as a sanity-check baseline

**Important scope note:** all pairs are quoted in SYM/BTC, not vs. USD. Design the price-feature pipeline around this from the start.

### 3.2 Secondary dataset — real coordination text for sentiment scoring

**Source:** Ardia & Bluteau, "Twitter and cryptocurrency pump-and-dumps" (International Review of Financial Analysis, 2024)
**Repository:** `https://github.com/ArdiaD/PumpDump`

Contents to use: scraped real Telegram messages (with timestamps) from 100+ real pump-and-dump coordination groups. This is the text the Sentiment-Anomaly Agent scores — real manipulation language, not generic news headlines.

### 3.3 Data reconciliation requirement

Before any modeling begins, build a single aligned table per event with the schema:

```
{
  coin_symbol: str,
  pump_start_time: datetime,
  dump_confirmed_time: datetime,
  price_series: DataFrame[timestamp, open, high, low, close, volume],
  telegram_messages: List[{timestamp, group_id, text}],
  label: "pump_and_dump_event"
}
```

Non-event windows (ordinary trading periods, same coins, no documented manipulation) must also be sampled in equal or greater proportion, to give the veto agent negative examples and to make the evaluation's false-positive rate measurable.

---

## 4. System Architecture

```
                     ┌─────────────────────┐
   Market Data ─────▶│   Analysis Agent     │──┐
   (price/volume)    │  (RL: price action)  │  │
                     └─────────────────────┘  │
                                               │
                     ┌─────────────────────┐  ▼
   Portfolio State ─▶│    Risk Agent        │──▶  Proposed Trade
                     │ (position/vol limits)│  │  (BUY/SELL/HOLD)
                     └─────────────────────┘  │
                                               │
   ┌───────────────────────────────────────┐  │
   │      SENTIMENT-ANOMALY AGENT          │  │
   │  (independent — not in the vote loop) │  │
   │  Input: real Telegram coordination    │  │
   │  text scored against rolling per-coin │  │
   │  baseline → Anomaly Index             │  │
   └───────────────┬───────────────────────┘  │
                    │                          │
                    ▼                          ▼
            ┌───────────────────────────────────────┐
            │            OVERRIDE LOOP               │
            │  If Anomaly Index > threshold:         │
            │    FORCE hedge/reversal regardless      │
            │    of Analysis+Risk output              │
            │  Else: pass through proposed trade      │
            └───────────────┬────────────────────────┘
                             ▼
                      Final Trade Decision
```

Each agent must be implemented as an independent module sharing a common interface (mirroring MSPM's `BaseAgent`/EAM pattern), so any agent can be replaced, retrained, or disabled (for ablation) without touching the others.

---

## 5. Phase-by-Phase Implementation Plan

Each phase below has an **Objective**, **Actions**, **Expected Outcome**, **Validation**, and **Tools**. The agent building this must present all five before writing code for that phase.

---

### Phase 0 — Environment Setup & Repository Structure

**Objective:** Establish a clean, reproducible project structure before any modeling work begins.

**Actions:**

1. Create the repository folder structure:
   ```
   /data/raw/            (downloaded datasets, untouched)
   /data/processed/      (reconciled event tables)
   /src/agents/          (one file per agent)
   /src/data/            (data loading + reconciliation scripts)
   /src/eval/            (evaluation + ablation scripts)
   /src/models/          (LSTM, FinBERT wrapper, RL policy code)
   /notebooks/           (exploratory analysis only — no production logic)
   /results/             (metrics, plots, tables)
   /docs/                (this spec, progress log, paper drafts)
   requirements.txt
   PROGRESS_LOG.md
   ```
2. Set up a Python virtual environment; pin dependencies: `torch`, `transformers` (FinBERT), `pandas`, `numpy`, `scikit-learn`, `python-binance` or `ccxt`, `matplotlib`/`seaborn`, `pytest`.
3. Initialize git with `.gitignore` excluding `/data/raw` and large model checkpoints.

**Expected Outcome:** A runnable, empty project skeleton; `pip install -r requirements.txt` succeeds; `pytest` runs (with zero tests, exit code 0).

**Validation:** Directory listing matches the structure above; environment installs without errors.

**Tools:** `python -m venv`, `pip`, `git`.

---

### Phase 1 — Data Acquisition & Reconciliation

**Objective:** Produce the single aligned event table described in Section 3.3, with real price data and real coordination text merged per event.

**Actions:**

1. Clone/download the La Morgia `pump-and-dump-dataset` repo; run its provided script to pull historical Binance data for every labeled event.
2. Clone/download the Ardia & Bluteau `PumpDump` repo; load the scraped Telegram messages.
3. Write `src/data/reconcile.py`: for each event in `pump_telegram.csv`, match the target coin and time window against the Telegram message corpus (by group ID where available, else by coin symbol mention and timestamp proximity), and assemble the schema from Section 3.3.
4. Sample an equal-or-greater number of non-event (normal trading) windows for the same coins, with no matching manipulation label, to serve as negative examples.
5. Split into train/validation/test sets **chronologically** (no shuffling — mirrors the no-look-ahead-leakage practice in the Singh paper), e.g. 70/15/15 by event date.
6. Save processed tables to `/data/processed/` as parquet files.

**Expected Outcome:** Three parquet files (`train.parquet`, `val.parquet`, `test.parquet`), each row one event or non-event window with price series and associated text, no missing critical fields, positive/negative label counts reported.

**Validation:** Print and manually inspect label balance, date ranges of each split (confirm no overlap/leakage), and a spot-check of 3 events showing the price series visibly containing a spike/dump matching the documented timestamp.

**Tools:** `pandas`, `ccxt`/`python-binance`, the two GitHub repos' own scripts.

---

### Phase 2 — Baseline System Reproduction (Control Condition)

**Objective:** Reproduce a system equivalent to Singh's 4-agent majority-vote architecture. This is the scientific control — the system your veto architecture must be shown to outperform specifically during manipulation events.

**Actions:**

1. Implement `src/agents/market_agent.py`: momentum/volatility signal exactly as in Singh's Table II (5-day momentum, 10-day return std → binary signal).
2. Implement `src/models/lstm_predictor.py`: the LSTM architecture and training config from Singh's Table I (2×LSTM(128, dropout=0.2) → Linear(64) → ReLU → Dropout(0.2) → Linear(1), 600 epochs, Adam lr=0.001).
3. Implement `src/agents/sentiment_agent.py`: FinBERT scoring (`P(positive) − P(negative)`) on the Telegram text from Phase 1 — **use real scores, not randomized placeholders** (explicitly correcting Singh's documented flaw).
4. Implement `src/agents/decision_agent.py`: majority vote exactly as Singh's Table II (`BUY if market + sentiment + (pred > last) ≥ 2`).
5. Implement the Q-learning refinement layer per Singh's Section II.E (tabular Q-learning, α=0.1, γ=0.9, 3-bit state space) — trained on the **real** sentiment signal this time.
6. Train and evaluate this baseline system on the Phase 1 train/test splits.

**Expected Outcome:** A fully working baseline system producing BUY/SELL/HOLD decisions, with RMSE/MAE/R² for the LSTM component and CAGR/Sharpe/max-drawdown for the full backtest, broadly consistent in shape with Singh's reported ranges (exact numbers will differ due to different asset universe — crypto vs. AAPL).

**Validation:** Confirm LSTM converges (loss curve decreasing, R² > 0.8 on validation); confirm the Decision Agent's majority-vote logic unit-tested against hand-computed examples; confirm backtest runs end-to-end without errors and produces a portfolio equity curve.

**Tools:** `torch`, `transformers` (FinBERT: `ProsusAI/finbert` or similar pretrained checkpoint), `pandas`.

---

### Phase 3 — Modular Agent Refactor (MSPM-style)

**Objective:** Refactor Phase 2's agents into MSPM's clean modular pattern so agents are independently swappable, and so the Risk Agent and independent veto layer can be added without touching existing code.

**Actions:**

1. Define a common `BaseAgent` abstract interface (mirroring MSPM's `act(state)` pattern): every agent implements a single required method taking a state and returning a signal.
2. Refactor Market/Sentiment/Prediction/Decision agents from Phase 2 to inherit from `BaseAgent`.
3. Implement `src/agents/risk_agent.py`: position-sizing and volatility/drawdown limit checks (new — not in Singh's paper; use MSPM's risk-adjusted reward formulation, Eq. 8, as a template: penalize deviation from target allocation and recent volatility).
4. Write unit tests confirming each agent can be instantiated, called, and swapped independently without breaking the orchestrator.

**Expected Outcome:** Same functional behavior as Phase 2's baseline, now organized so any single agent can be replaced or disabled via configuration, not code edits.

**Validation:** Run the full Phase 2 backtest again through the refactored code; results must match Phase 2 outputs numerically (regression test) before proceeding.

**Tools:** Python ABCs, `pytest`.

---

### Phase 4 — Independent Sentiment-Anomaly Veto Agent (Core Novelty)

**Objective:** Build the project's actual contribution — a Sentiment-Anomaly Agent with independent veto authority, distinct from the Phase 2 Sentiment Agent (which only contributes one vote).

**Actions:**

1. Implement `src/agents/sentiment_anomaly_agent.py`: for each coin, maintain a rolling baseline (mean, standard deviation) of FinBERT sentiment scores **and** message volume, computed over a trailing window (e.g. 30–90 days) of normal (non-event) periods only.
2. Compute the **Anomaly Index** as a z-score-style deviation of current sentiment/volume against that per-coin baseline (extending MSPM's "news buzz" concept into a formal anomaly statistic).
3. Define and justify a veto threshold (e.g. Anomaly Index exceeding N standard deviations); this threshold should be tuned on the validation set only, never the test set.
4. Implement the override logic: when the Anomaly Index crosses the threshold, the agent emits a `FORCE_HEDGE` or `FORCE_REVERSE` action that supersedes whatever the Analysis+Risk agents propose.
5. Critically: **train/calibrate this agent's baseline statistics only on non-manipulation windows**, so it never learns to "expect" pumps as normal — this is what gives it independence from the profit-seeking objective the other agents optimize.

**Expected Outcome:** A working agent that, given a coin and current text/price window, outputs an Anomaly Index and a veto decision (boolean + forced action if true).

**Validation:** Run the agent over the Phase 1 test set's labeled pump-and-dump events and non-event windows; report true positive rate (vetoed during real events) and false positive rate (vetoed during normal periods) separately — this table is a required result for the paper.

**Tools:** `pandas`, `scipy.stats` (z-score/anomaly statistics), `transformers`.

---

### Phase 5 — Integration: The Override Loop

**Objective:** Wire the Phase 4 veto agent into the Phase 3 modular system as an independent authority, not a fifth vote.

**Actions:**

1. Implement `src/orchestrator.py`: at each timestep, first compute the Analysis Agent + Risk Agent's proposed trade as before; separately and independently compute the Sentiment-Anomaly Agent's Anomaly Index and veto flag.
2. If veto flag is true, override the proposed trade with the forced hedge/reversal action, and log that an override occurred (timestamp, coin, anomaly index value, what the trade would have been without the override).
3. If false, pass the proposed trade through unchanged.
4. Add a configuration flag `VETO_ENABLED = True/False` at the orchestrator level — this single flag is what makes the Phase 6 ablation study possible without duplicating code.

**Expected Outcome:** A single end-to-end system that can be run in two modes (veto-enabled / veto-disabled) via one config flag, producing a full trade log and equity curve for each mode.

**Validation:** Run both modes on the same test data; confirm they produce identical trades except at timesteps where the veto fired (spot check the override log against a few known pump-and-dump events from Phase 1's labels).

**Tools:** Python config management (e.g. a simple `config.yaml` or `dataclass`).

---

### Phase 6 — Evaluation: The Ablation Study

**Objective:** Produce the core empirical result of the paper — quantified evidence that independent veto authority improves manipulation resistance compared to the fused-sentiment baseline.

**Actions:**

1. Run three system configurations on the same held-out test set (mirroring MSPM's EAM-enabled vs. EAM-disabled comparison):
   - **Baseline** (Phase 2 system — sentiment as one vote)
   - **Veto-disabled** (Phase 5 system with `VETO_ENABLED = False` — sentiment fused but not authoritative)
   - **Veto-enabled** (Phase 5 system with `VETO_ENABLED = True` — full novel architecture)
2. Compute standard metrics for all three: CAGR, Sharpe Ratio, Maximum Drawdown, RMSE/MAE/R² for the price predictor.
3. Compute manipulation-specific metrics (new, required for this paper's contribution):
   - **Manipulation-episode survival rate**: % of labeled pump-and-dump events where the system avoided or reversed its position before the documented dump.
   - **Drawdown during manipulation windows only** (isolate performance specifically during the labeled events, not overall).
   - **False-positive veto rate**: % of normal (non-event) periods incorrectly vetoed.
4. Run statistical significance tests (e.g. paired comparison across events, following MSPM's use of Mann–Whitney U where distributions are non-normal) comparing veto-enabled vs. the other two configurations.
5. Produce plots: equity curves for all three configurations overlaid; a case-study plot of at least 2–3 individual pump-and-dump events showing price, Anomaly Index, and the system's trade decisions across the event window (mirroring the Singh paper's Fig. 14 trade-signal-overlay style).

**Expected Outcome:** A results table and set of plots directly comparable in structure to both base papers' Table III/IV (Singh) and Table 3/6 (MSPM), plus the new manipulation-specific metrics table.

**Validation:** All three configurations must have run on identical data splits; check that veto-enabled shows measurably better drawdown/survival-rate specifically during manipulation windows (if it does not, this is a real finding to report honestly, not to hide — revisit Phase 4's threshold tuning if needed).

**Tools:** `pandas`, `matplotlib`/`seaborn`, `scipy.stats`.

---

### Phase 7 — Documentation & Paper Writing

**Objective:** Convert the implementation and results into the deliverables required by the institutional review pipeline.

**Actions:**

1. Update `PROGRESS_LOG.md` into a clean project report covering all phases.
2. Draft the paper sections in this order: Introduction/Problem (Section 1 of this spec) → Related Work (base papers + literature survey already compiled) → Proposed Architecture (Section 4) → Methodology (Phases 1–5) → Results (Phase 6 tables/plots) → Discussion (be honest about false-positive rate and any limitations, following the Singh paper's model of transparent limitation reporting) → Conclusion.
3. Prepare the Review 2 demo script: a live or recorded run of the veto-enabled system reacting to a real historical pump-and-dump event from the test set.

**Expected Outcome:** A complete draft paper, a working demo, and documentation sufficient for a new team member (or reviewer) to reproduce every result from this spec alone.

**Validation:** Have a teammate not involved in a given phase attempt to run that phase's code from the spec alone; any point of confusion is a documentation gap to fix.

**Tools:** LaTeX/Word for the paper, whatever the demo requires.

---

## 6. Institutional Review Milestone Mapping

| Review | Requirement | Phases that should be complete |
|---|---|---|
| 0th Review | Title, abstract, guide sign-off | Sections 1–4 of this spec (already complete) |
| 1st Review | Lit review + 30% completion | Phases 0–2 (baseline reproduced, data pipeline working) |
| 2nd Review | 70% + demo + draft paper | Phases 3–6 (full architecture built, ablation run, draft paper) |
| Final Viva | Full prototype + publication | Phase 7 complete, paper submitted/presented |

---

## 7. Non-Negotiable Constraints (flag if any agentic tool tries to skip these)

1. Do not use synthetic/injected manipulation events as the primary evaluation data — real labeled events (Section 3.1) are required.
2. Do not tune the veto threshold (Phase 4, Action 3) on the test set under any circumstances.
3. Do not repeat Singh's randomized-sentiment-in-training bug — real FinBERT scores must be used in all training loops from Phase 2 onward.
4. Do not merge the Sentiment-Anomaly Agent's logic into the same reward-optimization loop as the Analysis/Risk agents — its independence is the entire point of the architecture and the paper's novelty claim.
5. Every phase's validation step must pass before the next phase begins; do not parallelize phases 1→6 speculatively.
