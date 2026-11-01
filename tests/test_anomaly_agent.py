"""
Validation tests for Phase 4: Sentiment-Anomaly Veto Agent (Core Novelty).

Verifies:
1. Baseline calibration runs strictly on normal_trading windows
2. Anomaly Index computation accurately reflects hype + volume surges
3. Threshold tuning executes on validation split only (Constraint #2)
4. Veto override trigger behavior (is_veto=True, action=FORCE_REVERSE)
5. Test split evaluation: Reports True Positive Rate and False Positive Rate table
"""
import numpy as np
import pandas as pd
import pytest

from src.agents.sentiment_anomaly_agent import SentimentAnomalyAgent
from src.data.split_data import PROCESSED_DIR


@pytest.fixture(scope="session")
def split_data():
    """Load train, val, and test parquet splits."""
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    return train_df, val_df, test_df


def test_calibration_on_normal_windows(split_data):
    """Verify calibration consumes normal windows and initializes baselines."""
    train_df, _, _ = split_data
    agent = SentimentAnomalyAgent()
    agent.calibrate(train_df)

    assert agent.is_calibrated is True
    assert len(agent.coin_baselines) > 0
    assert "mean_sent" in agent.global_baseline
    assert "std_sent" in agent.global_baseline
    assert agent.global_baseline["std_sent"] > 0


def test_anomaly_index_contrast(split_data):
    """Verify anomaly index is higher on pump events than normal trading."""
    train_df, _, _ = split_data
    agent = SentimentAnomalyAgent()
    agent.calibrate(train_df)

    pump_msgs = [
        {"text": "BUY BRD NOW! PUMP STARTING ON BINANCE! 🚀🚀🚀"},
        {"text": "MASSIVE GAINS INCOMING! GO GO GO!"},
        {"text": "TARGET 300% BUY FAST AND HOLD!"},
        {"text": "PUMP IN PROGRESS! DO NOT DUMP!"},
        {"text": "HUGE VOLUME SURGE BUY NOW!"},
        {"text": "TO THE MOON 🌙 BUY BUY BUY!"},
    ]

    normal_msgs = [
        {"text": "Market is quiet today. Waiting for BTC breakout."},
        {"text": "Just DCA-ing as usual. Nothing exciting."},
    ]

    pump_anomaly, _, _ = agent.compute_anomaly_index("BRD", pump_msgs)
    norm_anomaly, _, _ = agent.compute_anomaly_index("BRD", normal_msgs)

    assert pump_anomaly > norm_anomaly, (
        f"Pump anomaly ({pump_anomaly}) should be greater than normal ({norm_anomaly})"
    )


def test_validation_threshold_tuning(split_data):
    """Verify threshold tuning optimizes on validation split (Constraint #2)."""
    train_df, val_df, _ = split_data
    agent = SentimentAnomalyAgent()
    agent.calibrate(train_df)

    tuning_results = agent.tune_threshold_on_validation(val_df)
    opt_theta = tuning_results["optimal_threshold"]
    best_stats = tuning_results["best_stats"]

    assert 1.0 <= opt_theta <= 4.0
    assert best_stats["tpr"] > 0.5, f"Validation TPR too low: {best_stats['tpr']}"
    assert best_stats["fpr"] < 0.5, f"Validation FPR too high: {best_stats['fpr']}"
    print(f"\n[Validation Tuning] Optimal Theta: {opt_theta:.2f} | TPR: {best_stats['tpr']:.2%} | FPR: {best_stats['fpr']:.2%}")


def test_veto_override_signal(split_data):
    """Verify agent emits FORCE_REVERSE with is_veto=True when threshold breached."""
    agent = SentimentAnomalyAgent(veto_threshold=1.5)
    agent.global_baseline = {"mean_sent": 0.0, "std_sent": 0.1, "mean_vol": 3.0, "std_vol": 2.0}
    agent.is_calibrated = True

    pump_state = {
        "coin_symbol": "BRD",
        "messages": [{"text": "PUMP BUY NOW 🚀"} for _ in range(25)],
    }

    signal = agent.act(pump_state)
    assert signal.metadata["is_veto"] is True
    assert signal.action == "FORCE_REVERSE"
    assert signal.raw_value >= 1.5


def test_test_split_tpr_fpr_report(split_data):
    """
    Evaluate tuned agent on held-out test split.
    Outputs True Positive Rate and False Positive Rate table required for paper.
    """
    import json

    train_df, val_df, test_df = split_data
    agent = SentimentAnomalyAgent()
    agent.calibrate(train_df)
    tuning_res = agent.tune_threshold_on_validation(val_df)
    theta = agent.veto_threshold

    labels, preds, anomalies = [], [], []

    for _, row in test_df.iterrows():
        coin = row["coin_symbol"]
        msgs = json.loads(row["telegram_messages_json"])
        anomaly, _, _ = agent.compute_anomaly_index(coin, msgs)
        is_pump = 1 if row["label"] == "pump_and_dump_event" else 0
        labels.append(is_pump)
        preds.append(1 if anomaly >= theta else 0)
        anomalies.append(anomaly)

    labels = np.array(labels)
    preds = np.array(preds)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 50)
    print("PHASE 4: TEST SET VETO DETECTION PERFORMANCE")
    print("=" * 50)
    print(f"Optimal Veto Threshold (theta*): {theta:.2f}")
    print(f"Total Test Events: {len(test_df)}")
    print(f"True Positives (Vetoed Pumps):     {tp} / {tp + fn} (TPR: {tpr:.2%})")
    print(f"False Positives (Vetoed Normals):  {fp} / {fp + tn} (FPR: {fpr:.2%})")
    print(f"True Negatives:                    {tn}")
    print(f"False Negatives:                   {fn}")
    print("=" * 50)

    # Core assertion: TPR significantly exceeds FPR (discrimination ability)
    assert tpr > fpr, f"Agent failed to discriminate: TPR ({tpr:.2%}) <= FPR ({fpr:.2%})"
    assert tpr >= 0.70, f"Target TPR not met: {tpr:.2%} < 70%"

