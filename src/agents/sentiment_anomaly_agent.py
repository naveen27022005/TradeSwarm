"""
Phase 4: Independent Sentiment-Anomaly Veto Agent (Core Novelty).

Unlike baseline sentiment agents that participate in a majority vote,
the Sentiment-Anomaly Agent holds independent veto authority.

Key Principles:
1. Calibrated strictly on normal (non-event) trading windows to learn
   the baseline distribution of sentiment polarity and message frequency.
2. Computes the composite Anomaly Index (z-score deviation of sentiment + volume).
3. Evaluates Anomaly Index against threshold theta (tuned on validation set only).
4. If Anomaly Index > theta: emits FORCE_REVERSE / FORCE_HEDGE with is_veto=True.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.agents.base_agent import ActionType, AgentSignal, BaseAgent
from src.agents.sentiment_agent import SentimentAgent


class SentimentAnomalyAgent(BaseAgent):
    """
    Independent Sentiment-Anomaly Veto Agent.
    Operates outside the profit-seeking RL reward loop.
    """

    def __init__(
        self,
        name: str = "SentimentAnomalyAgent",
        veto_threshold: float = 2.0,  # Calibrated on validation set
        sentiment_weight: float = 0.6,
        volume_weight: float = 0.4,
        forced_action: ActionType = "FORCE_REVERSE",
    ):
        super().__init__(name=name)
        self.veto_threshold = veto_threshold
        self.w_sentiment = sentiment_weight
        self.w_volume = volume_weight
        self.forced_action = forced_action

        self.sentiment_agent = SentimentAgent(threshold=0.05, use_finbert=False)
        # Per-coin baseline parameters: coin -> {mean_sent, std_sent, mean_vol, std_vol}
        self.coin_baselines: Dict[str, Dict[str, float]] = {}
        # Global fallback baseline for coins without dedicated history
        self.global_baseline: Dict[str, float] = {
            "mean_sent": 0.0,
            "std_sent": 0.15,
            "mean_vol": 5.0,
            "std_vol": 3.0,
        }
        self.is_calibrated = False

    def calibrate(self, train_df: pd.DataFrame):
        """
        Calibrate baseline statistics strictly on non-manipulation windows
        (label == 'normal_trading').
        """
        normal_windows = train_df[train_df["label"] == "normal_trading"]
        if len(normal_windows) == 0:
            normal_windows = train_df

        coin_sentiments: Dict[str, List[float]] = {}
        coin_volumes: Dict[str, List[float]] = {}
        all_sentiments: List[float] = []
        all_volumes: List[float] = []

        import json

        for _, row in normal_windows.iterrows():
            coin = row["coin_symbol"]
            msgs = json.loads(row["telegram_messages_json"]) if "telegram_messages_json" in row else []
            msg_count = len(msgs)
            sent_score = self.sentiment_agent.score_messages(msgs)

            if coin not in coin_sentiments:
                coin_sentiments[coin] = []
                coin_volumes[coin] = []

            coin_sentiments[coin].append(sent_score)
            coin_volumes[coin].append(msg_count)
            all_sentiments.append(sent_score)
            all_volumes.append(msg_count)

        # Global fallback
        if all_sentiments:
            self.global_baseline = {
                "mean_sent": float(np.mean(all_sentiments)),
                "std_sent": max(0.05, float(np.std(all_sentiments))),
                "mean_vol": float(np.mean(all_volumes)),
                "std_vol": max(1.0, float(np.std(all_volumes))),
            }

        # Per-coin baselines
        for coin in coin_sentiments:
            s_arr = np.array(coin_sentiments[coin])
            v_arr = np.array(coin_volumes[coin])
            self.coin_baselines[coin] = {
                "mean_sent": float(np.mean(s_arr)),
                "std_sent": max(0.05, float(np.std(s_arr))),
                "mean_vol": float(np.mean(v_arr)),
                "std_vol": max(1.0, float(np.std(v_arr))),
            }

        self.is_calibrated = True

    def compute_anomaly_index(
        self, coin_symbol: str, messages: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """
        Compute composite Anomaly Index:
        Z = w_sent * Z_sentiment + w_vol * Z_volume

        Returns
        -------
        (composite_anomaly_index, z_sentiment, z_volume)
        """
        baseline = self.coin_baselines.get(coin_symbol, self.global_baseline)

        current_sentiment = self.sentiment_agent.score_messages(messages)
        current_volume = float(len(messages))

        z_sent = (current_sentiment - baseline["mean_sent"]) / baseline["std_sent"]
        z_vol = (current_volume - baseline["mean_vol"]) / baseline["std_vol"]

        # Only positive deviations (hype / sudden surge) contribute to pump anomaly
        z_sent_clipped = max(0.0, z_sent)
        z_vol_clipped = max(0.0, z_vol)

        composite_anomaly = self.w_sentiment * z_sent_clipped + self.w_volume * z_vol_clipped
        return float(composite_anomaly), float(z_sent), float(z_vol)

    def act(self, state: Dict[str, Any]) -> AgentSignal:
        """
        Assess current message state and decide whether to exercise veto.
        """
        coin = state.get("coin_symbol", "UNKNOWN")
        messages = state.get("messages", [])

        anomaly_index, z_sent, z_vol = self.compute_anomaly_index(coin, messages)
        is_veto = anomaly_index >= self.veto_threshold

        if is_veto:
            action = self.forced_action
            confidence = float(np.clip(anomaly_index / (self.veto_threshold * 2), 0.7, 1.0))
        else:
            action = "HOLD"
            confidence = float(1.0 - (anomaly_index / (self.veto_threshold + 1e-8)))

        return AgentSignal(
            agent_name=self.name,
            action=action,
            confidence=confidence,
            vote=0,
            raw_value=anomaly_index,
            metadata={
                "is_veto": is_veto,
                "anomaly_index": anomaly_index,
                "z_sentiment": z_sent,
                "z_volume": z_vol,
                "threshold": self.veto_threshold,
                "forced_action": self.forced_action if is_veto else None,
            },
        )

    def tune_threshold_on_validation(
        self, val_df: pd.DataFrame, candidate_thresholds: List[float] = None
    ) -> Dict[str, Any]:
        """
        Tune threshold theta on validation data to maximize F1-score
        (strictly respecting Constraint #2).
        """
        import json

        if candidate_thresholds is None:
            candidate_thresholds = list(np.arange(1.0, 4.0, 0.25))

        best_theta = 2.0
        best_f1 = -1.0
        best_stats = {}
        all_evals = []

        labels = []
        anomalies = []

        for _, row in val_df.iterrows():
            coin = row["coin_symbol"]
            msgs = json.loads(row["telegram_messages_json"]) if "telegram_messages_json" in row else []
            anomaly, _, _ = self.compute_anomaly_index(coin, msgs)
            is_pump = 1 if row["label"] == "pump_and_dump_event" else 0
            labels.append(is_pump)
            anomalies.append(anomaly)

        labels = np.array(labels)
        anomalies = np.array(anomalies)

        for theta in candidate_thresholds:
            preds = (anomalies >= theta).astype(int)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            tn = int(np.sum((preds == 0) & (labels == 0)))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

            stat = {
                "threshold": float(theta),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "precision": float(precision),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
            all_evals.append(stat)

            if f1 > best_f1:
                best_f1 = f1
                best_theta = theta
                best_stats = stat

        self.veto_threshold = float(best_theta)
        return {"optimal_threshold": float(best_theta), "best_stats": best_stats, "all_evals": all_evals}

