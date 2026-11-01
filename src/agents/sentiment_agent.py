"""
Phase 2, Step 3: Sentiment Agent (Base paper: Singh Table II).

Extracts sentiment polarity from real Telegram coordination messages:
Polarity S = P(positive) - P(negative) in [-1, +1]

Binary signal:
1 (BUY)  if S > threshold (default 0.05)
0 (HOLD/SELL) otherwise

Uses real FinBERT (ProsusAI/finbert) via HuggingFace transformers,
with an aligned crypto-financial lexicon fallback for high-throughput testing.
"""
import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.agents.base_agent import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)


# Domain-specific financial & crypto sentiment lexicon for fast fallback scoring
CRYPTO_SENTIMENT_LEXICON = {
    # High bullish / pump terminology (+1.0 to +0.5)
    "moon": 0.9,
    "buy": 0.8,
    "pump": 0.85,
    "rocket": 0.8,
    "target": 0.6,
    "profit": 0.7,
    "breakout": 0.75,
    "bull": 0.7,
    "holding": 0.5,
    "hodl": 0.5,
    "gain": 0.6,
    "gem": 0.8,
    "accumulate": 0.6,
    "soar": 0.8,
    "rally": 0.7,
    # High bearish / dump terminology (-1.0 to -0.5)
    "dump": -0.9,
    "sell": -0.85,
    "crash": -0.95,
    "loss": -0.7,
    "bear": -0.7,
    "drop": -0.6,
    "fall": -0.6,
    "scam": -0.95,
    "panic": -0.85,
    "rekt": -0.9,
    "down": -0.5,
    "dip": -0.3,
    # Neutral terms (0.0)
    "quiet": 0.0,
    "range": 0.0,
    "watching": 0.1,
    "wait": -0.1,
    "volume": 0.0,
}


class SentimentAgent(BaseAgent):
    """
    Sentiment Agent scoring financial text using FinBERT or calibrated NLP,
    conforming to BaseAgent.
    """

    def __init__(
        self,
        name: str = "SentimentAgent",
        threshold: float = 0.05,
        use_finbert: bool = False,
        model_name: str = "ProsusAI/finbert",
    ):
        super().__init__(name=name)
        self.threshold = threshold
        self.use_finbert = use_finbert
        self.model_name = model_name
        self._pipeline = None

        if self.use_finbert:
            self._load_finbert()

    def act(self, state: Dict[str, Any]) -> AgentSignal:
        """Process messages and emit standard AgentSignal."""
        messages = state.get("messages", [])
        polarity = self.score_messages(messages)
        vote = 1 if polarity > self.threshold else 0
        action = "BUY" if vote == 1 else ("SELL" if polarity < -self.threshold else "HOLD")
        confidence = float(np.clip(abs(polarity), 0.5, 1.0))

        return AgentSignal(
            agent_name=self.name,
            action=action,
            confidence=confidence,
            vote=vote,
            raw_value=polarity,
            metadata={"message_count": len(messages), "polarity": polarity},
        )

    def _load_finbert(self):
        """Lazy load FinBERT pipeline."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            logger.info(f"Loading FinBERT model: {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
            )
            logger.info("FinBERT successfully initialized.")
        except Exception as e:
            logger.warning(
                f"Could not load HuggingFace FinBERT ({e}). Falling back to calibrated NLP lexicon."
            )
            self._pipeline = None
            self.use_finbert = False

    def score_text_lexicon(self, text: str) -> float:
        """
        Calibrated lexicon scoring approximating FinBERT output in [-1, +1]
        with negation awareness (e.g. 'do not dump' -> positive).
        """
        if not text or not isinstance(text, str):
            return 0.0

        # Normalize punctuation and tokenize
        clean_text = text.lower().replace("!", " ").replace("?", " ").replace(".", " ").replace(",", " ")
        words = clean_text.split()
        if not words:
            return 0.0

        negation_words = {"not", "dont", "don't", "no", "never", "without"}
        scores = []
        is_negated = False

        for w in words:
            if w in negation_words:
                is_negated = True
                continue

            if w in CRYPTO_SENTIMENT_LEXICON:
                val = CRYPTO_SENTIMENT_LEXICON[w]
                if is_negated:
                    val = -val  # Invert sentiment (e.g. 'not dump' becomes bullish)
                    is_negated = False
                scores.append(val)
            else:
                # Reset negation if token is far
                is_negated = False

        if not scores:
            return 0.0

        raw_score = float(np.mean(scores))
        return float(np.clip(raw_score, -1.0, 1.0))

    def score_text_finbert(self, text: str) -> float:
        """
        Score a single string with FinBERT: P(positive) - P(negative).
        """
        if not self._pipeline:
            return self.score_text_lexicon(text)

        try:
            # FinBERT output: list of dicts [{'label': 'positive', 'score': p}, ...]
            results = self._pipeline(text[:512])[0]
            scores_by_label = {item["label"].lower(): item["score"] for item in results}
            p_pos = scores_by_label.get("positive", 0.0)
            p_neg = scores_by_label.get("negative", 0.0)
            polarity = p_pos - p_neg
            return float(polarity)
        except Exception as e:
            logger.warning(f"FinBERT scoring failed on text: {e}")
            return self.score_text_lexicon(text)

    def score_messages(self, messages: list[dict]) -> float:
        """
        Compute aggregate sentiment polarity across a list of Telegram message dicts.

        Parameters
        ----------
        messages : list[dict]
            Each dict has {"text": str, "timestamp": str, ...}

        Returns
        -------
        float: Mean sentiment polarity in [-1.0, +1.0]
        """
        if not messages:
            return 0.0

        scores = []
        for msg in messages:
            text = msg.get("text", "")
            if self.use_finbert and self._pipeline:
                s = self.score_text_finbert(text)
            else:
                s = self.score_text_lexicon(text)
            scores.append(s)

        return float(np.mean(scores)) if scores else 0.0

    def predict_signal(self, messages: list[dict]) -> int:
        """
        Produce a binary trade signal from recent Telegram messages.

        Returns
        -------
        int: 1 for BUY, 0 for HOLD/SELL
        """
        polarity = self.score_messages(messages)
        return 1 if polarity > self.threshold else 0


if __name__ == "__main__":
    agent = SentimentAgent(use_finbert=False)
    pump_messages = [
        {"text": "BUY BRD NOW! PUMP TARGET ON BINANCE 🚀🚀🚀"},
        {"text": "Huge profits coming, hold and do not dump!"},
    ]
    normal_messages = [
        {"text": "Market is quiet today. Waiting for BTC breakout."},
        {"text": "Just DCA-ing as usual. Nothing exciting."},
    ]
    dump_messages = [
        {"text": "SELL SELL SELL! Take profits immediately!"},
        {"text": "Price is crashing, panic sell now!"},
    ]

    print("Pump messages sentiment:", agent.score_messages(pump_messages), "Signal:", agent.predict_signal(pump_messages))
    print("Normal messages sentiment:", agent.score_messages(normal_messages), "Signal:", agent.predict_signal(normal_messages))
    print("Dump messages sentiment:", agent.score_messages(dump_messages), "Signal:", agent.predict_signal(dump_messages))
