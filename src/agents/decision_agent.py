"""
Phase 2, Step 4: Decision Agent (Base paper: Singh Table II).

Majority Voting Logic:
Votes:
- Market Agent: 1 (BUY) or 0 (HOLD/SELL)
- Sentiment Agent: 1 (BUY) or 0 (HOLD/SELL)
- Prediction Agent: 1 (BUY) or 0 (HOLD/SELL)

Aggregate Decision:
BUY  if sum(votes) >= 2
HOLD if sum(votes) == 1
SELL if sum(votes) == 0
"""
from typing import Any, Dict, List, Literal
from src.agents.base_agent import ActionType, AgentSignal, BaseAgent

Action = Literal["BUY", "HOLD", "SELL"]


class DecisionAgent(BaseAgent):
    """
    Majority-vote Decision Agent combining signals from upstream agents,
    conforming to BaseAgent.
    """

    def __init__(self, name: str = "DecisionAgent"):
        super().__init__(name=name)

    def act(self, state: Dict[str, Any]) -> AgentSignal:
        """
        Aggregate votes from signals in state['signals'].
        """
        signals: List[AgentSignal] = state.get("signals", [])
        if not signals:
            return AgentSignal(agent_name=self.name, action="HOLD", confidence=0.0, vote=0)

        total_votes = sum(s.vote for s in signals)
        majority_threshold = len(signals) / 2.0

        if total_votes >= 2:
            action = "BUY"
            vote = 1
        elif total_votes == 1:
            action = "HOLD"
            vote = 0
        else:
            action = "SELL"
            vote = 0

        avg_confidence = float(sum(s.confidence for s in signals) / len(signals))

        return AgentSignal(
            agent_name=self.name,
            action=action,
            confidence=avg_confidence,
            vote=vote,
            raw_value=float(total_votes),
            metadata={"votes": [s.vote for s in signals], "total_votes": total_votes},
        )

    @staticmethod
    def vote(market_sig: int, sentiment_sig: int, pred_sig: int) -> Action:
        """
        Compute aggregate majority vote.

        Parameters
        ----------
        market_sig : int (0 or 1)
        sentiment_sig : int (0 or 1)
        pred_sig : int (0 or 1)

        Returns
        -------
        Action: "BUY", "HOLD", or "SELL"
        """
        total = int(market_sig) + int(sentiment_sig) + int(pred_sig)
        if total >= 2:
            return "BUY"
        elif total == 1:
            return "HOLD"
        else:
            return "SELL"

    @staticmethod
    def action_to_int(action: Action) -> int:
        """Map action string to integer code: BUY: 1, HOLD: 0, SELL: -1."""
        mapping = {"BUY": 1, "HOLD": 0, "SELL": -1}
        return mapping[action]

    @staticmethod
    def int_to_action(code: int) -> Action:
        """Map integer code to action string."""
        mapping = {1: "BUY", 0: "HOLD", -1: "SELL"}
        return mapping.get(code, "HOLD")
