"""
Phase 3, Step 1: BaseAgent Interface (MSPM Modular Architecture).

Defines the abstract contract for all agents in TradeSwarm:
- Standardized AgentSignal dataclass
- Abstract BaseAgent with act(state: dict) -> AgentSignal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

ActionType = Literal["BUY", "HOLD", "SELL", "FORCE_HEDGE", "FORCE_REVERSE"]


@dataclass
class AgentSignal:
    """Standardized output signal produced by any TradeSwarm agent."""

    agent_name: str
    action: ActionType
    confidence: float = 1.0  # [0.0, 1.0]
    vote: int = 0  # 1 for BUY, 0 for HOLD/SELL
    raw_value: float = 0.0  # continuous indicator value (e.g. sentiment score, momentum)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "action": self.action,
            "confidence": self.confidence,
            "vote": self.vote,
            "raw_value": self.raw_value,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract Base Class for modular agents (following MSPM EAM/SAM pattern).
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def act(self, state: Dict[str, Any]) -> AgentSignal:
        """
        Process the environment state and return a standardized AgentSignal.

        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary containing relevant data such as:
            - 'price_df': pd.DataFrame with OHLCV data
            - 'messages': list of Telegram message dicts
            - 'portfolio': dict with capital, position, current_drawdown
            - 'timestamp': current simulation timestamp

        Returns
        -------
        AgentSignal
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"

