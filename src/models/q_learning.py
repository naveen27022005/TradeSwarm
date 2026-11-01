"""
Phase 2, Step 5: Tabular Q-Learning Refinement Layer (Singh Section II.E).

State space: 3-bit binary vector (Market, Sentiment, Prediction) -> 8 states (0-7).
Action space: 3 discrete actions: 0 = HOLD, 1 = BUY, 2 = SELL.
Hyperparameters:
- alpha (learning rate) = 0.1
- gamma (discount factor) = 0.9
- epsilon (exploration rate) = 0.1

Trained on REAL sentiment signals, correcting Singh's randomized bug.
"""
from pathlib import Path
import numpy as np


class QLearningAgent:
    """
    Tabular Q-learning refinement policy.
    """

    ACTIONS = ["HOLD", "BUY", "SELL"]
    ACTION_TO_IDX = {"HOLD": 0, "BUY": 1, "SELL": 2}
    IDX_TO_ACTION = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        num_states: int = 8,
        num_actions: int = 3,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        # Q-table: shape (8, 3)
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)

    @staticmethod
    def state_to_index(market_sig: int, sentiment_sig: int, pred_sig: int) -> int:
        """
        Encode 3 binary signals (0 or 1) into an integer state index in [0, 7].
        Index = market * 4 + sentiment * 2 + pred * 1
        """
        m = 1 if int(market_sig) > 0 else 0
        s = 1 if int(sentiment_sig) > 0 else 0
        p = 1 if int(pred_sig) > 0 else 0
        return (m << 2) | (s << 1) | p

    def choose_action(self, state_idx: int, explore: bool = True) -> str:
        """
        Select action via epsilon-greedy policy.
        """
        if explore and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            # Break ties randomly
            q_values = self.q_table[state_idx]
            max_val = np.max(q_values)
            best_actions = np.where(q_values == max_val)[0]
            action_idx = np.random.choice(best_actions)

        return self.IDX_TO_ACTION[action_idx]

    def update(
        self,
        state_idx: int,
        action: str,
        reward: float,
        next_state_idx: int,
        done: bool = False,
    ):
        """
        Apply Bellman equation Q-table update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        a_idx = self.ACTION_TO_IDX[action]
        current_q = self.q_table[state_idx, a_idx]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state_idx])
            target = reward + self.gamma * best_next_q

        self.q_table[state_idx, a_idx] += self.alpha * (target - current_q)

    def train_on_episode(
        self,
        states: list[int],
        returns: list[float],
        epochs: int = 5,
    ):
        """
        Train Q-table across a time-series episode.

        Reward formulation:
        - BUY: reward = return
        - SELL: reward = -return (profitable if price falls)
        - HOLD: reward = 0
        """
        if len(states) < 2 or len(returns) < len(states):
            return

        for _ in range(epochs):
            for t in range(len(states) - 1):
                s = states[t]
                action = self.choose_action(s, explore=True)
                ret = returns[t + 1]

                if action == "BUY":
                    reward = ret
                elif action == "SELL":
                    reward = -ret
                else:  # HOLD
                    reward = 0.0

                s_next = states[t + 1]
                done = (t == len(states) - 2)
                self.update(s, action, reward, s_next, done=done)

    def save(self, path: Path | str):
        """Save Q-table to npy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: Path | str):
        """Load Q-table from npy file."""
        self.q_table = np.load(path)

