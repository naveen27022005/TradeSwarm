"""
Phase 2, Step 2: LSTM Price Predictor (Base paper: Singh Table I).

Architecture:
- 2 x LSTM layers (hidden_size=128, dropout=0.2)
- Linear(128 -> 64)
- ReLU
- Dropout(0.2)
- Linear(64 -> 1)

Optimizer: Adam (lr=0.001)
Loss: MSE
Evaluation Metrics: RMSE, MAE, R^2
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SinghLSTM(nn.Module):
    """
    Exact LSTM architecture specified in Singh (2026) Table I.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Take the output of the last timestep
        last_out = out[:, -1, :]
        h = self.fc1(last_out)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.fc2(h)
        return out


class LSTMPredictor:
    """
    High-level training and inference wrapper for SinghLSTM.
    """

    def __init__(
        self,
        lookback: int = 10,
        hidden_dim: int = 128,
        lr: float = 0.001,
        dropout: float = 0.2,
        device: str = None,
    ):
        self.lookback = lookback
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SinghLSTM(input_dim=1, hidden_dim=hidden_dim, dropout=dropout).to(
            self.device
        )
        self.scaler_mean = 0.0
        self.scaler_std = 1.0
        self.is_fitted = False

    def create_sequences(
        self, values: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a 1D price array into (X, y) sliding window pairs."""
        xs, ys = [], []
        for i in range(len(values) - self.lookback):
            x = values[i : i + self.lookback]
            y = values[i + self.lookback]
            xs.append(x)
            ys.append(y)
        if not xs:
            return torch.empty(0), torch.empty(0)
        X_tensor = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
        return X_tensor, y_tensor

    def fit(
        self,
        prices: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        """
        Fit LSTM on training price series.
        """
        prices = np.asarray(prices, dtype=np.float32)
        self.scaler_mean = float(np.mean(prices))
        self.scaler_std = float(np.std(prices))
        if self.scaler_std == 0:
            self.scaler_std = 1.0

        normalized = (prices - self.scaler_mean) / self.scaler_std
        X, y = self.create_sequences(normalized)
        if len(X) == 0:
            raise ValueError(f"Not enough price points for lookback={self.lookback}")

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(bx)

            avg_loss = epoch_loss / len(dataset)
            loss_history.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return loss_history

    def predict(self, recent_prices: np.ndarray) -> float:
        """
        Predict the next price given a sequence of recent prices.
        """
        if not self.is_fitted:
            # Fallback to naive prediction if not fitted yet
            return float(recent_prices[-1])

        prices = np.asarray(recent_prices[-self.lookback :], dtype=np.float32)
        normalized = (prices - self.scaler_mean) / self.scaler_std
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        x = x.to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x).item()

        # Denormalize
        pred_price = pred_norm * self.scaler_std + self.scaler_mean
        return float(pred_price)

    def evaluate(self, actual_prices: np.ndarray) -> dict[str, float]:
        """
        Compute RMSE, MAE, and R^2 on a price sequence.
        """
        prices = np.asarray(actual_prices, dtype=np.float32)
        norm_prices = (prices - self.scaler_mean) / self.scaler_std
        X, y = self.create_sequences(norm_prices)

        if len(X) == 0:
            return {"rmse": 0.0, "mae": 0.0, "r2": 0.0}

        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            preds_norm = self.model(X).cpu().numpy().flatten()
            actuals_norm = y.numpy().flatten()

        # Denormalize
        preds = preds_norm * self.scaler_std + self.scaler_mean
        actuals = actuals_norm * self.scaler_std + self.scaler_mean

        mae = float(np.mean(np.abs(preds - actuals)))
        mse = float(np.mean((preds - actuals) ** 2))
        rmse = float(np.sqrt(mse))

        # R^2 score
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        return {"rmse": rmse, "mae": mae, "r2": r2}

    def save(self, path: Path | str):
        """Save model checkpoint and scaler states."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "lookback": self.lookback,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: Path | str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.scaler_mean = ckpt["scaler_mean"]
        self.scaler_std = ckpt["scaler_std"]
        self.lookback = ckpt["lookback"]
        self.is_fitted = ckpt["is_fitted"]


if __name__ == "__main__":
    print("Testing LSTMPredictor training and inference...")
    # Synthetic trending random walk
    np.random.seed(42)
    series = 50.0 + np.cumsum(np.random.normal(0.05, 0.5, 300))
    train_series = series[:240]
    val_series = series[240:]

    predictor = LSTMPredictor(lookback=10, hidden_dim=64)
    losses = predictor.fit(train_series, epochs=20, batch_size=32, verbose=True)
    metrics = predictor.evaluate(val_series)
    next_price = predictor.predict(val_series[-10:])

    print(f"Final training loss: {losses[-1]:.6f}")
    print(f"Validation metrics: {metrics}")
    print(f"Last price: {val_series[-1]:.4f} -> Predicted next: {next_price:.4f}")

