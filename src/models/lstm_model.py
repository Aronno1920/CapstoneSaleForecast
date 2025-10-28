from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class _LSTMNet(nn.Module):
    def __init__(self, window: int, units: int):
        super().__init__()
        self.window = int(window)
        self.units = int(units)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.units, batch_first=True)
        self.fc1 = nn.Linear(self.units, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, 1)

    def forward(self, x):  # x: (B, T, 1)
        o, _ = self.lstm(x)
        last = o[:, -1, :]
        z = self.relu(self.fc1(last))
        y = self.out(z)
        return y


class LSTMWrapper:
    def __init__(self, window: int = 12, units: int = 32):
        self.window = int(window)
        self.units = int(units)
        self.model: "_LSTMNet | None" = None

    def _build(self) -> "_LSTMNet":
        if torch is None or nn is None:
            raise RuntimeError("torch is not installed")
        return _LSTMNet(self.window, self.units)

    def fit(self, y: pd.Series, epochs: int = 50, batch_size: int = 32):
        if torch is None:
            raise RuntimeError("torch is not installed")
        values = y.astype(float).to_numpy()
        if len(values) <= self.window:
            raise RuntimeError("Not enough data to train LSTM")
        X_list, Y_list = [], []
        for i in range(self.window, len(values)):
            X_list.append(values[i - self.window : i])
            Y_list.append(values[i])
        import numpy as np
        X = torch.tensor(np.array(X_list).reshape(-1, self.window, 1), dtype=torch.float32)
        Y = torch.tensor(np.array(Y_list).reshape(-1, 1), dtype=torch.float32)

        self.model = self._build()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        n = X.shape[0]
        for _ in range(int(epochs)):
            perm = torch.randperm(n)
            for i in range(0, n, int(batch_size)):
                idx = perm[i : i + int(batch_size)]
                xb = X[idx]
                yb = Y[idx]
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_next(self, window_vals) -> float:
        if torch is None:
            raise RuntimeError("torch is not installed")
        if self.model is None:
            raise RuntimeError("Model not trained")
        import numpy as np
        x = torch.tensor(np.array(window_vals, dtype=float).reshape(1, self.window, 1), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y = self.model(x)
        return float(y.detach().cpu().numpy().ravel()[0])

    def save(self, path: str | Path):
        if torch is None:
            raise RuntimeError("torch is not installed")
        if self.model is None:
            raise RuntimeError("Model not trained")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "window": self.window,
            "units": self.units,
        }, str(path))

    @staticmethod
    def load(path: str | Path) -> "LSTMWrapper":
        if torch is None:
            raise RuntimeError("torch is not installed")
        ckpt = torch.load(str(path), map_location="cpu")
        window = int(ckpt.get("window", 12))
        units = int(ckpt.get("units", 32))
        obj = LSTMWrapper(window=window, units=units)
        obj.model = _LSTMNet(obj.window, obj.units)
        obj.model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
        obj.model.eval()
        return obj
