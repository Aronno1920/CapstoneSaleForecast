from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None  # type: ignore


class ProphetWrapper:
    def __init__(self, seasonality_mode: str = "additive"):
        self.seasonality_mode = seasonality_mode
        self.model = Prophet(seasonality_mode=seasonality_mode) if Prophet else None

    def fit(self, df: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("Prophet is not installed")
        self.model.fit(df)
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Prophet is not installed")
        return self.model.predict(future)

    def save(self, path: str | Path):
        # Prophet has built-in serialization via pickle
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str | Path) -> "ProphetWrapper":
        import joblib
        obj = ProphetWrapper()
        obj.model = joblib.load(path)
        return obj
