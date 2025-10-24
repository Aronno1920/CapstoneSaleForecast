from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


class LightGBMWrapper:
    def __init__(self, params: dict | None = None):
        self.params = params or {"objective": "regression", "metric": "rmse"}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        train = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, train, num_boost_round=200)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return pd.Series(self.model.predict(X), index=X.index)

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Model not trained")
        self.model.save_model(str(path))

    @staticmethod
    def load(path: str | Path) -> "LightGBMWrapper":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        obj = LightGBMWrapper()
        booster = lgb.Booster(model_file=str(path))
        obj.model = booster
        return obj
