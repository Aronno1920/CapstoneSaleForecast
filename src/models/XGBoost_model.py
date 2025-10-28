from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore


class XGBoostWrapper:
    def __init__(self, params: dict | None = None):
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
            "tree_method": "hist",
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model: XGBRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        self.model = XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return pd.Series(self.model.predict(X), index=X.index)

    def save(self, path: str | Path):
        if self.model is None:
            raise RuntimeError("Model not trained")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON for portability
        self.model.save_model(str(path))

    @staticmethod
    def load(path: str | Path) -> "XGBoostWrapper":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        obj = XGBoostWrapper()
        mdl = XGBRegressor()
        mdl.load_model(str(path))
        obj.model = mdl
        return obj
