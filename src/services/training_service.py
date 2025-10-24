from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..utils.config import AppConfig
from ..database.queries import fetch_sales_aggregated
from ..utils.preprocessing import prepare_time_series
from ..models.prophet_model import ProphetWrapper
from ..models.lightgbm_model import LightGBMWrapper
from ..utils.features import add_lag_features, add_time_features


def train_pipeline(session, config: AppConfig, scope: str, horizon: int = 6) -> dict:
    df = fetch_sales_aggregated(session, scope=scope)
    if df.empty:
        return {"status": "no_data", "message": "No sales data available from DB"}

    ts = prepare_time_series(df, scope=scope)

    # Train per key group with Prophet, placeholder implementation
    keys = {
        "territory": ["Region", "Area", "Territory"],
        "area": ["Region", "Area"],
        "region": ["Region"],
    }[scope]

    artifacts = []
    for key_vals, grp in ts.groupby(keys):
        model = ProphetWrapper(seasonality_mode="additive")
        model.fit(grp.rename(columns={"ds": "ds", "y": "y"}))
        key_dir = Path(config.artifacts_dir) / scope / "prophet" / "__".join(map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        key_dir.mkdir(parents=True, exist_ok=True)
        model_path = key_dir / "model.pkl"
        model.save(model_path)
        artifacts.append(str(model_path))

    # Placeholder LightGBM training on engineered features could be added here

    return {"status": "ok", "trained": len(artifacts), "artifacts": artifacts}


def train_prophet_pipeline(session, config: AppConfig, scope: str, seasonality_mode: str = "additive") -> dict:
    """Train Prophet models per key group with configurable seasonality_mode.

    Saves one model per key under artifacts/<scope>/prophet/<key>/model.pkl
    """
    df = fetch_sales_aggregated(session, scope=scope)
    if df.empty:
        return {"status": "no_data", "message": "No sales data available from DB"}

    ts = prepare_time_series(df, scope=scope)

    keys = {
        "territory": ["Region", "Area", "Territory"],
        "area": ["Region", "Area"],
        "region": ["Region"],
    }[scope]

    artifacts: list[str] = []
    for key_vals, grp in ts.groupby(keys):
        model = ProphetWrapper(seasonality_mode=seasonality_mode)
        model.fit(grp.rename(columns={"ds": "ds", "y": "y"}))
        key_dir = Path(config.artifacts_dir) / scope / "prophet" / "__".join(
            map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,))
        )
        key_dir.mkdir(parents=True, exist_ok=True)
        model_path = key_dir / "model.pkl"
        model.save(model_path)
        artifacts.append(str(model_path))

    return {"status": "ok", "trained": len(artifacts), "artifacts": artifacts}


def train_lightgbm_pipeline(session, config: AppConfig, scope: str, horizon: int = 6) -> dict:
    """Train LightGBM models per key group using lag + calendar features.

    Saves one model per key under artifacts/<scope>/lightgbm/<key>/model.txt
    """
    df = fetch_sales_aggregated(session, scope=scope)
    if df.empty:
        return {"status": "no_data", "message": "No sales data available from DB"}

    ts = prepare_time_series(df, scope=scope)

    keys = {
        "territory": ["Region", "Area", "Territory"],
        "area": ["Region", "Area"],
        "region": ["Region"],
    }[scope]

    artifacts: list[str] = []
    for key_vals, grp in ts.groupby(keys):
        g = grp.sort_values("ds").copy()
        g = add_lag_features(g, lags=(1, 2, 3, 6, 12))
        g = add_time_features(g)
        g = g.dropna()
        if g.empty:
            continue
        feature_cols = [c for c in g.columns if c not in set(keys + ["ds", "y"])]
        X = g[feature_cols]
        y = g["y"]

        model = LightGBMWrapper()
        model.fit(X, y)

        key_dir = Path(config.artifacts_dir) / scope / "lightgbm" / "__".join(
            map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,))
        )
        key_dir.mkdir(parents=True, exist_ok=True)
        model_path = key_dir / "model.txt"
        model.save(model_path)
        artifacts.append(str(model_path))

    return {"status": "ok", "trained": len(artifacts), "artifacts": artifacts}
