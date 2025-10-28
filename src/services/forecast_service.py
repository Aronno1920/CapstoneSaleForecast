from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils.config import AppConfig
from src.utils.preprocessing import prepare_time_series
from src.utils.features import add_lag_features, add_time_features
from src.models.prophet_model import ProphetWrapper
from src.models.XGBoost_model import XGBoostWrapper
from src.models.lstm_model import LSTMWrapper
from src.services.training_service import _fetch_sales_aggregated_any, _sanitize_year_month


def _artifact_path(config: AppConfig, scope: str, key_vals) -> Path:
    key_dir = Path(config.artifacts_dir) / scope / "prophet" / "__".join(map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
    return key_dir / "model.pkl"


def forecast_pipeline(session, config: AppConfig, scope: str, horizon: int, filters: dict) -> dict:
    df = _fetch_sales_aggregated_any(session, scope=scope)
    if df.empty:
        return {"status": "no_data", "message": "No sales data available from DB"}

    df = _sanitize_year_month(df)
    if df.empty:
        return {"status": "no_data", "message": "No usable Year/Month after sanitization", "scope": scope}

    ts = prepare_time_series(df, scope=scope)

    keys = {
        "territory": ["Region", "Area", "Territory"],
        "area": ["Region", "Area"],
        "region": ["Region"],
    }[scope]

    # Apply filters if any
    for k, v in (filters or {}).items():
        if k in ts.columns:
            ts = ts[ts[k] == v]

    results = []
    for key_vals, grp in ts.groupby(keys):
        path = _artifact_path(config, scope, key_vals)
        if not path.exists():
            continue
        model = ProphetWrapper.load(path)
        last_ds = grp["ds"].max()
        future = pd.date_range(last_ds, periods=horizon + 1, freq="MS")[1:]
        future_df = pd.DataFrame({"ds": future})
        fcst = model.predict(future_df)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        record = {
            "key": key_vals if isinstance(key_vals, tuple) else (key_vals,),
            "forecast": fcst.to_dict(orient="records"),
        }
        results.append(record)

    return {"status": "ok", "results": results}


def _lstm_artifact_path(config: AppConfig, scope: str, key_vals) -> Path:
    key_dir = Path(config.artifacts_dir) / scope / "lstm" / "__".join(
        map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,))
    )
    return key_dir / "model.pt"


def forecast_lstm_pipeline(session, config: AppConfig, scope: str, horizon: int, filters: dict, window: int = 12) -> dict:
    df = _fetch_sales_aggregated_any(session, scope=scope)
    if df.empty:
        return {"status": "no_data", "message": "No sales data available from DB"}

    df = _sanitize_year_month(df)
    if df.empty:
        return {"status": "no_data", "message": "No usable Year/Month after sanitization", "scope": scope}

    ts = prepare_time_series(df, scope=scope)

    keys = {
        "territory": ["Region", "Area", "Territory"],
        "area": ["Region", "Area"],
        "region": ["Region"],
    }[scope]

    for k, v in (filters or {}).items():
        if k in ts.columns:
            ts = ts[ts[k] == v]

    results = []
    for key_vals, grp in ts.groupby(keys):
        path = _lstm_artifact_path(config, scope, key_vals)
        if not path.exists():
            continue
        try:
            model = LSTMWrapper.load(path)
            win = int(model.window or window)
        except Exception:
            continue

        g = grp.sort_values("ds").dropna(subset=["y"]).copy()
        if len(g) < win:
            continue
        last_ds = g["ds"].max()
        future_dates = pd.date_range(last_ds, periods=horizon + 1, freq="MS")[1:]

        # Prepare rolling series of y
        y_hist = g["y"].astype(float).to_list()
        forecasts = []
        for d in future_dates:
            window_vals = y_hist[-win:]
            if len(window_vals) < win:
                forecasts = []
                break
            yhat = model.predict_next(window_vals)
            forecasts.append({"ds": d, "yhat": yhat, "yhat_lower": yhat, "yhat_upper": yhat})
            y_hist.append(yhat)

        if not forecasts:
            continue

        record = {
            "key": key_vals if isinstance(key_vals, tuple) else (key_vals,),
            "forecast": forecasts,
        }
        results.append(record)

    return {"status": "ok", "results": results}


def _xgb_artifact_path(config: AppConfig, scope: str, key_vals) -> Path:
    key_dir = Path(config.artifacts_dir) / scope / "xgboost" / "__".join(
        map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,))
    )
    return key_dir / "model.json"


def forecast_xgboost_pipeline(session, config: AppConfig, scope: str, horizon: int, filters: dict) -> dict:
    df = _fetch_sales_aggregated_any(session, scope=scope)
    if df.empty:
        return {"status": "no_data", "message": "No sales data available from DB"}

    df = _sanitize_year_month(df)
    if df.empty:
        return {"status": "no_data", "message": "No usable Year/Month after sanitization", "scope": scope}

    ts = prepare_time_series(df, scope=scope)

    keys = {
        "territory": ["Region", "Area", "Territory"],
        "area": ["Region", "Area"],
        "region": ["Region"],
    }[scope]

    # Apply filters if any
    for k, v in (filters or {}).items():
        if k in ts.columns:
            ts = ts[ts[k] == v]

    results = []
    for key_vals, grp in ts.groupby(keys):
        path = _xgb_artifact_path(config, scope, key_vals)
        if not path.exists():
            continue
        try:
            model = XGBoostWrapper.load(path)
        except Exception:
            continue

        g = grp.sort_values("ds").copy()
        # Build initial features to ensure we have enough history for lags
        g_feat = add_lag_features(g, lags=(1, 2, 3, 6, 12))
        g_feat = add_time_features(g_feat)
        feature_cols = [c for c in g_feat.columns if c not in set(keys + ["ds", "y"])]
        # If we can't compute features (insufficient history), skip
        if g_feat[feature_cols].dropna().empty:
            continue

        # Rolling forecast
        last_ds = g["ds"].max()
        future_dates = pd.date_range(last_ds, periods=horizon + 1, freq="MS")[1:]
        forecasts = []
        g_roll = g.copy()
        for d in future_dates:
            tmp = g_roll.copy()
            tmp = add_lag_features(tmp, lags=(1, 2, 3, 6, 12))
            tmp = add_time_features(tmp)
            tmp_last = tmp.iloc[[-1]].copy()
            # Build feature row for next date by updating ds to next period
            feat_row = tmp_last.copy()
            feat_row["ds"] = d
            feat_row = add_time_features(feat_row)
            # Recompute lags relative to current g_roll (they are already in tmp_last)
            X = feat_row[feature_cols]
            if X.isna().any().any():
                # Not enough history for some lags; skip this key
                forecasts = []
                break
            yhat = float(model.predict(X)[0])
            forecasts.append({"ds": d, "yhat": yhat, "yhat_lower": yhat, "yhat_upper": yhat})
            # Append predicted point to history for next step
            g_roll = pd.concat([g_roll, pd.DataFrame({"ds": [d], "y": [yhat]})], ignore_index=True)

        if not forecasts:
            continue

        record = {
            "key": key_vals if isinstance(key_vals, tuple) else (key_vals,),
            "forecast": forecasts,
        }
        results.append(record)

    return {"status": "ok", "results": results}
