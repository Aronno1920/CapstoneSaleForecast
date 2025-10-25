from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.utils.config import AppConfig
from src.utils.preprocessing import prepare_time_series
from src.utils.features import add_lag_features, add_time_features
from src.models.prophet_model import ProphetWrapper
from src.models.lightgbm_model import LightGBMWrapper
from src.services.training_service import _fetch_sales_aggregated_any, _sanitize_year_month


def _keys_for_scope(scope: str) -> List[str]:
    return {
        "territory": ["Region", "Area", "Territory"],
        "area": ["Region", "Area"],
        "region": ["Region"],
    }[scope]


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    diff = (y_true - y_pred).to_numpy(dtype=float)
    return float(np.sqrt(np.nanmean(diff ** 2)))


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    diff = (y_true - y_pred).to_numpy(dtype=float)
    return float(np.nanmean(np.abs(diff)))


def _mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y = y_true.to_numpy(dtype=float)
    p = y_pred.to_numpy(dtype=float)
    mask = y != 0
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(np.abs((y[mask] - p[mask]) / y[mask])) * 100.0)


def _artifact_path_prophet(config: AppConfig, scope: str, key_vals) -> Path:
    key_dir = Path(config.artifacts_dir) / scope / "prophet" / "__".join(
        map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,))
    )
    return key_dir / "model.pkl"


def _artifact_path_lgb(config: AppConfig, scope: str, key_vals) -> Path:
    key_dir = Path(config.artifacts_dir) / scope / "lightgbm" / "__".join(
        map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,))
    )
    return key_dir / "model.txt"


def evaluate_prophet_pipeline(session, config: AppConfig, scope: str) -> Dict[str, Any]:
    try:
        df = _fetch_sales_aggregated_any(session, scope=scope)
        if df.empty:
            return {"status": "no_data", "message": "No sales data available from DB", "scope": scope}

        df = _sanitize_year_month(df)
        if df.empty:
            return {"status": "no_data", "message": "No usable Year/Month after sanitization", "scope": scope}

        ts = prepare_time_series(df, scope=scope)
        keys = _keys_for_scope(scope)

        results = []
        for key_vals, grp in ts.groupby(keys):
            path = _artifact_path_prophet(config, scope, key_vals)
            if not path.exists():
                continue
            # Load model and get in-sample predictions
            try:
                model = ProphetWrapper.load(path)
            except Exception as e:
                results.append({
                    "key": key_vals if isinstance(key_vals, tuple) else (key_vals,),
                    "status": "error", "error": f"load_failed: {e}"
                })
                continue
            g = grp.sort_values("ds").dropna(subset=["y"]).copy()
            if g.empty:
                continue
            y_true = g["y"]
            fcst = model.predict(pd.DataFrame({"ds": g["ds"]}))
            y_pred = fcst.set_index("ds").reindex(g["ds"]).loc[:, "yhat"].reset_index(drop=True)
            y_pred.index = y_true.index

            record = {
                "key": key_vals if isinstance(key_vals, tuple) else (key_vals,),
                "count": int(len(y_true)),
                "rmse": _rmse(y_true, y_pred),
                "mae": _mae(y_true, y_pred),
                "mape": _mape(y_true, y_pred),
            }
            results.append(record)

        if not results:
            return {"status": "no_models", "message": "No prophet models or evaluable groups found", "scope": scope}

        # Aggregate metrics across groups
        df_res = pd.DataFrame([r for r in results if "rmse" in r])
        summary = {
            "groups": int(len(df_res)),
            "rmse": float(df_res["rmse"].mean()) if not df_res.empty else float("nan"),
            "mae": float(df_res["mae"].mean()) if not df_res.empty else float("nan"),
            "mape": float(df_res["mape"].mean()) if not df_res.empty else float("nan"),
        }
        return {"status": "ok", "scope": scope, "summary": summary, "results": results}
    except Exception as e:
        return {"status": "error", "message": "evaluate_prophet failed", "error": str(e)}


def evaluate_lightgbm_pipeline(session, config: AppConfig, scope: str) -> Dict[str, Any]:
    try:
        df = _fetch_sales_aggregated_any(session, scope=scope)
        if df.empty:
            return {"status": "no_data", "message": "No sales data available from DB", "scope": scope}

        df = _sanitize_year_month(df)
        if df.empty:
            return {"status": "no_data", "message": "No usable Year/Month after sanitization", "scope": scope}

        ts = prepare_time_series(df, scope=scope)
        keys = _keys_for_scope(scope)

        results = []
        for key_vals, grp in ts.groupby(keys):
            path = _artifact_path_lgb(config, scope, key_vals)
            if not path.exists():
                continue
            try:
                model = LightGBMWrapper.load(path)
            except Exception as e:
                results.append({
                    "key": key_vals if isinstance(key_vals, tuple) else (key_vals,),
                    "status": "error", "error": f"load_failed: {e}"
                })
                continue

            g = grp.sort_values("ds").copy()
            g = add_lag_features(g, lags=(1, 2, 3, 6, 12))
            g = add_time_features(g)
            g = g.dropna()
            if g.empty:
                continue

            feature_cols = [c for c in g.columns if c not in set(keys + ["ds", "y"])]
            if not feature_cols:
                continue

            X = g[feature_cols]
            y_true = g["y"]
            y_pred = model.predict(X)

            record = {
                "key": key_vals if isinstance(key_vals, tuple) else (key_vals,),
                "count": int(len(y_true)),
                "rmse": _rmse(y_true, y_pred),
                "mae": _mae(y_true, y_pred),
                "mape": _mape(y_true, y_pred),
            }
            results.append(record)

        if not results:
            return {"status": "no_models", "message": "No lightgbm models or evaluable groups found", "scope": scope}

        df_res = pd.DataFrame([r for r in results if "rmse" in r])
        summary = {
            "groups": int(len(df_res)),
            "rmse": float(df_res["rmse"].mean()) if not df_res.empty else float("nan"),
            "mae": float(df_res["mae"].mean()) if not df_res.empty else float("nan"),
            "mape": float(df_res["mape"].mean()) if not df_res.empty else float("nan"),
        }
        return {"status": "ok", "scope": scope, "summary": summary, "results": results}
    except Exception as e:
        return {"status": "error", "message": "evaluate_lightgbm failed", "error": str(e)}
