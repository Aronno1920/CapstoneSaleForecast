from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.utils.config import AppConfig
from src.utils.preprocessing import prepare_time_series
from src.services.training_service import _fetch_sales_aggregated_any, _sanitize_year_month
from src.models.prophet_model import ProphetWrapper


_DEF_KEYS = {
    "territory": ["Region", "Area", "Territory"],
    "area": ["Region", "Area"],
    "region": ["Region"],
}


def _artifact_path_prophet(config: AppConfig, scope: str, key_vals) -> Path:
    key_dir = Path(config.artifacts_dir) / scope / "prophet" / "__".join(
        map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,))
    )
    return key_dir / "model.pkl"


def forecast_grid(session, config: AppConfig, scope: str, horizon: int, dimensions: List[str] | None = None) -> Dict[str, Any]:
    try:
        df = _fetch_sales_aggregated_any(session, scope=scope)
        if df.empty:
            return {"status": "no_data", "message": "No sales data available from DB", "scope": scope}

        df = _sanitize_year_month(df)
        if df.empty:
            return {"status": "no_data", "message": "No usable Year/Month after sanitization", "scope": scope}

        ts = prepare_time_series(df, scope=scope)
        keys = _DEF_KEYS[scope]

        # Decide grouping keys: limit to available columns
        dims = list(dimensions or keys)
        available_dims = [d for d in dims if d in ts.columns]
        missing_dims = [d for d in dims if d not in ts.columns]
        if not available_dims:
            available_dims = keys

        results: List[Dict[str, Any]] = []
        for key_vals, grp in ts.groupby(available_dims):
            # Ensure tuple key
            tuple_key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
            path = _artifact_path_prophet(config, scope, tuple_key if len(available_dims) == len(keys) else tuple(
                grp[keys].iloc[0].tolist()
            ))
            if not path.exists():
                # Skip if model for this group isn't trained
                continue
            model = ProphetWrapper.load(path)
            last_ds = grp["ds"].max()
            future = pd.date_range(last_ds, periods=horizon + 1, freq="MS")[1:]
            future_df = pd.DataFrame({"ds": future})
            fcst = model.predict(future_df)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            fcst["Year"] = fcst["ds"].dt.year
            fcst["Month"] = fcst["ds"].dt.month
            record = {
                "key_dims": available_dims,
                "key": tuple_key,
                "missing_dims": missing_dims,
                "forecast": fcst.assign(ds=fcst["ds"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
            }
            results.append(record)

        if not results:
            return {
                "status": "no_models",
                "message": "No matching Prophet models for the requested combinations",
                "scope": scope,
                "dimensions": dims,
                "missing_dims": missing_dims,
            }

        return {
            "status": "ok",
            "scope": scope,
            "dimensions": available_dims,
            "missing_dims": missing_dims,
            "results": results,
        }
    except Exception as e:
        return {"status": "error", "message": "forecast_grid failed", "error": str(e)}
