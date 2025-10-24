from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..utils.config import AppConfig
from ..database.queries import fetch_sales_aggregated
from ..utils.preprocessing import prepare_time_series
from ..models.prophet_model import ProphetWrapper


def _artifact_path(config: AppConfig, scope: str, key_vals) -> Path:
    key_dir = Path(config.artifacts_dir) / scope / "prophet" / "__".join(map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
    return key_dir / "model.pkl"


def forecast_pipeline(session, config: AppConfig, scope: str, horizon: int, filters: dict) -> dict:
    df = fetch_sales_aggregated(session, scope=scope)
    if df.empty:
        return {"status": "no_data", "message": "No sales data available from DB"}

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
