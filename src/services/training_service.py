from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..utils.config import AppConfig
from ..database.queries import fetch_sales_aggregated
from ..utils.preprocessing import prepare_time_series
from ..models.prophet_model import ProphetWrapper
from ..models.lightgbm_model import LightGBMWrapper
from ..utils.features import add_lag_features, add_time_features
from sqlalchemy import text


def _df_from_sql(session, sql: text, params: dict | None = None) -> pd.DataFrame:
    try:
        res = session.execute(sql, params or {})
        rows = res.mappings().all()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _fetch_sales_aggregated_any(session, scope: str) -> pd.DataFrame:
    """Try the standard aggregated fetch; if empty, fall back to MIS_OLL.MIS_PARTY_SURVEY.

    Returns a DataFrame with columns matching the expected schema:
    Region, Area, Territory, Year, Month, SalesAmount (some may be omitted per scope)
    """
    df = fetch_sales_aggregated(session, scope=scope)
    if not df.empty:
        return df

    group_cols_map = {
        "territory": ("[REGION_ID] AS Region, [AREA_ID] AS Area, [TERRITORY_ID] AS Territory", "Region, Area, Territory"),
        "area": ("[REGION_ID] AS Region, [AREA_ID] AS Area", "Region, Area"),
        "region": ("[REGION_ID] AS Region", "Region"),
    }
    select_cols, group_cols_alias = group_cols_map[scope]

    # Attempt 1: Revenue using robust derivation of Year/Month from PSC_DATE when missing
    sql_rev = text(
        f"""
        ;WITH base AS (
          SELECT 
            {select_cols},
            TRY_CAST([YEAR] AS INT) AS Year0,
            TRY_CAST([MONTH] AS INT) AS Month0,
            TRY_CAST([QT_PRS] AS FLOAT) AS QT0,
            TRY_CAST([UNIT_PRICE] AS FLOAT) AS Price0,
            [PSC_DATE]
          FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
        ),
        norm AS (
          SELECT 
            {group_cols_alias.replace('Region','Region').replace('Area','Area').replace('Territory','Territory')},
            COALESCE(Year0, TRY_CAST(DATEPART(YEAR, [PSC_DATE]) AS INT)) AS [Year],
            COALESCE(Month0, TRY_CAST(DATEPART(MONTH, [PSC_DATE]) AS INT)) AS [Month],
            COALESCE(QT0, 0) AS QT,
            COALESCE(Price0, 0) AS Price
          FROM base
        )
        SELECT {group_cols_alias}, [Year], [Month],
               SUM(QT * Price) AS SalesAmount
        FROM norm
        WHERE [Year] IS NOT NULL AND [Month] IS NOT NULL
        GROUP BY {group_cols_alias}, [Year], [Month]
        ORDER BY {group_cols_alias}, [Year], [Month]
        """
    )
    df_fb = _df_from_sql(session, sql_rev)

    if not df_fb.empty:
        return df_fb

    # Attempt 2: Quantity if prices are mostly missing (same robust Year/Month derivation)
    sql_qty = text(
        f"""
        ;WITH base AS (
          SELECT 
            {select_cols},
            TRY_CAST([YEAR] AS INT) AS Year0,
            TRY_CAST([MONTH] AS INT) AS Month0,
            TRY_CAST([QT_PRS] AS FLOAT) AS QT0,
            [PSC_DATE]
          FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
        ),
        norm AS (
          SELECT 
            {group_cols_alias.replace('Region','Region').replace('Area','Area').replace('Territory','Territory')},
            COALESCE(Year0, TRY_CAST(DATEPART(YEAR, [PSC_DATE]) AS INT)) AS [Year],
            COALESCE(Month0, TRY_CAST(DATEPART(MONTH, [PSC_DATE]) AS INT)) AS [Month],
            COALESCE(QT0, 0) AS QT
          FROM base
        )
        SELECT {group_cols_alias}, [Year], [Month],
               SUM(QT) AS SalesAmount
        FROM norm
        WHERE [Year] IS NOT NULL AND [Month] IS NOT NULL
        GROUP BY {group_cols_alias}, [Year], [Month]
        ORDER BY {group_cols_alias}, [Year], [Month]
        """
    )
    df_fb2 = _df_from_sql(session, sql_qty)
    return df_fb2


def train_prophet_pipeline(session, config: AppConfig, scope: str, seasonality_mode: str = "additive") -> dict:
    """Train Prophet models per key group with configurable seasonality_mode.

    Saves one model per key under artifacts/<scope>/prophet/<key>/model.pkl
    """
    df = _fetch_sales_aggregated_any(session, scope=scope)
    if df.empty:
        # Diagnostics to help understand emptiness
        diag = _df_from_sql(
            session,
            text(
                """
                SELECT COUNT(*) AS rows_all,
                       SUM(CASE WHEN [YEAR] IS NOT NULL AND [MONTH] IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_period
                FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
                """
            ),
        )
        info = diag.iloc[0].to_dict() if not diag.empty else {}
        return {
            "status": "no_data",
            "message": "No sales data available from DB",
            "scope": scope,
            "debug": info,
        }

    try:
        ts = prepare_time_series(df, scope=scope)
    except Exception as e:
        return {"status": "error", "message": "prepare_time_series failed", "error": str(e), "columns": list(df.columns)}

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
    df = _fetch_sales_aggregated_any(session, scope=scope)
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
