from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
from src.utils.config import AppConfig
from src.database.queries import fetch_sales_aggregated
from src.utils.preprocessing import prepare_time_series
from src.models.prophet_model import ProphetWrapper
from src.models.lightgbm_model import LightGBMWrapper
from src.utils.features import add_lag_features, add_time_features
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
        "territory": (
            "[REGION_ID] AS Region, [AREA_ID] AS Area, [TERRITORY_ID] AS Territory",
            "[REGION_ID], [AREA_ID], [TERRITORY_ID]",
        ),
        "area": (
            "[REGION_ID] AS Region, [AREA_ID] AS Area",
            "[REGION_ID], [AREA_ID]",
        ),
        "region": (
            "[REGION_ID] AS Region",
            "[REGION_ID]",
        ),
    }
    select_cols, group_cols_expr = group_cols_map[scope]

    # Attempt 1: Revenue using YEAR/MONTH only (no PSC_DATE)
    sql_rev = text(
        f"""
        SELECT {select_cols},
               TRY_CAST([YEAR] AS INT) AS [Year], TRY_CAST([MONTH] AS INT) AS [Month],
               SUM(CAST(COALESCE([QT_PRS],0) AS FLOAT) * CAST(COALESCE([UNIT_PRICE],0) AS FLOAT)) AS SalesAmount
        FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
        WHERE [YEAR] IS NOT NULL AND [MONTH] IS NOT NULL
        GROUP BY {group_cols_expr}, [YEAR], [MONTH]
        ORDER BY {group_cols_expr}, [YEAR], [MONTH]
        """
    )
    df_fb = _df_from_sql(session, sql_rev)

    if not df_fb.empty:
        return df_fb

    # Attempt 2: Quantity if prices are mostly missing (no PSC_DATE)
    sql_qty = text(
        f"""
        SELECT {select_cols},
               TRY_CAST([YEAR] AS INT) AS [Year], TRY_CAST([MONTH] AS INT) AS [Month],
               SUM(CAST(COALESCE([QT_PRS],0) AS FLOAT)) AS SalesAmount
        FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
        WHERE [YEAR] IS NOT NULL AND [MONTH] IS NOT NULL
        GROUP BY {group_cols_expr}, [YEAR], [MONTH]
        ORDER BY {group_cols_expr}, [YEAR], [MONTH]
        """
    )
    df_fb2 = _df_from_sql(session, sql_qty)
    return df_fb2


def _sanitize_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Year/Month are valid integers and months in 1..12. Drop invalid rows.

    Also cast to int to avoid downstream parsing errors.
    """
    if df.empty:
        return df
    out = df.copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Month"] = pd.to_numeric(out["Month"], errors="coerce").astype("Int64")
    out = out[(out["Year"].notna()) & (out["Month"].notna())]
    out = out[(out["Month"] >= 1) & (out["Month"] <= 12)]
    # Optional reasonable year bounds to filter corrupted values
    out = out[(out["Year"] >= 1900) & (out["Year"] <= 2100)]
    # Cast back to int
    out["Year"] = out["Year"].astype(int)
    out["Month"] = out["Month"].astype(int)
    return out


def train_prophet_pipeline(session, config: AppConfig, scope: str, seasonality_mode: str = "additive") -> dict:
    """Train Prophet models per key group with configurable seasonality_mode.

    Saves one model per key under artifacts/<scope>/prophet/<key>/model.pkl
    """
    try:
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

        # Sanitize period columns before building time series
        df = _sanitize_year_month(df)
        if df.empty:
            return {"status": "no_data", "message": "No usable Year/Month after sanitization", "scope": scope}
        try:
            ts = prepare_time_series(df, scope=scope)
        except Exception as e:
            return {"status": "error", "message": "prepare_time_series failed", "error": str(e), "columns": list(df.columns)}

        keys = {
            "territory": ["Region", "Area", "Territory"],
            "area": ["Region", "Area"],
            "region": ["Region"],
        }[scope]

        artifacts: List[str] = []
        for key_vals, grp in ts.groupby(keys):
            g = grp.sort_values("ds").copy()
            g = g.dropna(subset=["y"])  # ensure Prophet receives valid observations
            if len(g) < 2:
                continue  # skip groups with insufficient data

            model = ProphetWrapper(seasonality_mode=seasonality_mode)
            model.fit(g.rename(columns={"ds": "ds", "y": "y"}))
            key_dir = Path(config.artifacts_dir) / scope / "prophet" / "__".join(
                map(str, key_vals if isinstance(key_vals, tuple) else (key_vals,))
            )
            key_dir.mkdir(parents=True, exist_ok=True)
            model_path = key_dir / "model.pkl"
            model.save(model_path)
            artifacts.append(str(model_path))

        return {"status": "ok", "trained": len(artifacts), "artifacts": artifacts}
    except Exception as e:
        return {"status": "error", "message": "train_prophet failed", "error": str(e)}


def train_lightgbm_pipeline(session, config: AppConfig, scope: str, horizon: int = 6) -> dict:
    """Train LightGBM models per key group using lag + calendar features.

    Saves one model per key under artifacts/<scope>/lightgbm/<key>/model.txt
    """
    try:
        df = _fetch_sales_aggregated_any(session, scope=scope)
        if df.empty:
            # Provide diagnostics to help
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
            return {"status": "no_data", "message": "No sales data available from DB", "scope": scope, "debug": info}

        df = _sanitize_year_month(df)
        if df.empty:
            return {"status": "no_data", "message": "No usable Year/Month after sanitization", "scope": scope}

        ts = prepare_time_series(df, scope=scope)

        keys = {
            "territory": ["Region", "Area", "Territory"],
            "area": ["Region", "Area"],
            "region": ["Region"],
        }[scope]

        artifacts: List[str] = []
        for key_vals, grp in ts.groupby(keys):
            g = grp.sort_values("ds").copy()
            g = add_lag_features(g, lags=(1, 2, 3, 6, 12))
            g = add_time_features(g)
            g = g.dropna()
            if g.empty:
                continue
            feature_cols = [c for c in g.columns if c not in set(keys + ["ds", "y"])]
            if not feature_cols:
                # No features available
                continue
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
    except Exception as e:
        return {"status": "error", "message": "train_lightgbm failed", "error": str(e)}
