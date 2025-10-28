import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, Request
from sqlalchemy import text

router = APIRouter(
    tags=["Sales Related - EDA"],
    responses={404: {"description": "Not found"}},
)


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(row._mapping) for row in rows]


#####################################################
@router.get("/eda-sales/summary")
async def eda_summary(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT
              COUNT(*) AS row_count,
              MIN([YEAR_NO]) AS min_year,
              MAX([YEAR_NO]) AS max_year,
              COUNT(DISTINCT [PRODUCT_ID]) AS product_count,
              COUNT(DISTINCT [REGION_ID]) AS region_count,
              COUNT(DISTINCT [AREA_ID]) AS area_count,
              COUNT(DISTINCT [TERRITORY_ID]) AS territory_count
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql).all()
            return _rows_to_dicts(rows)[0] if rows else {}

    return await asyncio.to_thread(_run)


@router.get("/eda-sales/region-summary")
async def eda_region_summary(request: Request, year: Optional[int] = Query(None),month: Optional[int] = Query(None),
):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        conditions = []
        params: Dict[str, Any] = {}
        if year is not None:
            conditions.append("[YEAR_NO] = :year")
            params["year"] = year
        if month is not None:
            conditions.append("[MONTH_NO] = :month")
            params["month"] = month
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT
              [REGION_ID], [AREA_ID], [TERRITORY_ID],
              SUM(CAST([SALES_QTY] AS FLOAT)) AS total_qty,
              SUM(CAST([SALES_QTY] AS FLOAT) * CAST([UNIT_PRICE] AS FLOAT)) AS sales_amount
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            {where_clause}
            GROUP BY [REGION_ID], [AREA_ID], [TERRITORY_ID]
            ORDER BY [REGION_ID], [AREA_ID], [TERRITORY_ID]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


@router.get("/eda-sales/monthly-trend")
async def eda_monthly_trend(
    request: Request,
    region_id: Optional[str] = Query(None),
    area_id: Optional[str] = Query(None),
    territory_id: Optional[str] = Query(None),
    product_id: Optional[str] = Query(None),
):
    SessionFactory = request.app.state.SessionFactory
    def _run():
        conditions = []
        params: Dict[str, Any] = {}
        if region_id:
            conditions.append("[REGION_ID] = :region_id")
            params["region_id"] = region_id
        if area_id:
            conditions.append("[AREA_ID] = :area_id")
            params["area_id"] = area_id
        if territory_id:
            conditions.append("[TERRITORY_ID] = :territory_id")
            params["territory_id"] = territory_id
        if product_id:
            conditions.append("[PRODUCT_ID] = :product_id")
            params["product_id"] = product_id
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT
              CAST([YEAR_NO] AS INT) AS [YEAR_NO],
              CAST([MONTH_NO] AS INT) AS [MONTH_NO],
              SUM(CAST([SALES_QTY] AS FLOAT)) AS total_qty,
              SUM(CAST([SALES_QTY] AS FLOAT) * CAST([UNIT_PRICE] AS FLOAT)) AS sales_amount
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            {where_clause}
            GROUP BY [YEAR_NO], [MONTH_NO]
            ORDER BY [YEAR_NO], [MONTH_NO]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


@router.get("/eda-sales/top-products")
async def eda_top_products(
    request: Request,
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None),
    region_id: Optional[str] = Query(None),
    area_id: Optional[str] = Query(None),
    territory_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=200),
):
    SessionFactory = request.app.state.SessionFactory
    def _run():
        conditions = []
        params: Dict[str, Any] = {"limit": limit}
        if year is not None:
            conditions.append("[YEAR_NO] = :year")
            params["year"] = year
        if month is not None:
            conditions.append("[MONTH_NO] = :month")
            params["month"] = month
        if region_id:
            conditions.append("[REGION_ID] = :region_id")
            params["region_id"] = region_id
        if area_id:
            conditions.append("[AREA_ID] = :area_id")
            params["area_id"] = area_id
        if territory_id:
            conditions.append("[TERRITORY_ID] = :territory_id")
            params["territory_id"] = territory_id
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT TOP (:limit)
              [PRODUCT_ID],
              SUM(CAST([SALES_QTY] AS FLOAT)) AS total_qty,
              SUM(CAST([SALES_QTY] AS FLOAT) * CAST([UNIT_PRICE] AS FLOAT)) AS sales_amount
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            {where_clause}
            GROUP BY [PRODUCT_ID]
            ORDER BY sales_amount DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


@router.get("/eda-sales/sample")
async def eda_sample(request: Request, limit: int = Query(50, ge=1, le=500)):
    SessionFactory = request.app.state.SessionFactory
    def _run():
        sql = text(
            """
            SELECT TOP (:limit)
              [ROW_DATA_ID], [SURVEY_CIRCLE_ID], [YEAR_NO], [MONTH_NO], [PRODUCT_ID],
              [SALES_QTY], [UNIT_PRICE], [TERRITORY_ID], [AREA_ID], [REGION_ID],
              [TERRITORY_ID], [AREA_ID], [REGION_ID], [PRS_ID]
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            ORDER BY [ROW_DATA_ID] DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"limit": limit}).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


# 1) Summary statistics for numeric columns
@router.get("/eda-sales/summary-stats")
async def eda_summary_stats(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT 
              COUNT(*) AS row_count,
              -- SALES_QTY stats
              SUM(CASE WHEN [SALES_QTY] IS NULL THEN 1 ELSE 0 END) AS qt_nulls,
              MIN(CAST([SALES_QTY] AS FLOAT)) AS qt_min,
              MAX(CAST([SALES_QTY] AS FLOAT)) AS qt_max,
              AVG(CAST([SALES_QTY] AS FLOAT)) AS qt_mean,
              STDEV(CAST([SALES_QTY] AS FLOAT)) AS qt_std,
              -- UNIT_PRICE stats
              SUM(CASE WHEN [UNIT_PRICE] IS NULL THEN 1 ELSE 0 END) AS price_nulls,
              MIN(CAST([UNIT_PRICE] AS FLOAT)) AS price_min,
              MAX(CAST([UNIT_PRICE] AS FLOAT)) AS price_max,
              AVG(CAST([UNIT_PRICE] AS FLOAT)) AS price_mean,
              STDEV(CAST([UNIT_PRICE] AS FLOAT)) AS price_std
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            """
        )
        with SessionFactory() as session:
            row = session.execute(sql).first()
            base = dict(row._mapping) if row else {}

        # Additional quality checks (negatives)
        with SessionFactory() as session:
            negs = session.execute(
                text(
                    """
                    SELECT 
                      SUM(CASE WHEN CAST([SALES_QTY] AS FLOAT) < 0 THEN 1 ELSE 0 END) AS qt_neg_count,
                      SUM(CASE WHEN CAST([UNIT_PRICE] AS FLOAT) < 0 THEN 1 ELSE 0 END) AS price_neg_count
                    FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
                    """
                )
            ).first()
            if negs:
                base.update(dict(negs._mapping))
        return base

    return await asyncio.to_thread(_run)


# 10) Feature Engineering Exploration — Time feature extraction from YEAR/MONTH only (PSC_DATE removed)
@router.get("/eda-sales/features/time-extract")
async def eda_time_features(
    request: Request,
    sample: int = Query(5000, ge=100, le=200000),
):
    """Extracts basic time features from YEAR/MONTH: year, month, quarter.

    Note: PSC_DATE-derived fields (weekday, weekend) are removed.
    """
    SessionFactory = request.app.state.SessionFactory
    def _run():
        sql = text(
            """
            SELECT TOP (:sample)
              CAST([YEAR_NO] AS INT) AS [YEAR_NO],
              CAST([MONTH_NO] AS INT) AS [MONTH_NO],
              ((CAST([MONTH_NO] AS INT) + 2) / 3) AS quarter
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            WHERE [YEAR_NO] IS NOT NULL AND [MONTH_NO] IS NOT NULL
            ORDER BY [YEAR_NO] DESC, [MONTH_NO] DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"sample": sample}).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


# 11) Feature Engineering — Month-over-Month UNIT_PRICE change per product
@router.get("/eda-sales/features/price-change-mom")
async def eda_price_change_mom(
    request: Request,
    product_id: Optional[str] = Query(None),
    start_year: Optional[int] = Query(None),
    end_year: Optional[int] = Query(None),
    limit_products: int = Query(200, ge=1, le=5000),
):
    """Computes monthly average UNIT_PRICE and MoM percentage change per PRODUCT_ID.
    Filters by product and year range optional.
    """
    SessionFactory = request.app.state.SessionFactory
    def _run():
        conditions = []
        params: Dict[str, Any] = {"limit_products": limit_products}
        if product_id:
            conditions.append("s.[PRODUCT_ID] = :product_id")
            params["product_id"] = product_id
        if start_year is not None:
            conditions.append("CAST(s.[YEAR_NO] AS INT) >= :start_year")
            params["start_year"] = int(start_year)
        if end_year is not None:
            conditions.append("CAST(s.[YEAR_NO] AS INT) <= :end_year")
            params["end_year"] = int(end_year)
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = text(
            f"""
            WITH monthly AS (
              SELECT 
                s.[PRODUCT_ID], CAST(s.[YEAR_NO] AS INT) AS [YEAR_NO], CAST(s.[MONTH_NO] AS INT) AS [MONTH_NO],
                AVG(CAST(s.[UNIT_PRICE] AS FLOAT)) AS monthly_price
              FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY] s
              {where_clause}
              GROUP BY s.[PRODUCT_ID], s.[YEAR_NO], s.[MONTH_NO]
            ),
            top_prod AS (
              SELECT TOP (:limit_products) [PRODUCT_ID]
              FROM monthly
              GROUP BY [PRODUCT_ID]
              ORDER BY COUNT(*) DESC
            ),
            priced AS (
              SELECT m.*,
                LAG(m.monthly_price) OVER (PARTITION BY m.[PRODUCT_ID] ORDER BY m.[YEAR_NO], m.[MONTH_NO]) AS prev_price
              FROM monthly m
              JOIN top_prod t ON m.[PRODUCT_ID] = t.[PRODUCT_ID]
            )
            SELECT 
              [PRODUCT_ID], [YEAR_NO], [MONTH_NO], monthly_price,
              CASE WHEN prev_price IS NULL OR prev_price = 0 THEN NULL
                   ELSE ((monthly_price - prev_price) / prev_price) * 100.0 END AS pct_change
            FROM priced
            ORDER BY [PRODUCT_ID], [YEAR_NO], [MONTH_NO]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


# 12) Feature Engineering — Rolling average quantity over last N months by region
@router.get("/eda-sales/features/rolling-qty")
async def eda_rolling_qty(
    request: Request,
    window: int = Query(3, ge=2, le=12),
    region_id: Optional[str] = Query(None),
):
    """Computes rolling average of total monthly quantity (SALES_QTY) per REGION_ID using a window over the previous N months including current.
    """
    SessionFactory = request.app.state.SessionFactory
    def _run():
        conditions = []
        params: Dict[str, Any] = {"window": window}
        if region_id:
            conditions.append("[REGION_ID] = :region_id")
            params["region_id"] = region_id
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = text(
            f"""
            WITH monthly AS (
              SELECT 
                CAST([YEAR_NO] AS INT) AS [YEAR_NO], CAST([MONTH_NO] AS INT) AS [MONTH_NO],
                [REGION_ID], SUM(CAST([SALES_QTY] AS FLOAT)) AS total_qty
              FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
              {where_clause}
              GROUP BY [REGION_ID], [YEAR_NO], [MONTH_NO]
            )
            SELECT 
              [REGION_ID], [YEAR_NO], [MONTH_NO], total_qty,
              AVG(total_qty) OVER (
                PARTITION BY [REGION_ID] 
                ORDER BY [YEAR_NO], [MONTH_NO]
                ROWS BETWEEN (:window - 1) PRECEDING AND CURRENT ROW
              ) AS rolling_avg_qty
            FROM monthly
            ORDER BY [REGION_ID], [YEAR_NO], [MONTH_NO]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


# 2) Distribution snapshots (quantiles/IQR) for SALES_QTY and UNIT_PRICE
@router.get("/eda-sales/distribution")
async def eda_distribution(request: Request):
    
    SessionFactory = request.app.state.SessionFactory
    def _quantiles_for(col: str, session) -> Dict[str, Any]:
        # Pull a sample to estimate distribution efficiently
        df = pd.read_sql(
            text(
                f"""
                SELECT TOP (100000)
                  CAST([{col}] AS FLOAT) AS v
                FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
                WHERE [{col}] IS NOT NULL
                ORDER BY [ROW_DATA_ID] DESC
                """
            ),
            session.bind,
        )
        if df.empty:
            return {"count": 0}
        s = df["v"].dropna()
        q = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        iqr = float(q.get(0.75, np.nan) - q.get(0.25, np.nan))
        return {
            "count": int(s.size),
            "min": float(s.min()),
            "q1": float(q.get(0.25, np.nan)),
            "median": float(q.get(0.5, np.nan)),
            "q3": float(q.get(0.75, np.nan)),
            "max": float(s.max()),
            "iqr": iqr,
            "p99": float(s.quantile(0.99)),
        }
    def _run():
        with SessionFactory() as session:
            return {
                "SALES_QTY": _quantiles_for("SALES_QTY", session),
                "UNIT_PRICE": _quantiles_for("UNIT_PRICE", session),
            }

    return await asyncio.to_thread(_run)


# 3) Value counts for categorical IDs
@router.get("/eda-sales/value-counts")
async def eda_value_counts(
    request: Request,
    top_n: int = Query(50, ge=1, le=1000),
):
    SessionFactory = request.app.state.SessionFactory
    def _counts(col: str, session) -> List[Dict[str, Any]]:
        sql = text(
            f"""
            SELECT TOP (:top_n) [{col}] AS value, COUNT(*) AS cnt
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            GROUP BY [{col}]
            ORDER BY cnt DESC
            """
        )
        rows = session.execute(sql, {"top_n": top_n}).all()
        return _rows_to_dicts(rows)

    def _run():
        with SessionFactory() as session:
            return {
                "PRODUCT_ID": _counts("PRODUCT_ID", session),
                "SURVEY_CIRCLE_ID": _counts("SURVEY_CIRCLE_ID", session),
                "TERRITORY_ID": _counts("TERRITORY_ID", session),
                "AREA_ID": _counts("AREA_ID", session),
                "REGION_ID": _counts("REGION_ID", session),
            }

    return await asyncio.to_thread(_run)


# 4) Temporal coverage and seasonality
@router.get("/eda-sales/time-coverage")
async def eda_time_coverage(request: Request):
    SessionFactory = request.app.state.SessionFactory
    def _run():
        with SessionFactory() as session:
            # Counts per year
            by_year = _rows_to_dicts(
                session.execute(
                    text(
                        """
                        SELECT CAST([YEAR_NO] AS INT) AS [YEAR_NO], COUNT(*) AS cnt
                        FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
                        GROUP BY [YEAR_NO]
                        ORDER BY [YEAR_NO]
                        """
                    )
                ).all()
            )
            # Counts per month
            by_month = _rows_to_dicts(
                session.execute(
                    text(
                        """
                        SELECT CAST([MONTH_NO] AS INT) AS [MONTH_NO], COUNT(*) AS cnt
                        FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
                        GROUP BY [MONTH_NO]
                        ORDER BY [MONTH_NO]
                        """
                    )
                ).all()
            )
            return {"by_year": by_year, "by_month": by_month}

    return await asyncio.to_thread(_run)


# 5) Relationship: UNIT_PRICE vs SALES_QTY (correlation and sample)
@router.get("/eda-sales/price-quantity-corr")
async def eda_price_quantity_correlation(
    request: Request,
    sample: int = Query(10000, ge=100, le=200000),
):
    SessionFactory = request.app.state.SessionFactory
    def _run():
        with SessionFactory() as session:
            df = pd.read_sql(
                text(
                    """
                    SELECT TOP (:sample)
                      CAST([UNIT_PRICE] AS FLOAT) AS unit_price,
                      CAST([SALES_QTY] AS FLOAT) AS qty
                    FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
                    WHERE [UNIT_PRICE] IS NOT NULL AND [SALES_QTY] IS NOT NULL
                    ORDER BY [ROW_DATA_ID] DESC
                    """
                ),
                session.bind,
                params={"sample": sample},
            )
        if df.empty:
            return {"sample": 0, "corr": None}
        df = df.dropna()
        corr = float(df["unit_price"].corr(df["qty"])) if df.shape[0] > 1 else None
        return {"sample": int(df.shape[0]), "corr": corr}

    return await asyncio.to_thread(_run)


# 6) REGION_ID vs average UNIT_PRICE
@router.get("/eda-sales/region-avg-price")
async def eda_region_avg_price(request: Request, top_n: int = Query(200, ge=1, le=1000)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT TOP (:top_n)
              [REGION_ID], AVG(CAST([UNIT_PRICE] AS FLOAT)) AS avg_price,
              COUNT(*) AS n
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            GROUP BY [REGION_ID]
            ORDER BY avg_price DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"top_n": top_n}).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)
#####################################################

