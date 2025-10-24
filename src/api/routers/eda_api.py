import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request
from sqlalchemy import text

router = APIRouter(
    tags=["eda"],
    responses={404: {"description": "Not found"}},
)


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(row._mapping) for row in rows]


@router.get("/eda/health")
async def eda_health():
    return {"status": "ok"}


@router.get("/eda/summary")
async def eda_summary(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT
              COUNT(*) AS row_count,
              MIN([YEAR]) AS min_year,
              MAX([YEAR]) AS max_year,
              COUNT(DISTINCT [PRODUCT_CODE]) AS product_count,
              COUNT(DISTINCT [REGION_CODE]) AS region_count,
              COUNT(DISTINCT [AREA_CODE]) AS area_count,
              COUNT(DISTINCT [TERRITORY_CODE]) AS territory_count
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql).all()
            return _rows_to_dicts(rows)[0] if rows else {}

    return await asyncio.to_thread(_run)


@router.get("/eda/region-summary")
async def eda_region_summary(
    request: Request,
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None),
):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        conditions = []
        params: Dict[str, Any] = {}
        if year is not None:
            conditions.append("[YEAR] = :year")
            params["year"] = year
        if month is not None:
            conditions.append("[MONTH] = :month")
            params["month"] = month
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT
              [REGION_CODE], [AREA_CODE], [TERRITORY_CODE],
              SUM(CAST([QT_PRS] AS FLOAT)) AS total_qty,
              SUM(CAST([QT_PRS] AS FLOAT) * CAST([UNIT_PRICE] AS FLOAT)) AS sales_amount
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            {where_clause}
            GROUP BY [REGION_CODE], [AREA_CODE], [TERRITORY_CODE]
            ORDER BY [REGION_CODE], [AREA_CODE], [TERRITORY_CODE]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


@router.get("/eda/monthly-trend")
async def eda_monthly_trend(
    request: Request,
    region_code: Optional[str] = Query(None),
    area_code: Optional[str] = Query(None),
    territory_code: Optional[str] = Query(None),
    product_code: Optional[str] = Query(None),
):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        conditions = []
        params: Dict[str, Any] = {}
        if region_code:
            conditions.append("[REGION_CODE] = :region_code")
            params["region_code"] = region_code
        if area_code:
            conditions.append("[AREA_CODE] = :area_code")
            params["area_code"] = area_code
        if territory_code:
            conditions.append("[TERRITORY_CODE] = :territory_code")
            params["territory_code"] = territory_code
        if product_code:
            conditions.append("[PRODUCT_CODE] = :product_code")
            params["product_code"] = product_code
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT
              CAST([YEAR] AS INT) AS [year],
              CAST([MONTH] AS INT) AS [month],
              SUM(CAST([QT_PRS] AS FLOAT)) AS total_qty,
              SUM(CAST([QT_PRS] AS FLOAT) * CAST([UNIT_PRICE] AS FLOAT)) AS sales_amount
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            {where_clause}
            GROUP BY [YEAR], [MONTH]
            ORDER BY [YEAR], [MONTH]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


@router.get("/eda/top-products")
async def eda_top_products(
    request: Request,
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None),
    region_code: Optional[str] = Query(None),
    area_code: Optional[str] = Query(None),
    territory_code: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=200),
):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        conditions = []
        params: Dict[str, Any] = {"limit": limit}
        if year is not None:
            conditions.append("[YEAR] = :year")
            params["year"] = year
        if month is not None:
            conditions.append("[MONTH] = :month")
            params["month"] = month
        if region_code:
            conditions.append("[REGION_CODE] = :region_code")
            params["region_code"] = region_code
        if area_code:
            conditions.append("[AREA_CODE] = :area_code")
            params["area_code"] = area_code
        if territory_code:
            conditions.append("[TERRITORY_CODE] = :territory_code")
            params["territory_code"] = territory_code
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT TOP (:limit)
              [PRODUCT_CODE],
              SUM(CAST([QT_PRS] AS FLOAT)) AS total_qty,
              SUM(CAST([QT_PRS] AS FLOAT) * CAST([UNIT_PRICE] AS FLOAT)) AS sales_amount
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            {where_clause}
            GROUP BY [PRODUCT_CODE]
            ORDER BY sales_amount DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


@router.get("/eda/sample")
async def eda_sample(request: Request, limit: int = Query(50, ge=1, le=500)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT TOP (:limit)
              [ROW_DATA_ID], [SURVEY_CIRCLE_ID], [YEAR], [MONTH], [PRODUCT_CODE], [PRODUCT_ID],
              [QT_PRS], [UNIT_PRICE], [TERRITORY_CODE], [AREA_CODE], [REGION_CODE],
              [TERRITORY_ID], [AREA_ID], [REGION_ID], [PRS_ID], [PSC_DATE]
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            ORDER BY [ROW_DATA_ID] DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"limit": limit}).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)
