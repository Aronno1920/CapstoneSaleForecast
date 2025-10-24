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


##############################################
@router.get("/eda-sales/summary")
async def eda_summary(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT
              COUNT(*) AS row_count,
              MIN([YEAR]) AS min_year,
              MAX([YEAR]) AS max_year,
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
            conditions.append("[YEAR] = :year")
            params["year"] = year
        if month is not None:
            conditions.append("[MONTH] = :month")
            params["month"] = month
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT
              [REGION_ID], [AREA_ID], [TERRITORY_ID],
              SUM(CAST([QT_PRS] AS FLOAT)) AS total_qty,
              SUM(CAST([QT_PRS] AS FLOAT) * CAST([UNIT_PRICE] AS FLOAT)) AS sales_amount
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
            conditions.append("[YEAR] = :year")
            params["year"] = year
        if month is not None:
            conditions.append("[MONTH] = :month")
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
              SUM(CAST([QT_PRS] AS FLOAT)) AS total_qty,
              SUM(CAST([QT_PRS] AS FLOAT) * CAST([UNIT_PRICE] AS FLOAT)) AS sales_amount
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
              [ROW_DATA_ID], [SURVEY_CIRCLE_ID], [YEAR], [MONTH], [PRODUCT_ID],
              [QT_PRS], [UNIT_PRICE], [TERRITORY_ID], [AREA_ID], [REGION_ID],
              [TERRITORY_ID], [AREA_ID], [REGION_ID], [PRS_ID], [PSC_DATE]
            FROM [MIS_OLL].[dbo].[MIS_PARTY_SURVEY]
            ORDER BY [ROW_DATA_ID] DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"limit": limit}).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)
##############################################