import asyncio
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Query, Request
from sqlalchemy import text

router = APIRouter(
    tags=["Product Related - EDA"],
    responses={404: {"description": "Not found"}},
)


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(row._mapping) for row in rows]


##############################################
@router.get("/eda-product/summary")
async def eda_products_summary(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT 
              COUNT(*) AS product_count,
              COUNT(DISTINCT [PHARMA_NAME]) AS pharma_count,
              COUNT(DISTINCT [GENERIC_NAME]) AS generic_count,
              COUNT(DISTINCT [DOSAGE_NAME]) AS dosage_count,
              MIN(CAST([UNIT_PRICE] AS FLOAT)) AS min_price,
              MAX(CAST([UNIT_PRICE] AS FLOAT)) AS max_price,
              AVG(CAST([UNIT_PRICE] AS FLOAT)) AS avg_price
            FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql).all()
            return _rows_to_dicts(rows)[0] if rows else {}

    return await asyncio.to_thread(_run)


@router.get("/eda-product/products")
async def eda_products(
    request: Request,
    pharma_name: Optional[str] = Query(None),
    generic_name: Optional[str] = Query(None),
    dosage_name: Optional[str] = Query(None),
    product_name: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        conditions = []
        params: Dict[str, Any] = {"limit": limit}
        if pharma_name:
            conditions.append("[PHARMA_NAME] = :pharma_name")
            params["pharma_name"] = pharma_name
        if generic_name:
            conditions.append("[GENERIC_NAME] = :generic_name")
            params["generic_name"] = generic_name
        if dosage_name:
            conditions.append("[DOSAGE_NAME] = :dosage_name")
            params["dosage_name"] = dosage_name
        if product_name:
            conditions.append("[PRODUCT_NAME] LIKE :product_name")
            params["product_name"] = f"%{product_name}%"
        if min_price is not None:
            conditions.append("CAST([UNIT_PRICE] AS FLOAT) >= :min_price")
            params["min_price"] = float(min_price)
        if max_price is not None:
            conditions.append("CAST([UNIT_PRICE] AS FLOAT) <= :max_price")
            params["max_price"] = float(max_price)
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT TOP (:limit)
              [PRODUCT_ID], [PRODUCT_CODE], [PRODUCT_NAME],
              [DOSAGE_NAME], [PHARMA_NAME], [GENERIC_NAME],
              CAST([UNIT_PRICE] AS FLOAT) AS unit_price
            FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
            {where_clause}
            ORDER BY [PRODUCT_NAME]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


# 1) Missing value analysis for product master
@router.get("/eda-product/missing-values")
async def product_missing_values(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT 
              SUM(CASE WHEN [PRODUCT_ID] IS NULL THEN 1 ELSE 0 END) AS PRODUCT_ID,
              SUM(CASE WHEN [PRODUCT_CODE] IS NULL THEN 1 ELSE 0 END) AS PRODUCT_CODE,
              SUM(CASE WHEN [PRODUCT_NAME] IS NULL THEN 1 ELSE 0 END) AS PRODUCT_NAME,
              SUM(CASE WHEN [DOSAGE_NAME] IS NULL THEN 1 ELSE 0 END) AS DOSAGE_NAME,
              SUM(CASE WHEN [PHARMA_NAME] IS NULL THEN 1 ELSE 0 END) AS PHARMA_NAME,
              SUM(CASE WHEN [GENERIC_NAME] IS NULL THEN 1 ELSE 0 END) AS GENERIC_NAME,
              SUM(CASE WHEN [UNIT_PRICE] IS NULL THEN 1 ELSE 0 END) AS UNIT_PRICE,
              COUNT(*) AS total_rows
            FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
            """
        )
        with SessionFactory() as session:
            row = session.execute(sql).first()
            if not row:
                return {}
            base = dict(row._mapping)
            total = max(int(base.get("total_rows", 0)), 1)
            return {**base, **{f"{k}_pct": (float(v) / total) * 100.0 for k, v in base.items() if k != "total_rows"}}

    return await asyncio.to_thread(_run)


# 2) PRODUCT_ID uniqueness check
@router.get("/eda-product/uniqueness")
async def product_uniqueness(request: Request, list_limit: int = Query(200, ge=1, le=5000)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        dup_sql = text(
            """
            SELECT COUNT(*) AS total_ids,
                   SUM(CASE WHEN cnt > 1 THEN 1 ELSE 0 END) AS non_unique_ids,
                   SUM(CASE WHEN cnt > 1 THEN cnt ELSE 0 END) AS total_duplicate_rows
            FROM (
              SELECT [PRODUCT_ID], COUNT(*) AS cnt
              FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
              GROUP BY [PRODUCT_ID]
            ) t
            """
        )
        list_sql = text(
            """
            SELECT TOP (:limit) [PRODUCT_ID], COUNT(*) AS cnt
            FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
            GROUP BY [PRODUCT_ID]
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            """
        )
        with SessionFactory() as session:
            summary = dict(session.execute(dup_sql).first()._mapping)
            dups = _rows_to_dicts(session.execute(list_sql, {"limit": list_limit}).all())
        return {"summary": summary, "duplicate_ids": dups}

    return await asyncio.to_thread(_run)


# 3) UNIT_PRICE summary & distribution (quantiles)
@router.get("/eda-product/price-stats")
async def product_price_stats(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT 
              COUNT(*) AS n,
              SUM(CASE WHEN [UNIT_PRICE] IS NULL THEN 1 ELSE 0 END) AS nulls,
              MIN(CAST([UNIT_PRICE] AS FLOAT)) AS min_price,
              MAX(CAST([UNIT_PRICE] AS FLOAT)) AS max_price,
              AVG(CAST([UNIT_PRICE] AS FLOAT)) AS avg_price,
              STDEV(CAST([UNIT_PRICE] AS FLOAT)) AS std_price
            FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
            """
        )
        with SessionFactory() as session:
            base = dict(session.execute(sql).first()._mapping)
            # Quantiles via sample pulled to app (efficient enough for quick EDA)
            df = pd.read_sql(
                text(
                    """
                    SELECT TOP (200000) CAST([UNIT_PRICE] AS FLOAT) AS v
                    FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
                    WHERE [UNIT_PRICE] IS NOT NULL
                    ORDER BY [PRODUCT_ID]
                    """
                ),
                session.bind,
            )
        if df.empty:
            base.update({"q1": None, "median": None, "q3": None})
        else:
            q = df["v"].quantile([0.25, 0.5, 0.75]).to_dict()
            base.update({"q1": float(q.get(0.25)), "median": float(q.get(0.5)), "q3": float(q.get(0.75))})
        return base

    return await asyncio.to_thread(_run)


# 4) Value counts for PHARMA_NAME / GENERIC_NAME / DOSAGE_NAME
@router.get("/eda-product/value-counts")
async def product_value_counts(request: Request, top_n: int = Query(50, ge=1, le=1000)):
    SessionFactory = request.app.state.SessionFactory

    def _counts(col: str, session) -> List[Dict[str, Any]]:
        sql = text(
            f"""
            SELECT TOP (:top_n) [{col}] AS value, COUNT(*) AS cnt
            FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
            GROUP BY [{col}]
            ORDER BY cnt DESC
            """
        )
        return _rows_to_dicts(session.execute(sql, {"top_n": top_n}).all())

    def _distinct_count(col: str, session) -> int:
        sql = text(f"SELECT COUNT(DISTINCT [{col}]) AS c FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]")
        return int(session.execute(sql).first()._mapping["c"])

    def _run():
        with SessionFactory() as session:
            return {
                "PHARMA_NAME": {"top": _counts("PHARMA_NAME", session), "distinct": _distinct_count("PHARMA_NAME", session)},
                "GENERIC_NAME": {"top": _counts("GENERIC_NAME", session), "distinct": _distinct_count("GENERIC_NAME", session)},
                "DOSAGE_NAME": {"top": _counts("DOSAGE_NAME", session), "distinct": _distinct_count("DOSAGE_NAME", session)},
            }

    return await asyncio.to_thread(_run)


# 5) Text length analysis for PRODUCT_NAME
@router.get("/eda-product/name-length-stats")
async def product_name_length_stats(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            df = pd.read_sql(
                text(
                    """
                    SELECT [PRODUCT_ID], [PRODUCT_NAME]
                    FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
                    WHERE [PRODUCT_NAME] IS NOT NULL
                    """
                ),
                session.bind,
            )
        if df.empty:
            return {}
        s = df["PRODUCT_NAME"].astype(str).str.len()
        q = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        return {
            "count": int(s.size),
            "min": int(s.min()),
            "q1": float(q.get(0.25)),
            "median": float(q.get(0.5)),
            "q3": float(q.get(0.75)),
            "max": int(s.max()),
            "avg": float(s.mean()),
        }

    return await asyncio.to_thread(_run)


# 6) Text standardization issues (leading/trailing spaces, case variants)
@router.get("/eda-product/text-quality")
async def product_text_quality(
    request: Request,
    sample: int = Query(20000, ge=100, le=200000),
):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            df = pd.read_sql(
                text(
                    """
                    SELECT TOP (:sample)
                      [PRODUCT_ID], [PRODUCT_CODE], [PRODUCT_NAME], [PHARMA_NAME], [GENERIC_NAME], [DOSAGE_NAME]
                    FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
                    ORDER BY [PRODUCT_ID]
                    """
                ),
                session.bind,
                params={"sample": sample},
            )
        if df.empty:
            return {"sample": 0}
        report: Dict[str, Any] = {"sample": int(df.shape[0])}
        for col in ["PRODUCT_CODE", "PRODUCT_NAME", "PHARMA_NAME", "GENERIC_NAME", "DOSAGE_NAME"]:
            s = df[col].astype(str)
            leading = int((s.str.match(r"^\s+.*")).sum())
            trailing = int((s.str.match(r".*\s+$")).sum())
            multi_space = int((s.str.contains(r"\s{2,}")).sum())
            case_variants = int((s.str.lower().nunique() < s.nunique()))
            report[col] = {
                "leading_spaces": leading,
                "trailing_spaces": trailing,
                "multi_spaces": multi_space,
                "case_variant_issue": bool(case_variants),
            }
        return report

    return await asyncio.to_thread(_run)


# 7) Consistency check: PRODUCT_ID maps consistently to other attributes
@router.get("/eda-product/consistency")
async def product_consistency(request: Request, list_limit: int = Query(200, ge=1, le=5000)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT TOP (:limit) [PRODUCT_ID]
            FROM (
              SELECT [PRODUCT_ID],
                     COUNT(DISTINCT [PRODUCT_CODE]) AS codes,
                     COUNT(DISTINCT [PRODUCT_NAME]) AS names,
                     COUNT(DISTINCT [PHARMA_NAME]) AS pharmas,
                     COUNT(DISTINCT [GENERIC_NAME]) AS generics,
                     COUNT(DISTINCT [DOSAGE_NAME]) AS dosages
              FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
              GROUP BY [PRODUCT_ID]
            ) t
            WHERE codes > 1 OR names > 1 OR pharmas > 1 OR generics > 1 OR dosages > 1
            """
        )
        with SessionFactory() as session:
            bad_ids = [r._mapping["PRODUCT_ID"] for r in session.execute(sql, {"limit": list_limit}).all()]
        return {"inconsistent_product_ids": bad_ids}

    return await asyncio.to_thread(_run)


# 8) Co-occurrence between GENERIC_NAME and DOSAGE_NAME
@router.get("/eda-product/generic-dosage-cooccurrence")
async def generic_dosage_cooccurrence(request: Request, top_n: int = Query(200, ge=1, le=2000)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT TOP (:top_n)
              [GENERIC_NAME], [DOSAGE_NAME], COUNT(*) AS cnt
            FROM [MIS_OLL].[dbo].[MIS_PARTY_PRODUCT]
            GROUP BY [GENERIC_NAME], [DOSAGE_NAME]
            ORDER BY cnt DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"top_n": top_n}).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)
##############################################