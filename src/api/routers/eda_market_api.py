import asyncio
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Query, Request
from sqlalchemy import text

router = APIRouter(
    tags=["Market Structure Related - EDA"],
    responses={404: {"description": "Not found"}},
)


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(row._mapping) for row in rows]


##############################################
@router.get("/eda-market/summary")
async def eda_markets_summary(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT 
              COUNT(*) AS node_count,
              COUNT(DISTINCT [MKT_LEVEL]) AS level_count,
              MIN(CAST([MKT_LEVEL] AS INT)) AS min_level,
              MAX(CAST([MKT_LEVEL] AS INT)) AS max_level
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql).all()
            return _rows_to_dicts(rows)[0] if rows else {}

    return await asyncio.to_thread(_run)


@router.get("/eda-market/markets")
async def eda_markets(
    request: Request,
    level: Optional[int] = Query(None, ge=0),
    parent_mkt_code: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        conditions = []
        params: Dict[str, Any] = {"limit": limit}
        if level is not None:
            conditions.append("CAST([MKT_LEVEL] AS INT) = :level")
            params["level"] = int(level)
        if parent_mkt_code:
            conditions.append("[PARENT_MKT_CODE] = :parent_mkt_code")
            params["parent_mkt_code"] = parent_mkt_code
        if search:
            conditions.append("[MKT_NAME] LIKE :search")
            params["search"] = f"%{search}%"
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = text(
            f"""
            SELECT TOP (:limit)
              [FIELD_ID], [MKT_NAME], CAST([MKT_LEVEL] AS INT) AS MKT_LEVEL,
              [PARENT_MKT_CODE], [MKT_CODE_OLD]
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            {where_clause}
            ORDER BY CAST([MKT_LEVEL] AS INT), [MKT_NAME]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, params).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


@router.get("/eda-market/markets-tree")
async def eda_markets_tree(request: Request, max_nodes: int = Query(5000, ge=1, le=20000)):
    """Returns market hierarchy as a tree built from SYS_MARKET_HO using PARENT_MKT_CODE relations."""
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT TOP (:max_nodes)
              [FIELD_ID], [MKT_NAME], CAST([MKT_LEVEL] AS INT) AS MKT_LEVEL,
              [PARENT_MKT_CODE], [MKT_CODE_OLD]
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            ORDER BY CAST([MKT_LEVEL] AS INT), [MKT_NAME]
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"max_nodes": max_nodes}).all()
            nodes = _rows_to_dicts(rows)
        # Build index by code
        by_code: Dict[str, Dict[str, Any]] = {}
        for n in nodes:
            n["children"] = []
            by_code[n.get("MKT_CODE_OLD")] = n
        roots: List[Dict[str, Any]] = []
        for n in nodes:
            parent_code = n.get("PARENT_MKT_CODE")
            if parent_code and parent_code in by_code:
                by_code[parent_code]["children"].append(n)
            else:
                roots.append(n)
        return roots

    return await asyncio.to_thread(_run)


# 1) Level distribution and coverage for MKT_LEVEL
@router.get("/eda-market/level-distribution")
async def market_level_distribution(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT 
              CAST([MKT_LEVEL] AS INT) AS level,
              COUNT(*) AS row_count,
              COUNT(DISTINCT [FIELD_ID]) AS unique_field_ids
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            GROUP BY [MKT_LEVEL]
            ORDER BY level
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


# 2) Hierarchy integrity: children must have a parent that exists
@router.get("/eda-market/hierarchy-integrity")
async def market_hierarchy_integrity(request: Request, sample: int = Query(200, ge=1, le=5000)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            WITH parents AS (
              SELECT DISTINCT [FIELD_ID] AS parent_id
              FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            )
            SELECT TOP (:sample)
              c.[FIELD_ID] AS child_field_id,
              c.[PARENT_MKT_CODE] AS parent_code,
              CASE WHEN p.parent_id IS NULL THEN 1 ELSE 0 END AS missing_parent
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO] c
            LEFT JOIN parents p ON c.[PARENT_MKT_CODE] = p.parent_id
            WHERE c.[PARENT_MKT_CODE] IS NOT NULL AND p.parent_id IS NULL
            ORDER BY c.[FIELD_ID]
            """
        )
        with SessionFactory() as session:
            issues = _rows_to_dicts(session.execute(sql, {"sample": sample}).all())
            total_children = session.execute(
                text(
                    """
                    SELECT COUNT(*) AS n
                    FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
                    WHERE [PARENT_MKT_CODE] IS NOT NULL
                    """
                )
            ).first()._mapping["n"]
        return {"total_children": int(total_children), "missing_parent_cases": issues, "issue_count": len(issues)}

    return await asyncio.to_thread(_run)


# 4) FIELD_ID uniqueness check
@router.get("/eda-market/field-uniqueness")
async def market_field_uniqueness(request: Request, list_limit: int = Query(200, ge=1, le=5000)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        summary_sql = text(
            """
            SELECT COUNT(*) AS total_rows,
                   COUNT(DISTINCT [FIELD_ID]) AS distinct_field_ids
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            """
        )
        dup_list_sql = text(
            """
            SELECT TOP (:limit) [FIELD_ID], COUNT(*) AS cnt
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            GROUP BY [FIELD_ID]
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            """
        )
        with SessionFactory() as session:
            summary = dict(session.execute(summary_sql).first()._mapping)
            dups = _rows_to_dicts(session.execute(dup_list_sql, {"limit": list_limit}).all())
            summary["non_unique_count"] = sum(d["cnt"] for d in dups)
        return {"summary": summary, "duplicate_field_ids": dups}

    return await asyncio.to_thread(_run)

    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            df = pd.read_sql(
                text(
                    """
                    SELECT TOP (:sample) [FIELD_ID], [MKT_NAME]
                    FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
                    ORDER BY [FIELD_ID]
                    """
                ),
                session.bind,
                params={"sample": sample},
            )
        if df.empty:
            return {"sample": 0}
        s = df["MKT_NAME"].astype(str)
        report: Dict[str, Any] = {"sample": int(s.size)}
        report.update(
            {
                "leading_spaces": int((s.str.match(r"^\s+.*")).sum()),
                "trailing_spaces": int((s.str.match(r".*\s+$")).sum()),
                "multi_spaces": int((s.str.contains(r"\s{2,}")).sum()),
                "has_case_variants": bool(s.str.lower().nunique() < s.nunique()),
                "non_alnum_ratio": float(((~s.str.match(r"^[\w\s\-]+$")).sum()) / s.size),
            }
        )
        return report

    return await asyncio.to_thread(_run)


# 6) Missing values (including PARENT_MKT_CODE, MKT_CODE_OLD)
@router.get("/eda-market/missing-values")
async def market_missing_values(request: Request):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT 
              SUM(CASE WHEN [FIELD_ID] IS NULL THEN 1 ELSE 0 END) AS FIELD_ID,
              SUM(CASE WHEN [MKT_NAME] IS NULL THEN 1 ELSE 0 END) AS MKT_NAME,
              SUM(CASE WHEN [MKT_LEVEL] IS NULL THEN 1 ELSE 0 END) AS MKT_LEVEL,
              SUM(CASE WHEN [PARENT_MKT_CODE] IS NULL THEN 1 ELSE 0 END) AS PARENT_MKT_CODE,
              SUM(CASE WHEN [MKT_CODE_OLD] IS NULL THEN 1 ELSE 0 END) AS MKT_CODE_OLD,
              COUNT(*) AS total_rows
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            """
        )
        with SessionFactory() as session:
            row = session.execute(sql).first()
            if not row:
                return {}
            data = dict(row._mapping)
            total = max(int(data.get("total_rows", 0)), 1)
            return {**data, **{f"{k}_pct": (float(v) / total) * 100.0 for k, v in data.items() if k != "total_rows"}}

    return await asyncio.to_thread(_run)


# 9) Feature engineering — breadth: children count per parent
@router.get("/eda-market/children-count")
async def market_children_count(request: Request, top_n: int = Query(500, ge=1, le=10000)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            SELECT TOP (:top_n)
              [PARENT_MKT_CODE] AS parent_id,
              COUNT(*) AS children
            FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
            WHERE [PARENT_MKT_CODE] IS NOT NULL
            GROUP BY [PARENT_MKT_CODE]
            ORDER BY children DESC
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"top_n": top_n}).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)


# 10) Hierarchy path (FIELD_ID to root) for inspection
@router.get("/eda-market/hierarchy-path")
async def market_hierarchy_path(request: Request, field_id: str = Query(...), max_steps: int = Query(10, ge=1, le=50)):
    SessionFactory = request.app.state.SessionFactory

    def _run():
        sql = text(
            """
            ;WITH cte AS (
              SELECT [FIELD_ID], [MKT_NAME], [MKT_LEVEL], [PARENT_MKT_CODE], 0 AS depth
              FROM [MIS_OLL].[dbo].[SYS_MARKET_HO]
              WHERE [FIELD_ID] = :id OR [MKT_CODE_OLD] = :id
              UNION ALL
              SELECT p.[FIELD_ID], p.[MKT_NAME], p.[MKT_LEVEL], p.[PARENT_MKT_CODE], cte.depth + 1
              FROM [MIS_OLL].[dbo].[SYS_MARKET_HO] p
              JOIN cte ON p.[MKT_CODE_OLD] = cte.[PARENT_MKT_CODE]
              WHERE cte.depth < :max_steps
            )
            SELECT * FROM cte
            ORDER BY depth
            """
        )
        with SessionFactory() as session:
            rows = session.execute(sql, {"id": field_id, "max_steps": max_steps}).all()
            return _rows_to_dicts(rows)

    return await asyncio.to_thread(_run)
##############################################
