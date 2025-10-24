from __future__ import annotations
from sqlalchemy import text
import pandas as pd


def fetch_sales_aggregated(session, scope: str = "territory") -> pd.DataFrame:
    # scope in {territory, area, region}
    if scope not in {"territory", "area", "region"}:
        raise ValueError("Invalid scope")

    group_cols = {
        "territory": "Region, Area, Territory",
        "area": "Region, Area",
        "region": "Region",
    }[scope]

    sql = f"""
    SELECT {group_cols}, CAST([Year] as int) as [Year], CAST([Month] as int) as [Month],
           SUM(SalesAmount) as SalesAmount
    FROM SalesHistory
    GROUP BY {group_cols}, [Year], [Month]
    ORDER BY {group_cols}, [Year], [Month]
    """
    try:
        df = pd.read_sql(text(sql), session.bind)
    except Exception as e:
        # Return empty DataFrame to keep API alive without DB connected
        df = pd.DataFrame(columns=["Region", "Area", "Territory", "Year", "Month", "SalesAmount"])
    return df
