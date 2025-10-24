from __future__ import annotations
import pandas as pd


def prepare_time_series(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    # Expect columns including Region, Area, Territory, Year, Month, SalesAmount
    df = df.copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["ds"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1).rename(columns={"DAY": "day"}))
    df = df.rename(columns={"SalesAmount": "y"})
    keys = {
        "territory": ["Region", "Area", "Territory"],
        "area": ["Region", "Area"],
        "region": ["Region"],
    }[scope]
    return df[keys + ["ds", "y"]]
