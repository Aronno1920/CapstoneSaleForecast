from __future__ import annotations
import pandas as pd


def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    df = df.sort_values("ds").copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    return df
