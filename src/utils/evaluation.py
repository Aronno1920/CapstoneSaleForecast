from __future__ import annotations
import numpy as np
import pandas as pd


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.replace(0, np.nan)
    return float(np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100.0)
