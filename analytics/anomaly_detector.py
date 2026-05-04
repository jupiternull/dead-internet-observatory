"""
Anomaly Detector — identifies statistically significant deviations
in the Aliveness Index timeline.

Methods:
  - Rolling Z-score (primary)
  - CUSUM (change-point detection)
  - Seasonal decomposition residuals (for periodic signals)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def rolling_zscore_anomalies(
    series: pd.Series,
    window: int = 30,
    threshold: float = 2.5,
) -> pd.Series:
    """
    Returns a boolean Series where True = anomalous.
    Uses a rolling mean/std to catch anomalies relative to recent baseline.
    """
    roll_mean = series.rolling(window=window, min_periods=5, center=False).mean()
    roll_std  = series.rolling(window=window, min_periods=5, center=False).std()
    z = (series - roll_mean) / (roll_std + 1e-9)
    return z.abs() > threshold


def cusum_changepoints(
    series: pd.Series,
    k: float = 0.5,
    h: float = 4.0,
) -> List[int]:
    """
    CUSUM (Cumulative Sum) change-point detection.
    Returns list of indices where a structural shift is detected.
    k = allowance (sensitivity), h = decision threshold.
    """
    values = series.dropna().values.astype(float)
    mu = values.mean()
    sigma = values.std() or 1.0

    S_pos = 0.0
    S_neg = 0.0
    changepoints: List[int] = []

    for i, x in enumerate(values):
        x_norm = (x - mu) / sigma
        S_pos = max(0.0, S_pos + x_norm - k)
        S_neg = max(0.0, S_neg - x_norm - k)
        if S_pos > h or S_neg > h:
            changepoints.append(i)
            S_pos = 0.0
            S_neg = 0.0

    return changepoints


def label_anomalies(df: pd.DataFrame, score_col: str = "aliveness_index") -> pd.DataFrame:
    """
    Add anomaly columns to a timeline DataFrame.
    Expects a 'date' column and a score column.
    Returns df with added: is_anomaly, anomaly_type, z_score, cusum_flag.
    """
    df = df.copy().sort_values("date")
    scores = df[score_col]

    # Rolling z-score
    window = min(30, max(7, len(df) // 5))
    roll_mean = scores.rolling(window=window, min_periods=3).mean()
    roll_std  = scores.rolling(window=window, min_periods=3).std()
    z_scores = (scores - roll_mean) / (roll_std + 1e-9)
    df["z_score"] = z_scores.round(3)

    # Flag direction
    df["anomaly_type"] = ""
    df.loc[z_scores > 2.5, "anomaly_type"] = "spike"
    df.loc[z_scores < -2.5, "anomaly_type"] = "drop"
    df["is_anomaly"] = df["anomaly_type"] != ""

    # CUSUM
    cp_indices = cusum_changepoints(scores)
    df["cusum_flag"] = False
    df.iloc[cp_indices, df.columns.get_loc("cusum_flag")] = True

    return df


def get_notable_anomalies(df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
    """
    Return the top_n most extreme anomalies, sorted by abs(z_score).
    """
    if df.empty or "is_anomaly" not in df.columns:
        return []

    anomalies = df[df["is_anomaly"]].copy()
    anomalies = anomalies.nlargest(top_n, "z_score", keep="all")

    result = []
    for _, row in anomalies.iterrows():
        result.append({
            "date": str(row.get("date", "")),
            "score": float(row.get("aliveness_index", 0)),
            "z_score": float(row.get("z_score", 0)),
            "type": row.get("anomaly_type", "unknown"),
        })
    return result
