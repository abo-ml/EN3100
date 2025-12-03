"""Target variable construction for supervised learning models."""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute next-day return and direction targets without lookahead."""

    df = df.sort_values(["ticker", "date"]).copy()
    df["next_day_close"] = df.groupby("ticker")["close"].shift(-1)
    df["next_day_return"] = (df["next_day_close"] - df["close"]) / df["close"]
    df["target_direction"] = (df["next_day_return"] > 0).astype(int)
    targets = df[["ticker", "date", "next_day_return", "target_direction"]].dropna()
    return targets
