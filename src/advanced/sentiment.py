"""Future-work stubs for sentiment analysis integration."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def fetch_sentiment_scores(key_terms: Iterable[str]) -> pd.DataFrame:
    """Placeholder fetching logic returning neutral sentiment."""

    # TODO: Integrate with Twitter API, news feeds, or alternative sentiment vendors.
    rows = [{"timestamp": pd.Timestamp.utcnow(), "key_term": term, "sentiment_score": 0.0} for term in key_terms]
    return pd.DataFrame(rows)


def sentiment_to_feature(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Align sentiment scores with price data."""

    sentiment_df = sentiment_df.copy()
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"])
    sentiment_daily = sentiment_df.groupby(sentiment_df["timestamp"].dt.date)["sentiment_score"].agg(["mean", "std"]).reset_index()
    sentiment_daily.rename(columns={"timestamp": "date", "mean": "sentiment_mean", "std": "sentiment_std"}, inplace=True)
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    merged = price_df.merge(sentiment_daily, on="date", how="left")
    return merged
