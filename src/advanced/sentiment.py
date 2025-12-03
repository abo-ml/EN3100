"""Future-work stubs for sentiment analysis integration.

This module now includes a lightweight loader for externally prepared sentiment
scores (e.g. CSV of headline-level labels). It remains intentionally simple so
students can swap in their own data without altering the feature pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def fetch_sentiment_scores(key_terms: Iterable[str]) -> pd.DataFrame:
    """Placeholder fetching logic returning neutral sentiment."""

    # TODO: Integrate with Twitter API, news feeds, or alternative sentiment vendors.
    rows = [{"timestamp": pd.Timestamp.utcnow(), "key_term": term, "sentiment_score": 0.0} for term in key_terms]
    return pd.DataFrame(rows)


def load_external_sentiment(csv_path: Path) -> pd.DataFrame:
    """Load sentiment scores from a CSV prepared outside the pipeline.

    Expected columns
    ----------------
    date: ISO date (YYYY-MM-DD) matching the price data frequency.
    ticker: Ticker symbol aligned to the OHLCV files.
    sentiment_score: Numeric sentiment value per (date, ticker).
    """

    sentiment_df = pd.read_csv(csv_path, parse_dates=["date"])
    sentiment_df["ticker"] = sentiment_df["ticker"].astype(str)
    sentiment_df["sentiment_score"] = pd.to_numeric(sentiment_df["sentiment_score"], errors="coerce")
    return sentiment_df


def sentiment_to_feature(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, date_key: str = "date") -> pd.DataFrame:
    """Align sentiment scores with price data (left join on date and ticker)."""

    temp = sentiment_df.copy()
    temp[date_key] = pd.to_datetime(temp[date_key])
    price_df[date_key] = pd.to_datetime(price_df[date_key])
    merged = price_df.merge(temp[["ticker", date_key, "sentiment_score"]], on=["ticker", date_key], how="left")
    return merged
