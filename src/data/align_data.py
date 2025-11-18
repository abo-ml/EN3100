"""Align OHLCV data with sentiment, macro, and order flow placeholders.

This module consolidates heterogeneous data sources into a single daily panel.
It demonstrates forward-fill alignment logic and documents where future
microstructure and sentiment integrations should occur.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .download_data import RAW_DATA_DIR, fetch_orderbook_snapshot
          
PROCESSED_DIR = Path("data/processed")
LOGGER = logging.getLogger(__name__)


def load_price_data(tickers: Iterable[str]) -> pd.DataFrame:
    """Load previously downloaded OHLCV data from disk."""

    frames = []
    for ticker in tickers:
        path_parquet = RAW_DATA_DIR / f"{ticker.replace('^', '')}.parquet"
        path_csv = RAW_DATA_DIR / f"{ticker.replace('^', '')}.csv"
        if path_parquet.exists():
            frame = pd.read_parquet(path_parquet)
        elif path_csv.exists():
            frame = pd.read_csv(path_csv, parse_dates=["date"])
        else:
            raise FileNotFoundError(f"No raw data found for {ticker}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined.sort_values(["ticker", "date"], inplace=True)
    return combined


def fetch_sentiment_scores(tickers: Iterable[str]) -> pd.DataFrame:
    """Placeholder sentiment feed returning neutral scores."""

    rows = []
    for ticker in tickers:
        rows.append({"ticker": ticker, "timestamp": pd.Timestamp.utcnow(), "sentiment_score": 0.0})
    return pd.DataFrame(rows)


def enrich_with_orderflow(price_df: pd.DataFrame) -> pd.DataFrame:
    """Attach placeholder order flow imbalance features."""

    snapshots = []
    for ticker in price_df["ticker"].unique():
        snapshots.append(fetch_orderbook_snapshot(ticker))
    order_df = pd.concat(snapshots, ignore_index=True)
    order_df["timestamp"] = pd.to_datetime(order_df["timestamp"]).dt.tz_localize(None)
    return order_df


def align_sources(price_df: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Combine price, sentiment, and order flow data on a daily index."""

    sentiment_df = fetch_sentiment_scores(tickers)
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"]).dt.tz_localize(None)
    order_df = enrich_with_orderflow(price_df)

    order_df = order_df.set_index(["ticker", "timestamp"]).sort_index()
    order_daily = (
        order_df.groupby(level=0)
        .apply(lambda grp: grp.resample("1D").ffill())
        .drop(columns=["ticker"])
    )
    order_daily.reset_index(inplace=True)
    order_daily.rename(columns={"timestamp": "date"}, inplace=True)

    sentiment_daily = sentiment_df.copy()
    sentiment_daily = sentiment_daily.set_index(["ticker", "timestamp"]).sort_index()
    sentiment_daily = sentiment_daily.groupby(level=0).apply(lambda grp: grp.resample("1D").ffill())
    sentiment_daily.reset_index(inplace=True)
    sentiment_daily.rename(columns={"timestamp": "date"}, inplace=True)

    merged = (
        price_df.merge(order_daily, on=["ticker", "date"], how="left")
        .merge(sentiment_daily, on=["ticker", "date"], how="left")
    )
    merged.sort_values(["ticker", "date"], inplace=True)
    merged.fillna({"bid_volume": 0.0, "ask_volume": 0.0, "bid_price": np.nan, "ask_price": np.nan, "sentiment_score": 0.0}, inplace=True)
    return merged


def save_processed(df: pd.DataFrame, filename: str = "combined_features.csv") -> Path:
    """Persist the aligned dataframe to disk."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved processed data to %s", output_path)
    return output_path


def main(tickers: Optional[Iterable[str]] = None) -> Path:
    """Execute the alignment pipeline."""

    logging.basicConfig(level="INFO")
    if tickers is None:
        tickers = [p.stem for p in RAW_DATA_DIR.glob("*.parquet")]
        if not tickers:
            tickers = [p.stem for p in RAW_DATA_DIR.glob("*.csv")]
    price_df = load_price_data(tickers)
    aligned = align_sources(price_df, tickers)
    return save_processed(aligned)


if __name__ == "__main__":
    main()
