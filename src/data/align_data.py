"""Align OHLCV data with placeholder sentiment and order-flow features.

This module consolidates heterogeneous data sources into a single daily panel.
The entry-point mirrors the commands documented for Colab usage so that running

    python -m src.data.align_data

is sufficient after the raw downloads have completed.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..advanced.sentiment import load_external_sentiment
from ..utils import PROCESSED_DIR, RAW_DATA_DIR, ensure_directories
from .download_data import fetch_orderbook_snapshot

LOGGER = logging.getLogger(__name__)


def _candidate_paths(ticker: str) -> List[Path]:
    """Return possible filenames for a ticker in the raw directory."""

    normalised = ticker.replace("^", "")
    return [RAW_DATA_DIR / f"{normalised}.parquet", RAW_DATA_DIR / f"{normalised}.csv"]


def load_price_data(tickers: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Load previously downloaded OHLCV data from disk.

    Parameters
    ----------
    tickers:
        Optional iterable of tickers. When ``None`` all available raw files are
        loaded. When specified, both the raw ticker (e.g. ``"^GSPC"``) and the
        sanitised filename stem (``"GSPC"``) are accepted.
    """

    frames: List[pd.DataFrame] = []

    if tickers:
        requested = list(tickers)
        for ticker in requested:
            for path in _candidate_paths(ticker):
                if not path.exists():
                    continue
                if path.suffix == ".parquet":
                    frame = pd.read_parquet(path)
                else:
                    frame = pd.read_csv(path, parse_dates=["date"])
                frame["ticker"] = frame.get("ticker", ticker)
                frames.append(frame)
                break
            else:
                raise FileNotFoundError(f"No raw data found for {ticker}. Run the download step first.")
    else:
        parquet_files = sorted(RAW_DATA_DIR.glob("*.parquet"))
        csv_files = sorted(RAW_DATA_DIR.glob("*.csv")) if not parquet_files else []
        files = parquet_files or csv_files
        if not files:
            raise FileNotFoundError("No raw OHLCV files detected. Run src.data.download_data first.")
        for path in files:
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_csv(path, parse_dates=["date"])
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined.sort_values(["ticker", "date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    available_tickers = sorted(combined["ticker"].dropna().unique().tolist())
    return combined, available_tickers


def fetch_sentiment_scores(price_df: pd.DataFrame, sentiment_path: Optional[Path] = None) -> pd.DataFrame:
    """Return sentiment aligned to the price data, loading from CSV when provided."""

    if sentiment_path and sentiment_path.exists():
        sentiment = load_external_sentiment(sentiment_path)
        sentiment.rename(columns={"date": "date"}, inplace=True)
        sentiment["date"] = pd.to_datetime(sentiment["date"])
        return sentiment[["ticker", "date", "sentiment_score"]]

    sentiment = price_df[["ticker", "date"]].copy()
    sentiment["sentiment_score"] = 0.0
    return sentiment


def enrich_with_orderflow(price_df: pd.DataFrame) -> pd.DataFrame:
    """Attach placeholder order flow imbalance features on the daily grid."""

    for ticker in price_df["ticker"].unique():
        fetch_orderbook_snapshot(ticker)

    order_features = price_df[["ticker", "date"]].copy()
    order_features["bid_volume"] = 0.0
    order_features["ask_volume"] = 0.0
    order_features["bid_price"] = np.nan
    order_features["ask_price"] = np.nan
    return order_features


def align_sources(price_df: pd.DataFrame, sentiment_path: Optional[Path] = None) -> pd.DataFrame:
    """Combine price, sentiment, and order flow data on a daily index."""

    sentiment_df = fetch_sentiment_scores(price_df, sentiment_path)
    order_df = enrich_with_orderflow(price_df)

    merged = (
        price_df.merge(order_df, on=["ticker", "date"], how="left").merge(
            sentiment_df, on=["ticker", "date"], how="left"
        )
    )
    merged.sort_values(["ticker", "date"], inplace=True)
    merged.fillna(
        {
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "bid_price": np.nan,
            "ask_price": np.nan,
            "sentiment_score": 0.0,
        },
        inplace=True,
    )
    return merged


def save_processed(df: pd.DataFrame, filename: str = "combined_features.csv") -> Path:
    """Persist the aligned dataframe to disk."""

    ensure_directories(PROCESSED_DIR)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved processed data to %s", output_path)
    return output_path


def _read_ticker_file(path: Path) -> List[str]:
    tickers = []
    for line in path.read_text().splitlines():
        ticker = line.strip()
        if ticker:
            tickers.append(ticker)
    return tickers


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for alignment."""

    parser = argparse.ArgumentParser(description="Align OHLCV, sentiment, and order-flow placeholders")
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional subset of tickers to align. Defaults to all available downloads.",
    )
    parser.add_argument(
        "--ticker-file",
        type=Path,
        default=None,
        help="Optional file containing one ticker per line to align.",
    )
    parser.add_argument(
        "--output",
        default="combined_features.csv",
        help="Filename (within data/processed) for the aligned dataset.",
    )
    parser.add_argument(
        "--sentiment-csv",
        type=Path,
        default=None,
        help="Optional path to a CSV containing columns [date, ticker, sentiment_score]",
    )
    return parser.parse_args(args)


def main(cmd_args: Optional[Iterable[str]] = None) -> Path:
    """Execute the alignment pipeline."""

    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)

    selected_tickers: List[str] = []
    if ns.tickers:
        selected_tickers.extend(ns.tickers)
    if ns.ticker_file:
        if not ns.ticker_file.exists():
            raise FileNotFoundError(f"Ticker file not found: {ns.ticker_file}")
        selected_tickers.extend(_read_ticker_file(ns.ticker_file))

    price_df, available = load_price_data(selected_tickers or None)
    LOGGER.info("Aligning %d records across tickers: %s", len(price_df), ", ".join(sorted(set(price_df["ticker"]))))
    aligned = align_sources(price_df, ns.sentiment_csv)
    LOGGER.info("Available tickers: %s", ", ".join(available))
    return save_processed(aligned, ns.output)


if __name__ == "__main__":
    main()
