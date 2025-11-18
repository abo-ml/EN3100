"""Align OHLCV data with sentiment, macro, and order flow placeholders.

This module consolidates heterogeneous data sources into a single daily panel.
It demonstrates forward-fill alignment logic and documents where future
microstructure and sentiment integrations should occur. The entry-point mirrors
the command shown in the project README/Colab instructions so that running::

    python -m src.data.align_data

in a fresh environment is sufficient after the raw downloads have completed.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .download_data import RAW_DATA_DIR, fetch_orderbook_snapshot
from ..utils import PROCESSED_DIR
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


def fetch_sentiment_scores(price_df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder sentiment feed returning neutral scores aligned to prices."""

    sentiment = price_df[["ticker", "date"]].copy()
    sentiment["sentiment_score"] = 0.0
    return sentiment


def enrich_with_orderflow(price_df: pd.DataFrame) -> pd.DataFrame:
    """Attach placeholder order flow imbalance features on the daily grid."""

    # Pull a single snapshot per ticker to document where the live integration
    # would occur. For now we simply log/keep the placeholder to avoid an unused
    # function warning and return zeros aligned with the daily bars.
    for ticker in price_df["ticker"].unique():
        fetch_orderbook_snapshot(ticker)

    order_features = price_df[["ticker", "date"]].copy()
    order_features["bid_volume"] = 0.0
    order_features["ask_volume"] = 0.0
    order_features["bid_price"] = np.nan
    order_features["ask_price"] = np.nan
    return order_features


def align_sources(price_df: pd.DataFrame) -> pd.DataFrame:
    """Combine price, sentiment, and order flow data on a daily index."""

    sentiment_df = fetch_sentiment_scores(price_df)
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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved processed data to %s", output_path)
    return output_path


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for alignment."""

    parser = argparse.ArgumentParser(description="Align OHLCV, sentiment, and order-flow placeholders")
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional subset of tickers to align. Defaults to all available downloads.",
    )
    parser.add_argument(
        "--output",
        default="combined_features.csv",
        help="Filename (within data/processed) for the aligned dataset.",
    )
    return parser.parse_args(args)


def main(cmd_args: Optional[Iterable[str]] = None) -> Path:
    """Execute the alignment pipeline."""

    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)
    price_df, available = load_price_data(ns.tickers)
    aligned = align_sources(price_df)
    return save_processed(aligned, ns.output)


if __name__ == "__main__":
    main()
