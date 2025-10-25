"""Data acquisition utilities for downloading historical OHLCV data.

This module handles interaction with Yahoo Finance via yfinance, performs
basic cleaning, and stores the results under ``data/raw`` for subsequent
processing. Hooks are included for integrating higher-frequency order book data
from broker APIs. The functions are designed for reproducibility in the
accompanying dissertation project.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

RAW_DATA_DIR = Path("data/raw")
LOGGER = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    """Configuration for data downloads.

    Attributes
    ----------
    tickers:
        Iterable of ticker symbols recognised by Yahoo Finance (e.g. "AAPL",
        "EURUSD=X").
    start:
        Start date (inclusive) in YYYY-MM-DD format.
    end:
        End date (inclusive) in YYYY-MM-DD format.
    interval:
        Sampling interval for yfinance (default daily).
    max_retries:
        Number of retries when network requests fail.
    retry_backoff:
        Seconds to wait between retries (exponential backoff multiplier).
    format:
        Output format: ``"parquet"`` (default) or ``"csv"``.
    """

    tickers: Iterable[str]
    start: str
    end: str
    interval: str = "1d"
    max_retries: int = 3
    retry_backoff: float = 2.0
    format: str = "parquet"


def ensure_directory(path: Path) -> None:
    """Ensure that the parent directory exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def download_single_ticker(ticker: str, config: DownloadConfig) -> pd.DataFrame:
    """Download OHLCV data for a single ticker with retry logic."""

    attempt = 0
    while True:
        try:
            LOGGER.info("Downloading %s", ticker)
            data = yf.download(
                ticker,
                start=config.start,
                end=config.end,
                interval=config.interval,
                progress=False,
            )
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            data = data.rename(columns=str.lower)
            data.index = data.index.tz_localize(None)
            data["ticker"] = ticker
            data.reset_index(inplace=True)
            data.rename(columns={"index": "date"}, inplace=True)
            data.sort_values("date", inplace=True)
            data = data.ffill().dropna(subset=["close"])  # forward fill gaps
            return data
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > config.max_retries:
                LOGGER.exception("Failed to download %s after %d attempts", ticker, attempt)
                raise exc
            sleep_time = config.retry_backoff ** attempt
            LOGGER.warning("Retry %s/%s for %s after error: %s", attempt, config.max_retries, ticker, exc)
            time.sleep(sleep_time)


def download_ohlcv(config: DownloadConfig) -> pd.DataFrame:
    """Download OHLCV data for a list of tickers and persist to disk.

    Returns a concatenated dataframe across all tickers for convenience.
    """

    frames: List[pd.DataFrame] = []
    for ticker in config.tickers:
        frame = download_single_ticker(ticker, config)
        frames.append(frame)
        output_path = RAW_DATA_DIR / f"{ticker.replace('^', '')}.{config.format}"
        ensure_directory(output_path)
        if config.format == "csv":
            frame.to_csv(output_path, index=False)
        else:
            frame.to_parquet(output_path, index=False)
        LOGGER.info("Saved %s records for %s to %s", len(frame), ticker, output_path)
    combined = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    return combined


def fetch_orderbook_snapshot(ticker: str) -> pd.DataFrame:
    """Placeholder for integrating Level 2 order book data.

    TODO: Replace this stub with calls to broker APIs such as Interactive Brokers
    or Alpaca. The function should return granular bid/ask depth information that
    can be aggregated into order-flow imbalance signals.
    """

    now = pd.Timestamp.utcnow().floor("min")
    dummy = pd.DataFrame(
        {
            "timestamp": [now],
            "ticker": [ticker],
            "bid_volume": [np.nan],
            "ask_volume": [np.nan],
            "bid_price": [np.nan],
            "ask_price": [np.nan],
        }
    )
    return dummy


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Download OHLCV data via yfinance")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers to download")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Sampling interval (default 1d)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff multiplier for retries")
    return parser.parse_args(args)


def main(cmd_args: Optional[List[str]] = None) -> None:
    """Entry point for CLI execution."""

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    ns = parse_args(cmd_args)
    config = DownloadConfig(
        tickers=ns.tickers,
        start=ns.start,
        end=ns.end,
        interval=ns.interval,
        max_retries=ns.max_retries,
        retry_backoff=ns.retry_backoff,
        format=ns.format,
    )
    download_ohlcv(config)


if __name__ == "__main__":
    main()
