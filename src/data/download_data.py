"""Data acquisition utilities for downloading historical OHLCV data.

This module handles interaction with Alpha Vantage (primary) and Yahoo Finance
as a fallback, performs basic cleaning, and stores the results under
``data/raw`` for subsequent processing. Hooks are included for integrating
higher-frequency order book data from broker APIs. The functions are designed
for reproducibility in the accompanying dissertation project.
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

try:  # pragma: no cover - optional dependency for Alpha Vantage usage
    from alpha_vantage.foreignexchange import ForeignExchange
    from alpha_vantage.timeseries import TimeSeries
except ImportError:  # pragma: no cover
    ForeignExchange = None
    TimeSeries = None

from ..utils import RAW_DATA_DIR
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
        Sampling interval (currently ``1d`` supported for Alpha Vantage).
        Sampling interval for yfinance (default daily).
    max_retries:
        Number of retries when network requests fail.
    retry_backoff:
        Seconds to wait between retries (exponential backoff multiplier).
    format:
        Output format: ``"parquet"`` (default) or ``"csv"``.
    provider:
        Primary data source (``"alpha_vantage"`` or ``"yfinance"`` fallback).
    api_key:
        API key for Alpha Vantage. When ``None`` the ``ALPHAVANTAGE_API_KEY`` or
        ``ALPHA_VANTAGE_API_KEY`` environment variable is used.
    pause:
        Seconds to pause between Alpha Vantage requests to respect rate limits.
    """

    tickers: Iterable[str]
    start: str
    end: str
    interval: str = "1d"
    max_retries: int = 3
    retry_backoff: float = 2.0
    format: str = "parquet"
    provider: str = "alpha_vantage"
    api_key: Optional[str] = None
    pause: float = 12.0


def ensure_directory(path: Path) -> None:
    """Ensure that the parent directory exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _normalise_ticker_for_filename(ticker: str) -> str:
    """Normalise ticker symbols for filenames by stripping characters."""

    return ticker.replace("^", "")


def _alpha_vantage_equity(ticker: str, config: DownloadConfig) -> pd.DataFrame:
    """Download daily OHLCV data for equities via Alpha Vantage."""

    api_key = config.api_key or os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get(
        "ALPHA_VANTAGE_API_KEY"
    )
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key not provided. Set --api-key or ALPHAVANTAGE_API_KEY/ALPHA_VANTAGE_API_KEY."
        )
    if TimeSeries is None:
        raise ImportError("alpha_vantage package not installed. Install it or choose provider=yfinance")
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
    data = data.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
        }
    )
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end)
    data = data.loc[(data.index >= start) & (data.index <= end)]
    data["volume"] = data.get("volume")
    data["adj_close"] = data.get("adj_close", data["close"])
    data.reset_index(inplace=True)
    data.rename(columns={"index": "date"}, inplace=True)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    return data[["date", "open", "high", "low", "close", "adj_close", "volume"]]


def _alpha_vantage_fx(pair: str, config: DownloadConfig) -> pd.DataFrame:
    """Download FX/metals daily data via Alpha Vantage."""

    api_key = config.api_key or os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get(
        "ALPHA_VANTAGE_API_KEY"
    )
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key not provided. Set --api-key or ALPHAVANTAGE_API_KEY/ALPHA_VANTAGE_API_KEY."
        )
    if ForeignExchange is None:
        raise ImportError("alpha_vantage package not installed. Install it or choose provider=yfinance")
    fx_client = ForeignExchange(key=api_key, output_format="pandas")
    base = pair[:3].upper()
    quote = pair[3:].upper()
    data, _ = fx_client.get_currency_exchange_daily(
        from_symbol=base, to_symbol=quote, outputsize="full"
    )
    data = data.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
        }
    )
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end)
    data = data.loc[(data.index >= start) & (data.index <= end)]
    data["volume"] = np.nan
    data["adj_close"] = data["close"]
    data.reset_index(inplace=True)
    data.rename(columns={"index": "date"}, inplace=True)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    return data[["date", "open", "high", "low", "close", "adj_close", "volume"]]


def download_single_ticker(ticker: str, config: DownloadConfig) -> pd.DataFrame:
    """Download OHLCV data for a single ticker with retry logic and fallback."""

    providers = [config.provider]
    if "yfinance" not in providers:
        providers.append("yfinance")

    last_error: Optional[Exception] = None
    for provider in providers:
        attempt = 0
        while True:
            try:
                if provider == "alpha_vantage":
                    if config.interval != "1d":
                        raise ValueError("Alpha Vantage integration currently supports only daily data")
                    LOGGER.info("Downloading %s via Alpha Vantage", ticker)
                    if ticker.endswith("=X"):
                        pair = ticker.replace("=X", "")
                        data = _alpha_vantage_fx(pair, config)
                    elif ticker.startswith("^"):
                        raise ValueError(
                            "Alpha Vantage does not directly support index tickers prefixed with '^'."
                        )
                    else:
                        normalised = ticker.replace("^", "")
                        data = _alpha_vantage_equity(normalised, config)
                    time.sleep(config.pause)
                elif provider == "yfinance":
                    LOGGER.info("Downloading %s via yfinance", ticker)
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
                    data.reset_index(inplace=True)
                    data.rename(columns={"index": "date"}, inplace=True)
                    if "adj close" in data.columns:
                        data.rename(columns={"adj close": "adj_close"}, inplace=True)
                    else:
                        data["adj_close"] = data["close"]
                    if "volume" not in data.columns:
                        data["volume"] = np.nan
                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                if "date" not in data.columns:
                    raise ValueError("Downloaded data missing 'date' column")

                data["ticker"] = ticker
                data.sort_values("date", inplace=True)
                data = data.ffill().dropna(subset=["close"])
                return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempt += 1
                if attempt > config.max_retries:
                    LOGGER.warning(
                        "Provider %s failed for %s after %d attempts (%s)",
                        provider,
                        ticker,
                        attempt,
                        exc,
                    )
                    break
                sleep_time = config.retry_backoff ** attempt
                LOGGER.warning(
                    "Retry %s/%s for %s with provider %s after error: %s",
                    attempt,
                    config.max_retries,
                    ticker,
                    provider,
                    exc,
                )
                time.sleep(sleep_time)

        LOGGER.info("Falling back from provider %s for %s", provider, ticker)

    raise RuntimeError(f"Failed to download {ticker}") from last_error
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
        output_stem = _normalise_ticker_for_filename(ticker)
        output_path = RAW_DATA_DIR / f"{output_stem}.{config.format}"
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

    alpaca_api_key = "#"  # TODO: replace with Alpaca API key (e.g., read from env)
    alpaca_secret_key = "#"  # TODO: replace with Alpaca secret key (secure storage only)
    _ = (alpaca_api_key, alpaca_secret_key)  # keep placeholders referenced

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

    parser = argparse.ArgumentParser(description="Download OHLCV data from public APIs")
    parser = argparse.ArgumentParser(description="Download OHLCV data via yfinance")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers to download")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Sampling interval (default 1d)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff multiplier for retries")
    parser.add_argument(
        "--provider",
        choices=["alpha_vantage", "yfinance"],
        default="alpha_vantage",
        help="Primary data provider",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help=(
            "API key for Alpha Vantage (falls back to ALPHAVANTAGE_API_KEY or "
            "ALPHA_VANTAGE_API_KEY env vars)"
        ),
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=12.0,
        help="Seconds to pause between Alpha Vantage requests to respect rate limits",
    )
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
        provider=ns.provider,
        api_key=ns.api_key,
        pause=ns.pause,
    )
    download_ohlcv(config)


if __name__ == "__main__":
    main()
