"""Data acquisition utilities for downloading historical OHLCV data.

This module handles interaction with Alpha Vantage (primary), Yahoo Finance,
and optionally Stooq as fallbacks. It performs basic cleaning, and stores the
results under ``data/raw`` for subsequent processing. Hooks are included for
integrating higher-frequency order book data from broker APIs. The functions
are designed for reproducibility in the accompanying dissertation project.
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

# Optional dependency for Stooq fallback provider
try:  # pragma: no cover
    from pandas_datareader import data as pdr
    HAS_PANDAS_DATAREADER = True
except (ImportError, TypeError):  # pragma: no cover
    # TypeError can occur with incompatible pandas versions
    pdr = None
    HAS_PANDAS_DATAREADER = False

from ..utils import RAW_DATA_DIR
RAW_DATA_DIR = Path("data/raw")
LOGGER = logging.getLogger(__name__)

# Known tickers that require premium Alpha Vantage subscription.
# These will automatically skip Alpha Vantage and use yfinance/stooq instead.
PREMIUM_ONLY_TICKERS = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA"}


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
        Sampling interval (currently ``1d`` supported for Alpha Vantage, yfinance supports others).
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
    providers: Optional[List[str]] = None
    api_key: Optional[str] = None
    pause: float = 12.0


def ensure_directory(path: Path) -> None:
    """Ensure that the parent directory exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _normalise_ticker_for_filename(ticker: str) -> str:
    """Normalise ticker symbols for filenames by stripping characters."""

    return ticker.replace("^", "")


def _log_warning(message: str) -> None:
    """Log a warning message."""
    LOGGER.warning(message)


def _get_api_key(config: DownloadConfig, provider: str) -> Optional[str]:
    """Get API key for the specified provider with clear warning if missing.

    For Alpha Vantage, checks config.api_key, then ALPHAVANTAGE_API_KEY,
    then ALPHA_VANTAGE_API_KEY environment variables.
    """
    if provider == "alpha_vantage":
        api_key = config.api_key or os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get(
            "ALPHA_VANTAGE_API_KEY"
        )
        if not api_key:
            LOGGER.warning(
                "Alpha Vantage API key not configured. "
                "Set ALPHAVANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY environment variable, "
                "or pass --api-key on the command line. "
                "Falling back to other providers."
            )
        elif not api_key.strip():
            LOGGER.warning(
                "Alpha Vantage API key is empty. "
                "Ensure ALPHAVANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY contains a valid key."
            )
            return None
        return api_key
    return None


def _convert_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLCV column names to lowercase, excluding Close/Adj Close.

    Maps Open→open, High→high, Low→low, Volume→volume.
    Close and Adj Close are handled separately to support special logic.
    """
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    }
    return df.rename(columns=rename_map)


def _basic_clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning on OHLCV data (forward fill and drop NaN close)."""
    df = df.ffill()
    if "close" in df.columns:
        df = df.dropna(subset=["close"])
    return df


def _alpha_vantage_equity(ticker: str, config: DownloadConfig) -> pd.DataFrame:
    """Download daily OHLCV data for equities via Alpha Vantage."""

    api_key = _get_api_key(config, "alpha_vantage")
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

    api_key = _get_api_key(config, "alpha_vantage")
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


def _download_alpha_vantage(ticker: str, config: DownloadConfig) -> pd.DataFrame:
    """Download OHLCV data via Alpha Vantage with appropriate endpoint selection."""
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
        data = _alpha_vantage_equity(ticker, config)
    time.sleep(config.pause)
    if "date" not in data.columns:
        raise ValueError("Downloaded data missing 'date' column")
    data["ticker"] = ticker
    data.sort_values("date", inplace=True)
    return data


def _download_yfinance(ticker: str, config: DownloadConfig) -> Optional[pd.DataFrame]:
    """Download OHLCV data via yfinance.

    Returns None if data is empty or required columns are missing.
    """
    LOGGER.info("Downloading %s via yfinance", ticker)
    # Step 1: Call yfinance download
    data = yf.download(
        ticker,
        start=config.start,
        end=config.end,
        interval=config.interval,
        progress=False,
    )
    # Step 2: If empty, return None
    if data.empty:
        LOGGER.warning("No data returned for %s from yfinance", ticker)
        return None

    # Step 3: Reset index and rename Date to date
    data.reset_index(inplace=True)
    # Handle both 'Date' (typical) and 'index' (after reset)
    if "Date" in data.columns:
        data.rename(columns={"Date": "date"}, inplace=True)
    elif "index" in data.columns:
        data.rename(columns={"index": "date"}, inplace=True)

    if "date" not in data.columns:
        LOGGER.warning("Downloaded data missing 'date' column for %s", ticker)
        return None

    # Ensure date is timezone-naive
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)

    # Step 4: Convert OHLCV columns to lowercase (excluding Close/Adj Close)
    data = _convert_ohlcv_columns(data)

    # Step 5: Create "close" and "adj_close" columns
    if "Adj Close" in data.columns:
        data["close"] = data["Adj Close"]
        data["adj_close"] = data["Adj Close"]
    elif "Close" in data.columns:
        data["close"] = data["Close"]
        data["adj_close"] = data["Close"]
    else:
        LOGGER.warning("Downloaded data missing 'Close' or 'Adj Close' for %s", ticker)
        return None

    # Ensure volume column exists
    if "volume" not in data.columns:
        data["volume"] = np.nan

    # Step 6: Sort by date, forward-fill missing values, drop rows where close is NaN
    data.sort_values("date", inplace=True)
    data = data.ffill()
    data = data.dropna(subset=["close"])

    # Add ticker column
    data["ticker"] = ticker

    # Step 7: Return the cleaned DataFrame
    return data


def _download_stooq(ticker: str, config: DownloadConfig) -> Optional[pd.DataFrame]:
    """Download OHLCV data via Stooq (pandas_datareader fallback).

    Stooq provides free data for many tickers when Alpha Vantage and yfinance fail.
    Requires pandas_datareader to be installed: pip install pandas_datareader>=0.10

    Returns None if data is empty or pandas_datareader is not installed.
    """
    if not HAS_PANDAS_DATAREADER:
        LOGGER.warning("pandas_datareader not installed, cannot use Stooq provider")
        return None

    LOGGER.info("Downloading %s via Stooq", ticker)
    try:
        data = pdr.DataReader(ticker, "stooq", start=config.start, end=config.end)
    except Exception as e:
        LOGGER.warning("Stooq download failed for %s: %s", ticker, e)
        return None

    if data.empty:
        LOGGER.warning("No data returned for %s from Stooq", ticker)
        return None

    # Stooq returns data in descending order, so sort ascending
    data = data.sort_index()
    data.reset_index(inplace=True)
    data.rename(columns={"Date": "date"}, inplace=True)

    if "date" not in data.columns:
        LOGGER.warning("Downloaded data missing 'date' column for %s", ticker)
        return None

    # Ensure date is timezone-naive
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)

    # Convert OHLCV columns to lowercase
    data = _convert_ohlcv_columns(data)

    # Create close and adj_close columns
    if "Close" in data.columns:
        data["close"] = data["Close"]
        data["adj_close"] = data["Close"]  # Stooq doesn't provide adjusted close
    else:
        LOGGER.warning("Downloaded data missing 'Close' for %s", ticker)
        return None

    if "volume" not in data.columns:
        data["volume"] = np.nan

    # Sort by date, forward-fill missing values, drop rows where close is NaN
    data.sort_values("date", inplace=True)
    data = data.ffill()
    data = data.dropna(subset=["close"])

    data["ticker"] = ticker
    return data


def download_single_ticker(ticker: str, config: DownloadConfig) -> pd.DataFrame:
    """Download OHLCV data for a single ticker with provider fallback.

    Iterates over providers (default: alpha_vantage, yfinance, stooq) and returns
    the first successful download. Premium-only tickers (e.g., AAPL) automatically
    skip Alpha Vantage to avoid premium endpoint errors. Stooq is used as a final
    fallback if pandas_datareader is installed.
    """
    last_err: Optional[Exception] = None
    # Default provider order: alpha_vantage, yfinance, stooq (if installed)
    default_providers = ["alpha_vantage", "yfinance"]
    if HAS_PANDAS_DATAREADER:
        default_providers.append("stooq")
    providers = getattr(config, "providers", None) or default_providers

    for provider in providers:
        try:
            if provider == "alpha_vantage":
                # Skip Alpha Vantage for known premium-only tickers
                if ticker.upper() in PREMIUM_ONLY_TICKERS:
                    LOGGER.info(
                        "Skipping Alpha Vantage for %s (premium-only ticker)", ticker
                    )
                    continue
                df = _download_alpha_vantage(ticker, config)
            elif provider == "yfinance":
                df = _download_yfinance(ticker, config)
            elif provider == "stooq":
                df = _download_stooq(ticker, config)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            if df is None or df.empty:
                raise RuntimeError(f"No data returned by provider: {provider}")
            return _basic_clean_ohlcv(df)
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log_warning(f"Provider {provider} failed for {ticker}: {e}")
            continue
    raise RuntimeError(f"All providers failed for {ticker}") from last_err


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

    # Example wiring (commented out) for Alpaca Market Data v2 order book endpoint.
    # Replace the placeholders with your credentials and uncomment when ready:
    #
    # alpaca_api_key = "#"  # TODO: replace with Alpaca API key (e.g., read from env)
    # alpaca_secret_key = "#"  # TODO: replace with Alpaca secret key (secure storage only)
    # import requests
    # headers = {"APCA-API-KEY-ID": alpaca_api_key, "APCA-API-SECRET-KEY": alpaca_secret_key}
    # url = f"https://data.alpaca.markets/v2/stocks/{ticker}/orderbooks/latest"
    # response = requests.get(url, headers=headers, timeout=10)
    # response.raise_for_status()
    # payload = response.json()
    # return pd.DataFrame(
    #     {
    #         "timestamp": [pd.to_datetime(payload["timestamp"]).tz_localize(None)],
    #         "ticker": [ticker],
    #         "bid_volume": [payload["bids"][0]["s"] if payload.get("bids") else np.nan],
    #         "ask_volume": [payload["asks"][0]["s"] if payload.get("asks") else np.nan],
    #         "bid_price": [payload["bids"][0]["p"] if payload.get("bids") else np.nan],
    #         "ask_price": [payload["asks"][0]["p"] if payload.get("asks") else np.nan],
    #     }
    # )
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


# Module-level parser for testing and inspection
parser = argparse.ArgumentParser(description="Download OHLCV data from public APIs")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    _parser = argparse.ArgumentParser(description="Download OHLCV data from public APIs")
    _parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers to download")
    _parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    _parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    _parser.add_argument("--interval", default="1d", help="Sampling interval (default 1d)")
    _parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    _parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    _parser.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff multiplier for retries")
    _parser.add_argument(
        "--provider",
        choices=["alpha_vantage", "yfinance"],
        default="alpha_vantage",
        help="Primary data provider",
    )
    _parser.add_argument(
        "--api-key",
        dest="api_key",
        help=(
            "API key for Alpha Vantage (falls back to ALPHAVANTAGE_API_KEY or "
            "ALPHA_VANTAGE_API_KEY env vars)"
        ),
    )
    _parser.add_argument(
        "--pause",
        type=float,
        default=12.0,
        help="Seconds to pause between Alpha Vantage requests to respect rate limits",
    )
    return _parser.parse_args(args)


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
