#!/usr/bin/env python3
"""Equity Curve Comparison Plot with Alpaca Market Data Integration.

This script plots a comparison between a model's predicted equity curve and
the actual market returns for a given ticker using Alpaca's market data API.

Usage:
    python scripts/visualizations/plot_iteration_equity_curves.py \
        --ticker AAPL \
        --start 2023-01-01 \
        --end 2024-01-01 \
        --output figures/equity_curve_comparison.png

Environment Variables:
    APCA_API_KEY_ID: Alpaca API key ID
    APCA_API_SECRET_KEY: Alpaca API secret key
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import FIGURES_DIR, PROCESSED_DIR, get_api_key

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def fetch_alpaca_bars(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily bars from Alpaca using the alpaca-py SDK.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: open, high, low, close, volume, vwap, trade_count.
        Index is a DatetimeIndex.

    Raises
    ------
    ValueError
        If Alpaca API keys are not configured.
    """
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = get_api_key("APCA_API_KEY_ID")
    api_secret = get_api_key("APCA_API_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca API keys not configured. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY environment variables."
        )

    client = StockHistoricalDataClient(api_key, api_secret)

    start_time = pd.to_datetime(start_date).tz_localize("America/New_York")
    end_time = None
    if end_date:
        end_time = pd.to_datetime(end_date).tz_localize("America/New_York")

    request = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=start_time,
        end=end_time,
    )

    bars = client.get_stock_bars(request)
    bars_df = bars.df

    if bars_df.empty:
        raise ValueError(f"No data returned for {ticker} from {start_date} to {end_date}")

    # Convert multi-index to simple DatetimeIndex
    if isinstance(bars_df.index, pd.MultiIndex):
        bars_df = bars_df.droplevel("symbol")
    bars_df = bars_df.tz_convert("America/New_York")

    LOGGER.info("Fetched %d bars for %s from Alpaca", len(bars_df), ticker)
    return bars_df


def compute_equity_curve(prices: pd.Series) -> pd.Series:
    """Compute cumulative equity curve from prices.

    Parameters
    ----------
    prices : pd.Series
        Series of closing prices.

    Returns
    -------
    pd.Series
        Cumulative equity curve starting at 1.0.
    """
    returns = prices.pct_change().fillna(0)
    equity_curve = (1 + returns).cumprod()
    return equity_curve


def load_model_equity_curve(ticker: str) -> Optional[pd.DataFrame]:
    """Load the model's predicted equity curve from processed data.

    Attempts to load iteration-specific backtest results that include
    a predicted equity curve.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol to match.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with 'date' and 'equity_curve' columns, or None if not found.
    """
    # Try to load iteration 5 strategy results
    strategy_path = PROCESSED_DIR / "iteration5_strategy_returns.csv"
    if strategy_path.exists():
        df = pd.read_csv(strategy_path, parse_dates=["date"])
        if "equity_curve" in df.columns:
            LOGGER.info("Loaded model equity curve from %s", strategy_path)
            return df[["date", "equity_curve"]]

    # Try ticker-specific backtest results
    ticker_path = PROCESSED_DIR / f"{ticker.lower()}_backtest_results.csv"
    if ticker_path.exists():
        df = pd.read_csv(ticker_path, parse_dates=["date"])
        if "equity_curve" in df.columns:
            LOGGER.info("Loaded model equity curve from %s", ticker_path)
            return df[["date", "equity_curve"]]

    LOGGER.warning("No model equity curve found for %s", ticker)
    return None


def plot_equity_comparison(
    market_equity: pd.Series,
    model_equity: Optional[pd.Series],
    ticker: str,
    output_path: Path,
) -> None:
    """Plot market vs model equity curves.

    Parameters
    ----------
    market_equity : pd.Series
        Market (buy-and-hold) equity curve indexed by date.
    model_equity : pd.Series, optional
        Model's predicted equity curve indexed by date.
    ticker : str
        Stock ticker symbol for labeling.
    output_path : Path
        Path to save the output figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot market equity curve
    ax.plot(
        market_equity.index,
        market_equity.values,
        label=f"{ticker} Buy & Hold",
        color="steelblue",
        linewidth=2,
    )

    # Plot model equity curve if available
    if model_equity is not None:
        ax.plot(
            model_equity.index,
            model_equity.values,
            label="Model Strategy",
            color="darkorange",
            linewidth=2,
        )

    ax.set_title(f"Equity Curve Comparison: {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns (normalized to 1.0)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved equity curve comparison to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot equity curve comparison using Alpaca market data"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Start date in YYYY-MM-DD format (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path (default: figures/equity_curve_comparison_TICKER.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ticker = args.ticker.upper()
    output_path = args.output or FIGURES_DIR / f"equity_curve_comparison_{ticker.lower()}.png"

    LOGGER.info("Fetching Alpaca market data for %s", ticker)

    try:
        bars_df = fetch_alpaca_bars(ticker, args.start, args.end)
    except ValueError as e:
        LOGGER.error("Failed to fetch Alpaca data: %s", e)
        print(f"Error: {e}")
        return

    # Compute market equity curve from closing prices
    market_equity = compute_equity_curve(bars_df["close"])

    # Load model equity curve if available
    model_df = load_model_equity_curve(ticker)
    model_equity = None
    if model_df is not None:
        # Align model dates with market dates
        model_df = model_df.set_index("date")
        # Filter to overlapping dates
        common_dates = market_equity.index.intersection(model_df.index)
        if len(common_dates) > 0:
            model_equity = model_df.loc[common_dates, "equity_curve"]
            market_equity = market_equity.loc[common_dates]
            LOGGER.info("Found %d overlapping dates between market and model data", len(common_dates))
        else:
            LOGGER.warning("No overlapping dates between market and model data")

    # Plot comparison
    plot_equity_comparison(market_equity, model_equity, ticker, output_path)


if __name__ == "__main__":
    main()
