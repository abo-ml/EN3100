"""Macro factor loading and integration for market forecasting.

This module provides functions to download macro indicators (VIX, Treasury yields,
S&P 500) and align them with price data for use as features in ML models.

Data Sources:
- yfinance: VIX (^VIX), S&P 500 (^GSPC), Treasury yields (^TNX)
- pandas_datareader: Alternative source for FRED data (optional)

Environment Variables:
- FRED_API_KEY: Optional API key for FRED data via pandas_datareader
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default macro tickers for yfinance
DEFAULT_MACRO_TICKERS = {
    "vix": "^VIX",           # VIX Volatility Index
    "sp500": "^GSPC",        # S&P 500 Index
    "treasury_10y": "^TNX",  # 10-Year Treasury Yield
}


def load_macro_factors_yfinance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tickers: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Download macro indicators using yfinance.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format. Defaults to 2 years ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.
    tickers : Dict[str, str], optional
        Mapping of factor names to yfinance ticker symbols.
        Defaults to VIX, S&P 500, and 10-year Treasury.

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and columns for each macro factor.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed. Returning empty macro factors.")
        return pd.DataFrame()

    tickers = tickers or DEFAULT_MACRO_TICKERS

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    macro_data = {}

    for name, ticker in tickers.items():
        try:
            logger.info(f"Downloading {name} ({ticker})...")
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
            if not data.empty:
                # Use close price for the macro indicator
                close_col = "Close" if "Close" in data.columns else data.columns[0]
                macro_data[name] = data[close_col]
        except Exception as e:
            logger.warning(f"Failed to download {name}: {e}")

    if not macro_data:
        return pd.DataFrame()

    # Combine all macro factors into a single DataFrame
    macro_df = pd.DataFrame(macro_data)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df.index.name = "date"

    # Forward-fill missing values (weekends, holidays)
    macro_df = macro_df.ffill()

    return macro_df


def load_macro_factors_datareader(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fred_series: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Download macro indicators from FRED using pandas_datareader.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format. Defaults to 2 years ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.
    fred_series : Dict[str, str], optional
        Mapping of factor names to FRED series IDs.
        Defaults to VIX, Treasury rates, etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and columns for each macro factor.

    Notes
    -----
    Requires FRED_API_KEY environment variable or pandas_datareader installation.
    """
    try:
        import pandas_datareader as pdr
        import pandas_datareader.data as web
    except ImportError:
        logger.warning("pandas_datareader not installed. Use yfinance source instead.")
        return pd.DataFrame()

    fred_series = fred_series or {
        "vix": "VIXCLS",        # VIX
        "treasury_10y": "DGS10",  # 10-Year Treasury
        "fed_funds": "FEDFUNDS",  # Federal Funds Rate
    }

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    api_key = os.environ.get("FRED_API_KEY")

    macro_data = {}

    for name, series_id in fred_series.items():
        try:
            logger.info(f"Downloading {name} from FRED ({series_id})...")
            data = web.DataReader(
                series_id,
                "fred",
                start=start_date,
                end=end_date,
            )
            if not data.empty:
                macro_data[name] = data.iloc[:, 0]
        except Exception as e:
            logger.warning(f"Failed to download {name} from FRED: {e}")

    if not macro_data:
        return pd.DataFrame()

    macro_df = pd.DataFrame(macro_data)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df.index.name = "date"
    macro_df = macro_df.ffill()

    return macro_df


def load_macro_factors(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Load macro factors from the specified source.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.
    source : str
        Data source: 'yfinance' (default) or 'fred'.

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and columns for each macro factor.
    """
    if source == "yfinance":
        return load_macro_factors_yfinance(start_date, end_date)
    elif source == "fred":
        return load_macro_factors_datareader(start_date, end_date)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'yfinance' or 'fred'.")


def compute_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived macro features from raw macro indicators.

    Parameters
    ----------
    macro_df : pd.DataFrame
        DataFrame with macro indicators (vix, sp500, treasury_10y, etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame with original and derived macro features.
    """
    features = macro_df.copy()

    # VIX-related features
    if "vix" in features.columns:
        features["vix_pct_change"] = features["vix"].pct_change()
        features["vix_20d_ma"] = features["vix"].rolling(20).mean()
        features["vix_above_ma"] = (features["vix"] > features["vix_20d_ma"]).astype(int)

    # S&P 500 features
    if "sp500" in features.columns:
        features["sp500_return"] = features["sp500"].pct_change()
        features["sp500_20d_momentum"] = features["sp500"].pct_change(20)

    # Treasury yield features
    if "treasury_10y" in features.columns:
        features["treasury_change"] = features["treasury_10y"].diff()

    return features


def merge_macro_features(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Merge macro features with price data.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price DataFrame with a date column.
    macro_df : pd.DataFrame
        Macro features DataFrame with date index.
    date_col : str
        Name of date column in price_df. Default is 'date'.

    Returns
    -------
    pd.DataFrame
        Price DataFrame with macro features added.
    """
    if macro_df.empty:
        return price_df

    # Reset index to make date a column for merging
    macro_reset = macro_df.reset_index()
    macro_reset.columns = [date_col if c == "date" else f"macro_{c}" for c in macro_reset.columns]

    # Ensure date columns are datetime
    price_df = price_df.copy()
    price_df[date_col] = pd.to_datetime(price_df[date_col])
    macro_reset[date_col] = pd.to_datetime(macro_reset[date_col])

    # Merge on date
    merged = price_df.merge(macro_reset, on=date_col, how="left")

    # Forward-fill macro values for any missing dates
    macro_cols = [c for c in merged.columns if c.startswith("macro_")]
    merged[macro_cols] = merged[macro_cols].ffill()

    return merged
