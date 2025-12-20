"""S&P 500 universe selection utilities."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from src.utils import REFERENCE_DIR, ensure_directories

LOGGER = logging.getLogger(__name__)
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _normalise_ticker(ticker: str) -> str:
    """Normalise ticker symbols so yfinance understands them."""

    return ticker.replace(".", "-").strip()


def fetch_sp500_constituents(cache_path: Path = REFERENCE_DIR / "sp500_constituents.csv") -> pd.DataFrame:
    """Fetch the S&P 500 table from Wikipedia and cache it to disk."""

    ensure_directories(cache_path.parent)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    tables = pd.read_html(WIKI_URL)
    if not tables:
        raise RuntimeError("Unable to fetch S&P 500 constituents from Wikipedia.")
    table = tables[0]
    table.columns = [col.lower().replace(" ", "_") for col in table.columns]
    table["ticker"] = table["symbol"].apply(_normalise_ticker)
    table.rename(columns={"gics_sector": "sector"}, inplace=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, index=False)
    LOGGER.info("Cached %d S&P 500 constituents to %s", len(table), cache_path)
    return table


def _latest_price(ticker: str) -> Optional[float]:
    """Return the latest available price for a ticker."""

    ticker_obj = yf.Ticker(ticker)
    try:
        fast_info = getattr(ticker_obj, "fast_info", None)
        last_price = None
        if isinstance(fast_info, dict):
            last_price = fast_info.get("last_price")
        else:
            last_price = getattr(fast_info, "last_price", None)
        if last_price is not None:
            return float(last_price)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("fast_info unavailable for %s (%s)", ticker, exc)

    try:
        hist = ticker_obj.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("history lookup failed for %s (%s)", ticker, exc)
    return None


def select_equity_universe(
    n: int = 20,
    sector: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    random_state: int = 42,
) -> List[str]:
    """Select an equity universe with optional sector and price filters."""

    constituents = fetch_sp500_constituents()
    candidates = constituents.copy()

    if sector:
        candidates = candidates[candidates["sector"].str.lower() == sector.lower()]
        LOGGER.info("Filtered to sector '%s' -> %d tickers", sector, len(candidates))

    if price_min is not None or price_max is not None:
        prices = []
        for ticker in candidates["ticker"]:
            price = _latest_price(ticker)
            prices.append({"ticker": ticker, "price": price})
        price_df = pd.DataFrame(prices).dropna(subset=["price"])
        if price_df.empty:
            LOGGER.warning("Skipping price filters because no price data were available.")
        else:
            if price_min is not None:
                price_df = price_df[price_df["price"] >= price_min]
            if price_max is not None:
                price_df = price_df[price_df["price"] <= price_max]
            candidates = candidates.merge(price_df[["ticker"]], on="ticker", how="inner")
            LOGGER.info("Price filter retained %d tickers", len(candidates))

    available_tickers = candidates["ticker"].unique().tolist()
    rng = random.Random(random_state)
    if len(available_tickers) <= n:
        selected = available_tickers
    else:
        selected = rng.sample(available_tickers, n)
    selected = sorted(selected)
    LOGGER.info("Selected %d tickers", len(selected))
    return selected
