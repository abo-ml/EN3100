"""Download an optional S&P 500 equity universe and persist raw files."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from src.data.download_data import DownloadConfig, download_ohlcv
from src.universe.sp500_universe import select_equity_universe
from src.utils import RAW_DATA_DIR, REFERENCE_DIR, ensure_directories

LOGGER = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and download an S&P 500 equity universe.")
    parser.add_argument("--n", type=int, default=20, help="Number of tickers to sample.")
    parser.add_argument("--sector", type=str, default=None, help="Optional GICS sector filter.")
    parser.add_argument("--price-min", type=float, default=None, help="Minimum last price filter.")
    parser.add_argument("--price-max", type=float, default=None, help="Maximum last price filter.")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--provider", choices=["alpha_vantage", "yfinance"], default="yfinance", help="Data provider.")
    parser.add_argument("--interval", type=str, default="1d", help="Sampling interval for downloads.")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output format.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args(args)


def save_universe(tickers: List[str], n: int) -> Path:
    ensure_directories(REFERENCE_DIR)
    output_path = REFERENCE_DIR / f"equity_universe_{n}.txt"
    output_path.write_text("\n".join(tickers))
    return output_path


def main(cmd_args: Optional[List[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)
    ensure_directories(RAW_DATA_DIR, REFERENCE_DIR)

    tickers = select_equity_universe(
        n=ns.n, sector=ns.sector, price_min=ns.price_min, price_max=ns.price_max, random_state=ns.random_state
    )
    if not tickers:
        raise RuntimeError("No tickers selected; adjust filters and retry.")
    universe_path = save_universe(tickers, ns.n)
    LOGGER.info("Selected tickers saved to %s", universe_path)
    print("\n".join(tickers))

    config = DownloadConfig(
        tickers=tickers,
        start=ns.start,
        end=ns.end,
        interval=ns.interval,
        format=ns.format,
        provider=ns.provider,
    )
    download_ohlcv(config)


if __name__ == "__main__":
    main()
