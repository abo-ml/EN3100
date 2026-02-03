#!/usr/bin/env python3
"""Prompt 2 — Line Chart per Stock Across Iterations.

Select a handful of representative tickers (e.g. top, median, worst performers)
and plot each stock's directional accuracy across model iterations.

This plot highlights whether improvements are consistent across assets or
concentrated on a few.

Usage:
    python scripts/visualizations/plot_stock_line_comparison.py \
        --input results/20_stock/ticker_iteration_metrics.csv \
        --tickers AAPL MSFT TSLA \
        --output figures/stock_line_comparison.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import FIGURES_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate line chart of directional accuracy across iterations for selected stocks"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "20_stock" / "ticker_iteration_metrics.csv",
        help="Path to ticker_iteration_metrics.csv",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="List of tickers to plot. If not provided, selects top, median, and worst performers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "stock_line_comparison.png",
        help="Output figure path",
    )
    return parser.parse_args()


def select_representative_tickers(df: pd.DataFrame) -> List[str]:
    """Select top, median, and worst performers based on mean directional accuracy."""
    mean_da = df.groupby("ticker")["directional_accuracy"].mean().sort_values(ascending=False)

    if len(mean_da) >= 3:
        top = mean_da.index[0]
        median_idx = len(mean_da) // 2
        median = mean_da.index[median_idx]
        worst = mean_da.index[-1]
        return [top, median, worst]
    return mean_da.index.tolist()


def plot_stock_line_comparison(
    input_path: Path, output_path: Path, tickers: Optional[List[str]] = None
) -> None:
    """Generate line chart of directional accuracy across iterations for selected stocks."""
    df = pd.read_csv(input_path)

    # Select tickers if not provided
    if not tickers:
        tickers = select_representative_tickers(df)

    # Convert iteration to numeric for sorting
    def to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return float("inf")

    df["iteration_num"] = df["iteration"].astype(str).apply(to_float)

    fig, ax = plt.subplots(figsize=(10, 5))

    for ticker in tickers:
        subset = df[df["ticker"] == ticker].copy()
        if subset.empty:
            print(f"Warning: Ticker {ticker} not found in data")
            continue
        subset = subset.sort_values("iteration_num")
        ax.plot(
            subset["iteration"].astype(str),
            subset["directional_accuracy"],
            marker="o",
            label=ticker,
            linewidth=2,
        )

    ax.set_title("Directional Accuracy Across Iterations for Selected Stocks")
    ax.set_xlabel("Model Iteration")
    ax.set_ylabel("Directional Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved line chart to {output_path}")


def main() -> None:
    args = parse_args()
    plot_stock_line_comparison(args.input, args.output, args.tickers)


if __name__ == "__main__":
    main()
