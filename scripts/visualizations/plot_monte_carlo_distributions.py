#!/usr/bin/env python3
"""Prompt 5 — Risk Distribution Histograms (Monte Carlo Results).

After running monte_carlo.py, plot distributions of final equity and
max drawdown to link predictive power with risk outcomes.

These histograms let you discuss the variability of strategy outcomes
beyond mean accuracy.

Usage:
    python scripts/visualizations/plot_monte_carlo_distributions.py \
        --input results/iteration5_monte_carlo_results.csv \
        --output figures/monte_carlo_distributions.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import FIGURES_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Monte Carlo distribution histograms for final equity and max drawdown"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "iteration5_monte_carlo_results.csv",
        help="Path to Monte Carlo results CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "monte_carlo_distributions.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins",
    )
    return parser.parse_args()


def plot_monte_carlo_distributions(
    input_path: Path, output_path: Path, bins: int = 50
) -> None:
    """Generate side-by-side histograms of final equity and max drawdown distributions."""
    sim_df = pd.read_csv(input_path)

    # Determine available columns (support various naming conventions)
    equity_col = None
    drawdown_col = None

    for col in ["final_equity", "equity", "final_value"]:
        if col in sim_df.columns:
            equity_col = col
            break

    for col in ["max_drawdown", "drawdown", "max_dd"]:
        if col in sim_df.columns:
            drawdown_col = col
            break

    if equity_col is None and drawdown_col is None:
        raise ValueError(
            f"No suitable columns found in {input_path}. "
            f"Expected 'final_equity' or 'max_drawdown'. Available: {sim_df.columns.tolist()}"
        )

    # Create subplot layout based on available data
    num_plots = sum([equity_col is not None, drawdown_col is not None])
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    if num_plots == 1:
        axs = [axs]

    plot_idx = 0

    if equity_col is not None:
        axs[plot_idx].hist(sim_df[equity_col], bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
        axs[plot_idx].set_title("Distribution of Final Equity (Monte Carlo)")
        axs[plot_idx].set_xlabel("Final Equity")
        axs[plot_idx].set_ylabel("Frequency")
        axs[plot_idx].axvline(
            sim_df[equity_col].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {sim_df[equity_col].mean():.2f}",
        )
        axs[plot_idx].legend()
        plot_idx += 1

    if drawdown_col is not None:
        axs[plot_idx].hist(sim_df[drawdown_col], bins=bins, color="darkorange", edgecolor="black", alpha=0.7)
        axs[plot_idx].set_title("Distribution of Max Drawdown")
        axs[plot_idx].set_xlabel("Maximum Drawdown")
        axs[plot_idx].set_ylabel("Frequency")
        axs[plot_idx].axvline(
            sim_df[drawdown_col].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {sim_df[drawdown_col].mean():.4f}",
        )
        axs[plot_idx].legend()

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Monte Carlo distributions to {output_path}")


def main() -> None:
    args = parse_args()
    plot_monte_carlo_distributions(args.input, args.output, args.bins)


if __name__ == "__main__":
    main()
