#!/usr/bin/env python3
"""Prompt 1 — Bar Chart with Error Bars (Mean ± Std per Iteration).

Load per-ticker metrics, compute mean ± std of directional accuracy per iteration,
and plot a bar chart with error bars.

This figure gives examiners a quick read of overall performance and variability
across all 20 stocks.

Usage:
    python scripts/visualizations/plot_iteration_mean_accuracy_bar.py \
        --input results/20_stock/ticker_iteration_metrics.csv \
        --output figures/iteration_mean_accuracy_bar.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import FIGURES_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bar chart of mean directional accuracy per iteration with error bars"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "20_stock" / "ticker_iteration_metrics.csv",
        help="Path to ticker_iteration_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "iteration_mean_accuracy_bar.png",
        help="Output figure path",
    )
    return parser.parse_args()


def plot_iteration_mean_accuracy_bar(input_path: Path, output_path: Path) -> None:
    """Generate bar chart with error bars showing mean ± std of directional accuracy."""
    df = pd.read_csv(input_path)
    summary = df.groupby("iteration")["directional_accuracy"].agg(["mean", "std"]).reset_index()

    # Sort iterations numerically
    def to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return float("inf")

    summary["iteration_num"] = summary["iteration"].astype(str).apply(to_float)
    summary = summary.sort_values("iteration_num").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        summary["iteration"].astype(str),
        summary["mean"],
        yerr=summary["std"],
        capsize=4,
        color="steelblue",
        edgecolor="black",
    )
    ax.set_title("Mean Directional Accuracy per Iteration (with Std)")
    ax.set_xlabel("Model Iteration")
    ax.set_ylabel("Mean Directional Accuracy")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart to {output_path}")


def main() -> None:
    args = parse_args()
    plot_iteration_mean_accuracy_bar(args.input, args.output)


if __name__ == "__main__":
    main()
