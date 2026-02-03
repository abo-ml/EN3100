#!/usr/bin/env python3
"""Prompt 3 — Pairwise Improvement Heatmap.

Use ticker_iteration_metrics.csv to show per-stock improvement over the
linear baseline (Iteration 1) for each subsequent iteration.

This heatmap reveals which assets benefit from complex models and which don't.

Usage:
    python scripts/visualizations/plot_improvement_heatmap.py \
        --input results/20_stock/ticker_iteration_metrics.csv \
        --output figures/improvement_heatmap.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import FIGURES_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heatmap of directional accuracy improvement over baseline per stock"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "20_stock" / "ticker_iteration_metrics.csv",
        help="Path to ticker_iteration_metrics.csv",
    )
    parser.add_argument(
        "--baseline-iteration",
        type=str,
        default="1",
        help="Iteration to use as baseline (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "improvement_heatmap.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-0.3,
        help="Minimum value for color scale (default: -0.3)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.3,
        help="Maximum value for color scale (default: 0.3)",
    )
    return parser.parse_args()


def plot_improvement_heatmap(
    input_path: Path,
    output_path: Path,
    baseline_iteration: str = "1",
    vmin: float = -0.3,
    vmax: float = 0.3,
) -> None:
    """Generate heatmap showing improvement over baseline iteration."""
    df = pd.read_csv(input_path)
    df["iteration"] = df["iteration"].astype(str)

    # Get baseline values
    baseline = df[df["iteration"] == baseline_iteration][
        ["ticker", "directional_accuracy"]
    ].set_index("ticker")

    # Sort iterations numerically
    def to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return float("inf")

    iter_cols = sorted(df["iteration"].unique(), key=to_float)

    # Compute difference from baseline for each iteration/ticker
    improvement = pd.DataFrame(index=baseline.index, columns=iter_cols, dtype=float)

    for it in iter_cols:
        this_iter = df[df["iteration"] == it][
            ["ticker", "directional_accuracy"]
        ].set_index("ticker")
        diff = this_iter["directional_accuracy"] - baseline["directional_accuracy"]
        improvement[it] = diff

    # Sort tickers alphabetically
    improvement = improvement.sort_index()

    # Convert to numeric array for imshow
    values = improvement.values.astype(float)

    # Handle NaN values
    values = np.nan_to_num(values, nan=0.0)

    # Plot as heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Directional Accuracy Improvement over Baseline")

    ax.set_xticks(range(len(iter_cols)))
    ax.set_xticklabels(iter_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(improvement.index)))
    ax.set_yticklabels(improvement.index)
    ax.set_xlabel("Model Iteration")
    ax.set_ylabel("Ticker")
    ax.set_title(f"Model Improvement (DA) over Linear Baseline (Iteration {baseline_iteration}) per Stock")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved improvement heatmap to {output_path}")


def main() -> None:
    args = parse_args()
    plot_improvement_heatmap(
        args.input, args.output, args.baseline_iteration, args.vmin, args.vmax
    )


if __name__ == "__main__":
    main()
