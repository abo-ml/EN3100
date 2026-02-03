#!/usr/bin/env python3
"""Optional 3D Surface Plot (Appendix).

For completeness, produce a 3D surface showing directional accuracy by
stock (y-axis) and iteration (x-axis) with DA as the z-axis.

Consider keeping it in the appendix given readability concerns.

Usage:
    python scripts/visualizations/plot_3d_surface_accuracy.py \
        --input results/20_stock/ticker_iteration_metrics.csv \
        --output figures/3d_surface_accuracy.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.utils import FIGURES_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D surface plot of directional accuracy by stock and iteration"
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
        default=FIGURES_DIR / "3d_surface_accuracy.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=30,
        help="Elevation angle for 3D view",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=45,
        help="Azimuth angle for 3D view",
    )
    return parser.parse_args()


def plot_3d_surface_accuracy(
    input_path: Path,
    output_path: Path,
    elevation: float = 30,
    azimuth: float = 45,
) -> None:
    """Generate 3D surface plot of directional accuracy by stock and iteration."""
    df = pd.read_csv(input_path)

    stocks = sorted(df["ticker"].unique())

    # Sort iterations numerically
    def to_float(x: str) -> float:
        try:
            return float(str(x))
        except ValueError:
            return float("inf")

    iters = sorted(df["iteration"].unique(), key=to_float)
    iters = [str(i) for i in iters]

    # Build z-matrix
    z = np.empty((len(stocks), len(iters)))
    for i, stock in enumerate(stocks):
        for j, it in enumerate(iters):
            val = df[(df["ticker"] == stock) & (df["iteration"].astype(str) == it)][
                "directional_accuracy"
            ].mean()
            z[i, j] = val if not np.isnan(val) else 0

    x, y = np.meshgrid(range(len(iters)), range(len(stocks)))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none", alpha=0.8)

    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels(iters, rotation=45)
    ax.set_yticks(range(len(stocks)))
    ax.set_yticklabels(stocks, fontsize=8)
    ax.set_xlabel("Model Iteration")
    ax.set_ylabel("Stock")
    ax.set_zlabel("Directional Accuracy")
    ax.set_title("3D Surface of Directional Accuracy by Stock & Iteration")
    ax.view_init(elev=elevation, azim=azimuth)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Directional Accuracy")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D surface plot to {output_path}")


def main() -> None:
    args = parse_args()
    plot_3d_surface_accuracy(args.input, args.output, args.elevation, args.azimuth)


if __name__ == "__main__":
    main()
