#!/usr/bin/env python3
"""Generate a walk-forward split schematic timeline.

Usage:
    python scripts/plot_walk_forward.py \
        --start 2013-01-01 --end 2023-12-31 \
        --splits 5 --train_ratio 0.8 --gap_days 0 \
        --fig reports/fig_walk_forward.png
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_splits(
    start: datetime,
    end: datetime,
    splits: int,
    train_ratio: float,
    gap_days: int,
) -> list[dict]:
    """Compute walk-forward train/test splits.

    Returns a list of dicts with 'train_start', 'train_end', 'test_start', 'test_end'.
    """
    total_days = (end - start).days
    split_size = total_days // splits
    results = []
    for i in range(splits):
        split_start = start + timedelta(days=i * split_size)
        split_end = split_start + timedelta(days=split_size)
        if i == splits - 1:
            split_end = end
        train_days = int((split_end - split_start).days * train_ratio)
        train_start = split_start
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=gap_days)
        test_end = split_end
        results.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return results


def plot_walk_forward(
    start: datetime,
    end: datetime,
    splits: int,
    train_ratio: float,
    gap_days: int,
    fig_path: Path,
) -> None:
    """Draw and save the walk-forward schematic."""
    split_data = compute_splits(start, end, splits, train_ratio, gap_days)
    fig, ax = plt.subplots(figsize=(12, max(3, splits * 0.8)))
    ax.set_xlim(start, end)
    ax.set_ylim(0, splits + 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Split")
    ax.set_title("Walk-Forward Validation Splits")
    ax.set_yticks(range(1, splits + 1))
    ax.set_yticklabels([f"Split {i}" for i in range(1, splits + 1)])
    ax.invert_yaxis()

    for i, s in enumerate(split_data, start=1):
        # Train bar (blue)
        train_width = (s["train_end"] - s["train_start"]).days
        ax.barh(
            i,
            train_width,
            left=s["train_start"],
            height=0.4,
            color="steelblue",
            edgecolor="black",
            label="Train" if i == 1 else "",
        )
        # Test bar (orange)
        test_width = (s["test_end"] - s["test_start"]).days
        ax.barh(
            i,
            test_width,
            left=s["test_start"],
            height=0.4,
            color="darkorange",
            edgecolor="black",
            label="Test" if i == 1 else "",
        )

    # Legend
    train_patch = mpatches.Patch(color="steelblue", label="Train")
    test_patch = mpatches.Patch(color="darkorange", label="Test")
    ax.legend(handles=[train_patch, test_patch], loc="upper right")

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved walk-forward schematic to {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate walk-forward split schematic")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--splits", type=int, default=5, help="Number of walk-forward splits")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Fraction of each split for training")
    parser.add_argument("--gap_days", type=int, default=0, help="Gap days between train and test")
    parser.add_argument("--fig", required=True, help="Output figure path (e.g., reports/fig_walk_forward.png)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    plot_walk_forward(
        start=start,
        end=end,
        splits=args.splits,
        train_ratio=args.train_ratio,
        gap_days=args.gap_days,
        fig_path=Path(args.fig),
    )


if __name__ == "__main__":
    main()
