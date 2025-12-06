"""Aggregate iteration metrics across reports.

This module scans the per-iteration markdown reports, extracts common
performance metrics (RMSE, MAE, R^2, Sharpe, Hit rate), and builds a
cross-iteration comparison table. It also produces quick bar charts for
the key metrics to visualise progress over iterations.

Usage
-----
python -m src.evaluation.summary
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import FIGURES_DIR, REPORTS_DIR


METRIC_KEYWORDS = {
    "RMSE": ["rmse"],
    "MAE": ["mae"],
    "R2": ["r2", "r²"],
    "Sharpe": ["sharpe"],
    "Hit rate": ["hit", "directional_accuracy", "accuracy"],
}


def _extract_numeric(line: str) -> Dict[str, float]:
    """Extract metric-name/value pairs from a markdown line.

    Expects patterns like ``**metric**: value``; returns a dict of
    lowercased keys to floats where possible.
    """

    matches = re.findall(r"\*\*(.+?)\*\*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", line)
    return {name.strip().lower(): float(val) for name, val in matches}


def parse_metrics(path: Path) -> Dict[str, float]:
    """Parse a metrics markdown file and summarise common metric types."""

    contents = path.read_text(encoding="utf-8")
    raw_pairs: Dict[str, float] = {}
    for line in contents.splitlines():
        raw_pairs.update(_extract_numeric(line))

    summary: Dict[str, float] = {}
    for canonical, keywords in METRIC_KEYWORDS.items():
        values: List[float] = []
        for key, val in raw_pairs.items():
            for kw in keywords:
                if kw in key:
                    values.append(val)
                    break
        if values:
            summary[canonical] = float(pd.Series(values).mean())
    return summary


def build_summary() -> pd.DataFrame:
    """Scan iteration result reports and collate metrics."""

    rows = []
    for report in sorted(REPORTS_DIR.glob("iteration_*_results.md")):
        iteration_name = report.stem.replace("_results", "")
        metrics = parse_metrics(report)
        if metrics:
            row = {"iteration": iteration_name}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("iteration") if rows else pd.DataFrame()
    if not df.empty:
        df.sort_index(inplace=True)
    return df


def save_summary(df: pd.DataFrame) -> None:
    """Persist the summary metrics to CSV and markdown."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "summary_metrics.csv"
    md_path = REPORTS_DIR / "summary_metrics.md"

    df.to_csv(csv_path)
    md_lines = ["# Iteration Metrics Summary", ""]
    md_lines.append(df.to_markdown())
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def plot_bars(df: pd.DataFrame) -> None:
    """Generate bar plots for key metrics across iterations."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = [m for m in ["RMSE", "MAE", "R2", "Sharpe"] if m in df.columns]

    for metric in metrics_to_plot:
        ax = df[metric].plot(kind="bar", title=f"{metric} by iteration", figsize=(8, 4))
        ax.set_ylabel(metric)
        fig = ax.get_figure()
        fig.tight_layout()
        fig_path = FIGURES_DIR / f"summary_{metric.lower()}.png"
        fig.savefig(fig_path, bbox_inches="tight")
        fig.savefig(f"summary_{metric.lower()}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    df = build_summary()
    if df.empty:
        print("No iteration reports found to summarise.")
        return
    save_summary(df)
    plot_bars(df)
    print("Saved summary metrics to reports/summary_metrics.csv and reports/summary_metrics.md")


if __name__ == "__main__":
    main()

