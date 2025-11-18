"""Reporting utilities for writing markdown and plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def save_metrics_report(metrics_dict: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Metrics Summary", ""]
    for key, value in metrics_dict.items():
        lines.append(f"- **{key}**: {value:.4f}" if isinstance(value, (int, float)) else f"- **{key}**: {value}")
    path.write_text("\n".join(lines))


def plot_pred_vs_actual(dates: pd.Series, actual: pd.Series, predicted: pd.Series, title: str, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, actual, label="Actual")
    ax.plot(dates, predicted, label="Predicted", alpha=0.7)
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_equity_curve(dates: pd.Series, equity_curve: pd.Series, title: str, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, equity_curve, label="Equity Curve")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
