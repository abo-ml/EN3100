"""Per-asset evaluation for the core 4-asset universe."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.models.iteration1_1_svr import run_iteration as run_iteration_1_1
from src.models.iteration1_baseline import run_iteration as run_iteration_1
from src.models.iteration2_1_lightgbm import run_iteration as run_iteration_2_1
from src.models.iteration2_ensemble import run_iteration as run_iteration_2
from src.utils import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_directories

LOGGER = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "EURUSD=X", "XAUUSD=X", "^GSPC"]
MODEL_CONFIGS = [
    (
        "Iteration 1 Linear",
        run_iteration_1,
        {
            "rmse": "rmse_linear_mean",
            "mae": "mae_linear_mean",
            "r2": "r2_linear_mean",
            "directional_accuracy": "directional_accuracy_linear_mean",
        },
    ),
    (
        "Iteration 1.1 SVR",
        run_iteration_1_1,
        {
            "rmse": "rmse_svr_mean",
            "mae": "mae_svr_mean",
            "r2": "r2_svr_mean",
            "directional_accuracy": "directional_accuracy_svr_mean",
        },
    ),
    (
        "Iteration 2 RandomForest",
        run_iteration_2,
        {
            "rmse": "rmse_rf_mean",
            "mae": "mae_rf_mean",
            "r2": "r2_rf_mean",
            "directional_accuracy": "directional_accuracy_rf_mean",
        },
    ),
    (
        "Iteration 2 XGBoost",
        run_iteration_2,
        {
            "rmse": "rmse_xgb_mean",
            "mae": "mae_xgb_mean",
            "r2": "r2_xgb_mean",
            "directional_accuracy": "directional_accuracy_xgb_mean",
        },
    ),
    (
        "Iteration 2.1 LightGBM",
        run_iteration_2_1,
        {
            "rmse": "rmse_lgbm_mean",
            "mae": "mae_lgbm_mean",
            "r2": "r2_lgbm_mean",
            "directional_accuracy": "directional_accuracy_lgbm_mean",
        },
    ),
]


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-asset evaluation for the 4-asset universe.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Tickers to evaluate. Defaults to the core 4-asset universe.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_DIR / "model_features.parquet",
        help="Path to model features parquet file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPORTS_DIR / "per_asset_metrics.csv",
        help="Where to write the metrics CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORTS_DIR / "per_asset_metrics.md",
        help="Where to write the markdown summary.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to append to output filenames (e.g., '_it' -> per_asset_metrics_it.csv).",
    )
    return parser.parse_args(args)


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model features not found at {path}. Run feature engineering first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def extract_metrics(summary: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for metric, key in mapping.items():
        metrics[metric] = summary.get(key)
    return metrics


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + " | ".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_directional_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    da_df = df.dropna(subset=["directional_accuracy"])
    if da_df.empty:
        LOGGER.warning("No directional accuracy values available to plot.")
        return

    mean_da = da_df.groupby("model_name")["directional_accuracy"].mean()
    std_da = da_df.groupby("model_name")["directional_accuracy"].std().fillna(0.0)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(mean_da.index, mean_da.values, yerr=std_da.values, capsize=5)
    ax.set_ylabel("Directional Accuracy")
    ax.set_title("Directional Accuracy by Model (Per-Asset Evaluation)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_ticker(ticker: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    ticker_df = df[df["ticker"] == ticker].reset_index(drop=True)
    if ticker_df.empty:
        LOGGER.warning("No data available for %s; skipping.", ticker)
        return []

    results: List[Dict[str, object]] = []
    for model_name, runner, mapping in MODEL_CONFIGS:
        summary = runner(df=ticker_df, generate_reports=False, ticker=ticker)
        metrics = extract_metrics(summary, mapping)
        if all(value is None for value in metrics.values()):
            LOGGER.info("Skipping %s for %s due to missing metrics.", model_name, ticker)
            continue
        results.append({"ticker": ticker, "model_name": model_name, **metrics})
    return results


def main(cmd_args: Optional[List[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)
    ensure_directories(REPORTS_DIR, FIGURES_DIR)

    feature_df = load_features(ns.features_path)
    records: List[Dict[str, object]] = []
    for ticker in ns.tickers:
        records.extend(evaluate_ticker(ticker, feature_df))

    if not records:
        raise RuntimeError("No metrics computed; check input data.")

    metrics_df = pd.DataFrame(records)
    output_csv = ns.output_csv
    output_md = ns.output_md
    figure_path = FIGURES_DIR / "per_asset_directional_accuracy.png"
    if ns.tag:
        output_csv = output_csv.with_stem(f"{output_csv.stem}_{ns.tag}")
        output_md = output_md.with_stem(f"{output_md.stem}_{ns.tag}")
        figure_path = figure_path.with_stem(f"{figure_path.stem}_{ns.tag}")

    metrics_df.to_csv(output_csv, index=False)

    markdown_path = output_md
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_lines = [
        "# Per-Asset Evaluation Metrics",
        "",
        dataframe_to_markdown(metrics_df),
        "",
        "Generated with consistent walk-forward evaluation across all models.",
    ]
    markdown_path.write_text("\n".join(markdown_lines))

    plot_directional_accuracy(metrics_df, figure_path)
    LOGGER.info("Saved per-asset metrics to %s and %s", output_csv, markdown_path)


if __name__ == "__main__":
    main()
