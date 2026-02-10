"""Per-stock evaluation for the optional S&P 500 equity universe."""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.models.iteration1_1_svr import run_iteration as run_iteration_1_1
from src.models.iteration1_baseline import run_iteration as run_iteration_1
from src.models.iteration2_1_lightgbm import run_iteration as run_iteration_2_1
from src.models.iteration2_ensemble import run_iteration as run_iteration_2
from src.evaluation.walkforward import aggregate_metrics
from src.utils import FIGURES_DIR, PROCESSED_DIR, REFERENCE_DIR, REPORTS_DIR, ensure_directories

LOGGER = logging.getLogger(__name__)

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
    parser = argparse.ArgumentParser(description="Per-stock evaluation for the sampled equity universe.")
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="Path to equity_universe_*.txt. Defaults to equity_universe_20.txt or the newest match.",
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
        default=REPORTS_DIR / "per_asset_equity_metrics.csv",
        help="Where to write the metrics CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORTS_DIR / "per_asset_equity_metrics.md",
        help="Where to write the markdown summary.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to append to output filenames and figures.",
    )
    return parser.parse_args(args)


def resolve_universe_file(candidate: Optional[Path]) -> Path:
    if candidate and candidate.exists():
        return candidate
    default = REFERENCE_DIR / "equity_universe_20.txt"
    if default.exists():
        return default
    matches = sorted(REFERENCE_DIR.glob("equity_universe_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError("No equity universe file found. Run download_equity_universe first.")
    return matches[0]


def read_universe(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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


def evaluate_ticker(ticker: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    ticker_df = df[df["ticker"] == ticker].reset_index(drop=True)
    if ticker_df.empty:
        LOGGER.warning("No data available for %s; skipping.", ticker)
        return []

    results: List[Dict[str, object]] = []
    for model_name, runner, mapping in MODEL_CONFIGS:
        metrics_df, _ = runner(data=ticker_df, generate_reports=False, ticker=ticker)
        summary = aggregate_metrics(metrics_df.to_dict("records"))
        metrics = extract_metrics(summary, mapping)
        if all(value is None for value in metrics.values()):
            LOGGER.info("Skipping %s for %s due to missing metrics.", model_name, ticker)
            continue
        results.append({"ticker": ticker, "model_name": model_name, **metrics})
    return results


def plot_average_directional_accuracy(df: pd.DataFrame, output_path: Path) -> None:
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
    ax.set_title("Average Directional Accuracy Across Equity Universe")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    da_df = df.dropna(subset=["directional_accuracy"])
    if da_df.empty:
        LOGGER.warning("No directional accuracy values available for heatmap.")
        return

    pivot = da_df.pivot(index="ticker", columns="model_name", values="directional_accuracy")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Directional Accuracy by Ticker and Model")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main(cmd_args: Optional[List[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)
    ensure_directories(REPORTS_DIR, FIGURES_DIR)

    universe_path = resolve_universe_file(ns.universe_file)
    tickers = read_universe(universe_path)
    feature_df = load_features(ns.features_path)

    records: List[Dict[str, object]] = []
    for ticker in tickers:
        records.extend(evaluate_ticker(ticker, feature_df))
        gc.collect()

    if not records:
        raise RuntimeError("No metrics computed; check input data and tickers.")

    metrics_df = pd.DataFrame(records)

    output_csv = ns.output_csv
    output_md = ns.output_md
    avg_da_fig = FIGURES_DIR / "equity_avg_directional_accuracy.png"
    heatmap_fig = FIGURES_DIR / "equity_da_heatmap.png"
    if ns.tag:
        output_csv = output_csv.with_stem(f"{output_csv.stem}_{ns.tag}")
        output_md = output_md.with_stem(f"{output_md.stem}_{ns.tag}")
        avg_da_fig = avg_da_fig.with_stem(f"{avg_da_fig.stem}_{ns.tag}")
        heatmap_fig = heatmap_fig.with_stem(f"{heatmap_fig.stem}_{ns.tag}")

    metrics_df.to_csv(output_csv, index=False)

    markdown_path = output_md
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_lines = [
        "# Equity Universe Per-Stock Metrics",
        "",
        f"_Universe file: {universe_path.name}_",
        "",
        dataframe_to_markdown(metrics_df),
    ]
    markdown_path.write_text("\n".join(markdown_lines))

    plot_average_directional_accuracy(metrics_df, avg_da_fig)
    plot_heatmap(metrics_df, heatmap_fig)
    LOGGER.info("Saved equity metrics to %s and %s", output_csv, markdown_path)


if __name__ == "__main__":
    main()
