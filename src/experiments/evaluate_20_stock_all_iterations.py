"""Run per-ticker evaluation across iterations 1, 1.1, 2, 2.1, 3, and 5."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.models.iteration1_1_svr import run_iteration as run_iteration_1_1
from src.models.iteration1_baseline import run_iteration as run_iteration_1
from src.models.iteration2_1_lightgbm import run_iteration as run_iteration_2_1
from src.models.iteration2_ensemble import run_iteration as run_iteration_2
from src.models.iteration3_lstm import run_iteration as run_iteration_3
from src.models.iteration5_meta_ensemble import run_iteration as run_iteration_5
from src.utils import PROCESSED_DIR, PROJECT_ROOT, REFERENCE_DIR, ensure_directories

LOGGER = logging.getLogger(__name__)

IterationRunner = Callable[..., Dict[str, float]]

ITERATION_CONFIGS = [
    (
        "1",
        "Iteration 1",
        run_iteration_1,
        {
            "rmse": "rmse_linear_mean",
            "mae": "mae_linear_mean",
            "r2": "r2_linear_mean",
            "directional_accuracy": "directional_accuracy_linear_mean",
        },
    ),
    (
        "1.1",
        "Iteration 1.1",
        run_iteration_1_1,
        {
            "rmse": "rmse_svr_mean",
            "mae": "mae_svr_mean",
            "r2": "r2_svr_mean",
            "directional_accuracy": "directional_accuracy_svr_mean",
        },
    ),
    (
        "2",
        "Iteration 2",
        run_iteration_2,
        {
            "rmse": "rmse_rf_mean",
            "mae": "mae_rf_mean",
            "r2": "r2_rf_mean",
            "directional_accuracy": "directional_accuracy_rf_mean",
        },
    ),
    (
        "2.1",
        "Iteration 2.1",
        run_iteration_2_1,
        {
            "rmse": "rmse_lgbm_mean",
            "mae": "mae_lgbm_mean",
            "r2": "r2_lgbm_mean",
            "directional_accuracy": "directional_accuracy_lgbm_mean",
        },
    ),
    (
        "3",
        "Iteration 3",
        run_iteration_3,
        {
            "rmse": "rmse_lstm_mean",
            "mae": "mae_lstm_mean",
            "r2": "r2_lstm_mean",
            "directional_accuracy": "directional_accuracy_lstm_mean",
        },
    ),
    (
        "5",
        "Iteration 5",
        run_iteration_5,
        {
            "rmse": "rmse_meta_mean",
            "mae": "mae_meta_mean",
            "r2": "r2_meta_mean",
            "directional_accuracy": "directional_accuracy_meta_mean",
        },
    ),
]


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 20-stock universe across all iterations.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_DIR / "model_features.parquet",
        help="Path to model features parquet file.",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="Path to equity_universe_20.txt. Defaults to equity_universe_20.txt or the newest match.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "20_stock",
        help="Directory for CSV + plots.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional list of tickers to override the universe file.",
    )
    return parser.parse_args(args)


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model features not found at {path}. Run feature engineering first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def resolve_universe_file(candidate: Optional[Path]) -> Optional[Path]:
    if candidate and candidate.exists():
        return candidate
    default = REFERENCE_DIR / "equity_universe_20.txt"
    if default.exists():
        return default
    matches = sorted(REFERENCE_DIR.glob("equity_universe_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_universe(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def select_tickers(feature_df: pd.DataFrame, universe_path: Optional[Path], override: Optional[List[str]]) -> List[str]:
    if override:
        return override
    if universe_path:
        tickers = read_universe(universe_path)
    else:
        tickers = sorted(feature_df["ticker"].unique().tolist())
    if len(tickers) > 20:
        tickers = tickers[:20]
    return tickers


def extract_metrics(summary: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, Optional[float]]:
    return {metric: summary.get(key) for metric, key in mapping.items()}


def slugify_iteration(iteration_id: str) -> str:
    return iteration_id.replace(".", "_")


def plot_accuracy_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot(index="ticker", columns="iteration", values="directional_accuracy")
    if pivot.empty:
        LOGGER.warning("No directional accuracy values available for heatmap.")
        return
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Directional Accuracy by Ticker and Iteration")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_accuracy_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        LOGGER.warning("No directional accuracy values available for boxplot.")
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="iteration", y="directional_accuracy")
    plt.title("Directional Accuracy Distribution by Iteration")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_iteration_bars(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        LOGGER.warning("No directional accuracy values available for bar charts.")
        return
    for iteration, group in df.groupby("iteration", sort=False):
        fig, ax = plt.subplots(figsize=(12, 6))
        group = group.sort_values("ticker")
        ax.bar(group["ticker"], group["directional_accuracy"], color="steelblue")
        ax.set_title(f"Directional Accuracy by Ticker - Iteration {iteration}")
        ax.set_ylabel("Directional Accuracy")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        filename = output_dir / f"accuracy_by_ticker_iter_{slugify_iteration(iteration)}.png"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def evaluate_ticker(ticker: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    ticker_df = df[df["ticker"] == ticker].reset_index(drop=True)
    if ticker_df.empty:
        LOGGER.warning("No data available for %s; skipping.", ticker)
        return []

    records: List[Dict[str, object]] = []
    for iteration_id, iteration_label, runner, mapping in ITERATION_CONFIGS:
        summary = runner(df=ticker_df, generate_reports=False, ticker=ticker)
        metrics = extract_metrics(summary, mapping)
        if all(value is None for value in metrics.values()):
            LOGGER.info("Skipping iteration %s for %s due to missing metrics.", iteration_id, ticker)
            continue
        records.append(
            {
                "ticker": ticker,
                "iteration": iteration_id,
                "iteration_label": iteration_label,
                **metrics,
            }
        )
    return records


def main(cmd_args: Optional[List[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)

    feature_df = load_features(ns.features_path)
    universe_path = resolve_universe_file(ns.universe_file)
    tickers = select_tickers(feature_df, universe_path, ns.tickers)
    ensure_directories(ns.output_dir)

    records: List[Dict[str, object]] = []
    for ticker in tickers:
        records.extend(evaluate_ticker(ticker, feature_df))

    if not records:
        raise RuntimeError("No metrics computed; check input data and tickers.")

    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.sort_values(["iteration", "ticker"]).reset_index(drop=True)

    output_csv = ns.output_dir / "ticker_iteration_metrics.csv"
    metrics_df.to_csv(output_csv, index=False)

    heatmap_path = ns.output_dir / "accuracy_heatmap.png"
    boxplot_path = ns.output_dir / "accuracy_boxplot.png"
    plot_accuracy_heatmap(metrics_df, heatmap_path)
    plot_accuracy_boxplot(metrics_df, boxplot_path)
    plot_iteration_bars(metrics_df, ns.output_dir)

    LOGGER.info("Saved metrics to %s", output_csv)


if __name__ == "__main__":
    main()
